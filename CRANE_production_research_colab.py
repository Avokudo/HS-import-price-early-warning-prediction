# %% [markdown]
# # CRANE Production Research Notebook
#
# 국가 간 상대가격 비교를 활용한 수입단가 이상충격 조기경보 분석.
#
# 신청서에 적힌 데이터 출처를 기준으로 구성한다.
#
# - 필수: 관세청 `관세청_품목별 국가별 수출입실적(GW)`
# - 선택: 한국은행 ECOS 환율 및 가격지표
# - 선택: 통계청 KOSIS 물가/산업 보조통계
# - 선택: TRASS, UN Comtrade, ITC 등 외부 HS-국가 통계 CSV
#
# 핵심 아이디어:
#
# 1. 같은 HS 코드와 같은 월의 여러 공급국을 하나의 상대가격 네트워크로 본다.
# 2. 각 국가의 로그 수입단가가 HS-월 가격분포에서 얼마나 벗어나는지 측정한다.
# 3. 다음 달 금액가중 상대가격 이상충격을 예측한다.
# 4. 단일 금액회귀만 쓰지 않고, 발생확률, 충격규모, 구조위험점수를 분리해 최종 경보점수를 만든다.

# %%
import importlib.util
import os
import sys
import time
import xml.etree.ElementTree as ET
from getpass import getpass
from pathlib import Path
from urllib.parse import unquote


def ensure_package(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


ensure_package("sklearn", "scikit-learn")
ensure_package("lightgbm")
ensure_package("openpyxl")
ensure_package("networkx")

# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

pd.set_option("display.max_columns", 240)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")

RANDOM_STATE = 123
EPS = 1e-6

Path("data_raw").mkdir(exist_ok=True)
Path("data_processed").mkdir(exist_ok=True)
Path("output_production").mkdir(exist_ok=True)
Path("figures_production").mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Configuration
#
# Recommended workflows:
#
# - Full production run: keep `SOURCE_MODE="customs_api"` and enter a Public Data Portal service key.
# - Fast rerun: set `SOURCE_MODE="existing_features"` and upload `crane_features.csv`.
# - Local data: set `SOURCE_MODE="local_trade"` and put paths in `LOCAL_TRADE_PATHS`.
#
# External monthly features are optional. Each file should have a date/month column and numeric variables.

# %%
SOURCE_MODE = "customs_api"  # customs_api, existing_features, local_trade, auto

LOCAL_FEATURE_PATHS = [
    "crane_features.csv",
    "output/crane_features.csv",
    "output_production/crane_features_production.csv",
]

LOCAL_TRADE_PATHS = [
    "data_processed/trade_clean.csv",
    "customs_raw.csv",
    "data_raw/customs_raw.csv",
]

EXTERNAL_MONTHLY_PATHS = [
    # Example: "macro_monthly.csv"
]

# Customs API defaults. HS codes can be 2, 4, 6, or 10 digit. For production,
# collect all HS chapters 01-99. The API often returns lower-level HS rows even
# when a parent HS chapter/code is used.
START_YYYYMM = "202001"
END_YYYYMM = "202412"
HS_SCOPE = "all_chapters"  # all_chapters, selected
HS_CODES = [f"{i:02d}" for i in range(1, 100)]
COUNTRY_CODES = [None]  # None means all countries if the API accepts omitted cntyCd.
CUSTOMS_CACHE_DIR = "data_raw/customs_api_cache"
OVERWRITE_CUSTOMS_CACHE = False
CUSTOMS_SLEEP_SEC = 0.25
CUSTOMS_MAX_RETRIES = 3
CUSTOMS_MAX_PAGES_PER_JOB = None  # set small int for API debugging only

# BOK ECOS optional augmentation. USD/KRW monthly average is included as a
# working default from ECOS table 731Y004.
USE_ECOS_USDKRW = False
ECOS_START_YYYYMM = START_YYYYMM
ECOS_END_YYYYMM = END_YYYYMM

# Model settings.
ANOMALY_K = 1.5
MIN_COUNTRIES_PER_HS_MONTH = 3
TRAIN_END = "2023-01-01"
VALID_END = "2024-01-01"
MATERIAL_QUANTILE = 0.90

# Top-K values used in validation and charts.
K_VALUES = [10, 30, 50, 100, 200]

# Exploratory HS-to-HS propagation settings. Full HS coverage can contain many
# HS codes, so the visualization is restricted to the most relevant HS nodes.
RISK_PROPAGATION_ENABLED = True
RISK_PROPAGATION_MIN_MONTHS = 6
RISK_PROPAGATION_MAX_LAG = 3
RISK_PROPAGATION_CORR_THRESHOLD = 0.45
RISK_PROPAGATION_MAX_NODES = 40
RISK_PROPAGATION_NODE_SELECTION = "actual_amount"  # actual_amount, import_value, warning

# %% [markdown]
# ## 2. Model in plain language
#
# For each cell `(h, c, t)`, where `h` is HS code, `c` is partner country, and
# `t` is month:
#
# - Unit price: `P_hct = import_value_hct / import_weight_hct`
# - Log price: `p_hct = log(P_hct)`
# - HS-month robust center: `median_c(p_hct)`
# - HS-month robust scale: `MAD_c(p_hct)`
# - Relative deviation: `rel_z = (p_hct - median) / MAD`
# - Target: `current_import_value * max(abs(next_month_rel_z) - k, 0)`
#
# The final operational score separates three questions:
#
# 1. Probability: will a material next-month anomaly happen?
# 2. Severity: if it happens, how large is the amount?
# 3. Structure: is the current cell already isolated, persistent, concentrated, or idiosyncratic?
#
# The final warning list uses a validation-selected hybrid of probability,
# expected amount, structural risk, and current excess relative-price amount.

# %% [markdown]
# ## 3. Flexible data loaders

# %%
def read_table_auto(path_or_url):
    path_or_url = str(path_or_url)
    lower = path_or_url.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(path_or_url)
        return pd.read_csv(path_or_url)
    path = Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".zip"):
        return pd.read_csv(path, compression="zip")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def maybe_colab_upload():
    """Prompt for file upload when running in Colab and no local data file exists."""
    try:
        from google.colab import files as colab_files
    except Exception:
        return None, None

    print(
        "No local data file found in this Colab runtime.\n"
        "Upload one of these files: crane_features.csv, trade_clean.csv, customs_raw.csv, or an Excel/CSV trade table."
    )
    uploaded = colab_files.upload()
    if not uploaded:
        return None, None
    first_name = next(iter(uploaded.keys()))
    print("Uploaded:", first_name)
    return first_name, read_table_auto(first_name)


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def pick_column(df, candidates):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def standardize_trade_dataframe(df):
    """Standardize Customs/API/local trade data to HS-country-month cells."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    col_date = pick_column(out, ["date", "ym", "year_month", "yearmonth", "time", "TIME"])
    col_year = pick_column(out, ["year", "YEAR"])
    col_hs = pick_column(out, ["hs_code", "hsCd", "hscd", "hs", "hsSgn", "cmdCode"])
    col_country = pick_column(out, ["country_code", "statCd", "cntyCd", "partnerCode", "partner_code"])
    col_country_name = pick_column(out, ["country_name", "statCdCntnKor1", "partnerDesc", "partner_name"])
    col_hs_name = pick_column(out, ["hs_name", "statKor", "cmdDesc", "commodity"])
    col_value = pick_column(out, ["import_value", "impDlr", "importValue", "tradeValue", "cifvalue"])
    col_weight = pick_column(out, ["import_weight", "impWgt", "netWgt", "net_weight", "weight"])

    if col_hs is None or col_country is None or col_value is None or col_weight is None:
        raise KeyError(
            "Could not identify required columns. Need HS, country, import value, import weight."
        )

    if col_date is not None:
        date_raw = out[col_date].astype(str).str.strip()
    elif col_year is not None:
        date_raw = out[col_year].astype(str).str.strip()
    else:
        raise KeyError("Could not identify date/year_month column.")

    date_raw = date_raw.str.replace(".", "", regex=False).str.replace("-", "", regex=False)
    date_raw = date_raw.str.extract(r"(\d{6})", expand=False)
    out["date"] = pd.to_datetime(date_raw + "01", format="%Y%m%d", errors="coerce")

    out["hs_code"] = out[col_hs].astype(str).str.strip()
    out["country_code"] = out[col_country].astype(str).str.strip()
    out["country_name"] = (
        out[col_country_name].astype(str).str.strip() if col_country_name else out["country_code"]
    )
    out["hs_name"] = out[col_hs_name].astype(str).str.strip() if col_hs_name else out["hs_code"]
    out["import_value"] = clean_numeric(out[col_value])
    out["import_weight"] = clean_numeric(out[col_weight])

    out = out[
        out["date"].notna()
        & out["hs_code"].notna()
        & out["country_code"].notna()
        & (out["hs_code"] != "-")
        & (out["country_code"] != "-")
        & out["import_value"].notna()
        & out["import_weight"].notna()
        & (out["import_value"] > 0)
        & (out["import_weight"] > 0)
    ].copy()

    agg = (
        out.groupby(["date", "hs_code", "country_code"], as_index=False)
        .agg(
            country_name=("country_name", "first"),
            hs_name=("hs_name", "first"),
            import_value=("import_value", "sum"),
            import_weight=("import_weight", "sum"),
        )
        .sort_values(["hs_code", "country_code", "date"])
    )
    agg["year_month"] = agg["date"].dt.strftime("%Y%m")
    agg["year"] = agg["date"].dt.year
    agg["month"] = agg["date"].dt.month
    agg["unit_price"] = agg["import_value"] / agg["import_weight"]
    agg["log_price"] = np.log(agg["unit_price"])
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_price"])
    return agg.reset_index(drop=True)


def make_yearly_periods(start_yyyymm, end_yyyymm):
    start_year = int(start_yyyymm[:4])
    end_year = int(end_yyyymm[:4])
    periods = []
    for year in range(start_year, end_year + 1):
        start_month = int(start_yyyymm[4:6]) if year == start_year else 1
        end_month = int(end_yyyymm[4:6]) if year == end_year else 12
        periods.append((f"{year}{start_month:02d}", f"{year}{end_month:02d}"))
    return periods


def resolve_hs_codes():
    if HS_SCOPE == "all_chapters":
        return [f"{i:02d}" for i in range(1, 100)]
    if HS_SCOPE == "selected":
        return [str(x).zfill(2) if str(x).isdigit() and len(str(x)) == 1 else str(x) for x in HS_CODES]
    raise ValueError("HS_SCOPE must be 'all_chapters' or 'selected'.")


def safe_cache_name(hs_code, country_code, start_yyyymm, end_yyyymm):
    country = country_code if country_code not in (None, "") else "ALL"
    return f"customs_hs_{hs_code}_country_{country}_{start_yyyymm}_{end_yyyymm}.csv"


def call_customs_api_xml(
    service_key,
    start_yyyymm,
    end_yyyymm,
    hs_code=None,
    country_code=None,
    page_no=1,
    num_rows=10000,
):
    base_url = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
    params = {
        "serviceKey": service_key,
        "strtYymm": start_yyyymm,
        "endYymm": end_yyyymm,
        "pageNo": page_no,
        "numOfRows": num_rows,
    }
    if hs_code:
        params["hsSgn"] = str(hs_code)
    if country_code:
        params["cntyCd"] = str(country_code)
    response = requests.get(base_url, params=params, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(f"Customs HTTP {response.status_code}: {response.text[:300]}")
    return response.text


def parse_customs_xml(xml_text):
    root = ET.fromstring(xml_text)
    result_code = root.findtext(".//resultCode")
    result_msg = root.findtext(".//resultMsg")
    if result_code not in (None, "00"):
        raise RuntimeError(f"Customs API resultCode={result_code}, resultMsg={result_msg}")
    total_text = root.findtext(".//totalCount")
    total_count = int(total_text) if total_text else None
    rows = [{child.tag: child.text for child in item} for item in root.findall(".//item")]
    return pd.DataFrame(rows), total_count


def collect_customs_api(
    service_key,
    hs_codes,
    start_yyyymm,
    end_yyyymm,
    country_codes=(None,),
    cache_dir=CUSTOMS_CACHE_DIR,
    overwrite_cache=OVERWRITE_CUSTOMS_CACHE,
    sleep_sec=CUSTOMS_SLEEP_SEC,
    max_retries=CUSTOMS_MAX_RETRIES,
    max_pages_per_job=CUSTOMS_MAX_PAGES_PER_JOB,
):
    periods = make_yearly_periods(start_yyyymm, end_yyyymm)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    failures = []
    for hs in hs_codes:
        for country in country_codes:
            for start, end in periods:
                cache_path = cache_dir / safe_cache_name(hs, country, start, end)
                if cache_path.exists() and not overwrite_cache:
                    try:
                        cached = read_table_auto(cache_path)
                        frames.append(cached)
                        print(f"Cache hit: hs={hs}, country={country or 'ALL'}, {start}-{end}, rows={len(cached)}")
                        continue
                    except Exception as err:
                        print("Cache read failed, refetching:", cache_path, err)

                page = 1
                job_pages = []
                while True:
                    last_error = None
                    try:
                        for attempt in range(1, max_retries + 1):
                            try:
                                xml_text = call_customs_api_xml(
                                    service_key, start, end, hs, country, page_no=page
                                )
                                df_page, total_count = parse_customs_xml(xml_text)
                                break
                            except Exception as err:
                                last_error = err
                                wait = sleep_sec * attempt
                                print(f"Retry {attempt}/{max_retries}: hs={hs}, {start}-{end}, page={page}, error={err}")
                                time.sleep(wait)
                        else:
                            raise RuntimeError(last_error)

                        if df_page.empty:
                            break
                        df_page["query_hs_code"] = hs
                        df_page["query_country_code"] = country if country else ""
                        df_page["query_start_yyyymm"] = start
                        df_page["query_end_yyyymm"] = end
                        df_page["query_page_no"] = page
                        job_pages.append(df_page)
                        print(f"Customs hs={hs}, country={country or 'ALL'}, {start}-{end}, page={page}, rows={len(df_page)}, total={total_count}")

                        if max_pages_per_job is not None and page >= max_pages_per_job:
                            print("Stopped by CUSTOMS_MAX_PAGES_PER_JOB for debugging.")
                            break
                        if total_count is None or page * 10000 >= total_count:
                            break
                        page += 1
                        time.sleep(sleep_sec)
                    except Exception as err:
                        failures.append(
                            {
                                "hs_code": hs,
                                "country_code": country,
                                "start_yyyymm": start,
                                "end_yyyymm": end,
                                "page": page,
                                "error": str(err),
                            }
                        )
                        print("Customs failure:", failures[-1])
                        break
                if job_pages:
                    job_df = pd.concat(job_pages, ignore_index=True)
                    job_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
                    frames.append(job_df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    failed = pd.DataFrame(failures)
    raw.to_csv("data_raw/customs_raw_production.csv", index=False, encoding="utf-8-sig")
    failed.to_csv("data_raw/customs_failed_jobs_production.csv", index=False, encoding="utf-8-sig")
    return raw, failed


def ecos_statistic_search(api_key, stat_code, cycle, start, end, *items):
    item_path = "/".join(str(x) for x in items if x not in (None, ""))
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/{stat_code}/{cycle}/{start}/{end}"
    if item_path:
        url += "/" + item_path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "StatisticSearch" not in data:
        raise RuntimeError(str(data)[:500])
    rows = data["StatisticSearch"].get("row", [])
    return pd.DataFrame(rows)


def load_ecos_usdkrw(api_key, start_yyyymm, end_yyyymm):
    # 731Y004: monthly exchange rate, 0000001 USD/KRW, 0000100 average.
    raw = ecos_statistic_search(api_key, "731Y004", "M", start_yyyymm, end_yyyymm, "0000001", "0000100")
    out = raw[["TIME", "DATA_VALUE"]].copy()
    out["date"] = pd.to_datetime(out["TIME"] + "01", format="%Y%m%d", errors="coerce")
    out["usdkrw"] = pd.to_numeric(out["DATA_VALUE"], errors="coerce")
    out = out.dropna(subset=["date", "usdkrw"]).sort_values("date")
    out["usdkrw_growth"] = np.log(out["usdkrw"]).diff()
    out["usdkrw_momentum_3m"] = out["usdkrw_growth"].rolling(3, min_periods=2).mean()
    out["usdkrw_volatility_3m"] = out["usdkrw_growth"].rolling(3, min_periods=2).std()
    return out[["date", "usdkrw", "usdkrw_growth", "usdkrw_momentum_3m", "usdkrw_volatility_3m"]]


def load_external_monthly_features(paths):
    frames = []
    for path in paths:
        if not Path(path).exists():
            print("External monthly file not found, skipped:", path)
            continue
        df = read_table_auto(path)
        date_col = pick_column(df, ["date", "ym", "year_month", "time", "TIME"])
        if date_col is None:
            print("No date column in external file, skipped:", path)
            continue
        tmp = df.copy()
        raw = tmp[date_col].astype(str).str.replace(".", "", regex=False).str.replace("-", "", regex=False)
        ym = raw.str.extract(r"(\d{6})", expand=False)
        tmp["date"] = pd.to_datetime(ym + "01", format="%Y%m%d", errors="coerce")
        numeric_cols = [c for c in tmp.columns if c != "date" and pd.api.types.is_numeric_dtype(tmp[c])]
        if not numeric_cols:
            for c in tmp.columns:
                if c not in [date_col, "date"]:
                    tmp[c] = pd.to_numeric(tmp[c], errors="ignore")
            numeric_cols = [c for c in tmp.columns if c != "date" and pd.api.types.is_numeric_dtype(tmp[c])]
        tmp = tmp[["date"] + numeric_cols].dropna(subset=["date"]).copy()
        prefix = Path(path).stem.lower().replace(" ", "_")
        tmp = tmp.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols})
        frames.append(tmp)
    if not frames:
        return pd.DataFrame({"date": []})
    result = frames[0]
    for nxt in frames[1:]:
        result = result.merge(nxt, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)

# %% [markdown]
# ## 4. Load trade data

# %%
source_log = []

feature_path = first_existing(LOCAL_FEATURE_PATHS)
trade_path = first_existing(LOCAL_TRADE_PATHS)
uploaded_name, uploaded_df = (None, None)

if SOURCE_MODE == "auto" and feature_path is None and trade_path is None:
    uploaded_name, uploaded_df = maybe_colab_upload()

if SOURCE_MODE in ["auto", "existing_features"] and feature_path:
    crane_or_trade = read_table_auto(feature_path)
    crane_or_trade["date"] = pd.to_datetime(crane_or_trade["date"])
    if "next_date" in crane_or_trade.columns:
        crane_or_trade["next_date"] = pd.to_datetime(crane_or_trade["next_date"])
    source_log.append(f"Loaded existing feature file: {feature_path}")
elif SOURCE_MODE in ["auto", "local_trade"] and trade_path:
    raw_local = read_table_auto(trade_path)
    crane_or_trade = standardize_trade_dataframe(raw_local)
    source_log.append(f"Loaded local trade file: {trade_path}")
elif SOURCE_MODE == "auto" and uploaded_df is not None:
    crane_or_trade = uploaded_df.copy()
    if "date" in crane_or_trade.columns:
        crane_or_trade["date"] = pd.to_datetime(crane_or_trade["date"], errors="coerce")
    if "next_date" in crane_or_trade.columns:
        crane_or_trade["next_date"] = pd.to_datetime(crane_or_trade["next_date"], errors="coerce")
    source_log.append(f"Loaded uploaded file from Colab runtime: {uploaded_name}")
elif SOURCE_MODE == "customs_api":
    active_hs_codes = resolve_hs_codes()
    print(
        f"Customs API production collection: HS_SCOPE={HS_SCOPE}, "
        f"HS jobs={len(active_hs_codes)}, period={START_YYYYMM}-{END_YYYYMM}. "
        "This can take a long time for all chapters; cached jobs are reused."
    )
    raw_key = getpass("Public Data Portal service key: ").strip()
    raw = pd.DataFrame()
    last_error = None
    for key in [unquote(raw_key), raw_key]:
        try:
            raw, failed = collect_customs_api(
                key,
                hs_codes=active_hs_codes,
                start_yyyymm=START_YYYYMM,
                end_yyyymm=END_YYYYMM,
                country_codes=COUNTRY_CODES,
            )
            if not raw.empty:
                break
        except Exception as err:
            last_error = err
            print("Key attempt failed:", err)
    if raw.empty:
        raise RuntimeError(f"Customs API collection failed: {last_error}")
    crane_or_trade = standardize_trade_dataframe(raw)
    source_log.append("Collected trade data from Customs API")
else:
    cwd_files = [p.name for p in Path(".").glob("*") if p.is_file()]
    raise FileNotFoundError(
        "No usable data source found. In Colab, upload crane_features.csv when prompted, "
        "or set SOURCE_MODE='customs_api'. Current runtime files: "
        + ", ".join(cwd_files[:30])
    )

print("\n".join(source_log))
print("loaded shape:", crane_or_trade.shape)
display(crane_or_trade.head())


def diagnose_hs_scope(df, label):
    if "hs_code" not in df.columns:
        return
    hs = df["hs_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    chapters = hs.str[:2]
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_hs_codes",
                "unique_hs_chapters",
                "min_chapter",
                "max_chapter",
            ],
            "value": [
                len(df),
                hs.nunique(),
                chapters.nunique(),
                chapters.min(),
                chapters.max(),
            ],
        }
    )
    print(f"HS scope diagnostic: {label}")
    display(summary)
    display(
        chapters.value_counts()
        .sort_index()
        .rename_axis("hs_chapter")
        .reset_index(name="rows")
        .head(120)
    )
    if chapters.nunique() <= 3 or chapters.isin(["85"]).mean() > 0.80:
        print(
            "WARNING: This data is narrow in HS coverage. "
            "For full production, use SOURCE_MODE='customs_api' and HS_SCOPE='all_chapters'."
        )


diagnose_hs_scope(crane_or_trade, "loaded input")

# %% [markdown]
# ## 5. Optional external monthly features

# %%
monthly_features = load_external_monthly_features(EXTERNAL_MONTHLY_PATHS)

if USE_ECOS_USDKRW:
    ecos_key = getpass("BOK ECOS API key: ").strip()
    usdkrw = load_ecos_usdkrw(ecos_key, ECOS_START_YYYYMM, ECOS_END_YYYYMM)
    monthly_features = usdkrw if monthly_features.empty else monthly_features.merge(usdkrw, on="date", how="outer")
    source_log.append("Merged BOK ECOS USD/KRW monthly features")

if not monthly_features.empty:
    print("monthly_features:", monthly_features.shape)
    display(monthly_features.head())
else:
    print("No external monthly features merged.")

# %% [markdown]
# ## 6. CRANE feature engineering

# %%
def robust_mad(x):
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    med = np.nanmedian(arr)
    return np.nanmedian(np.abs(arr - med))


def leave_one_out_median_series(s):
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    if n <= 2:
        return pd.Series(np.nan, index=s.index)
    result = np.empty(n, dtype=float)
    for i in range(n):
        result[i] = np.nanmedian(np.delete(vals, i))
    return pd.Series(result, index=s.index)


def isolation_score_series(s, eps=EPS):
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    if n <= 2:
        return pd.Series(np.nan, index=s.index)
    scale = robust_mad(vals)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(vals)
    scale = max(float(scale), eps)
    dist = np.abs(vals[:, None] - vals[None, :])
    sim = np.exp(-dist / scale)
    np.fill_diagonal(sim, np.nan)
    return pd.Series(1.0 - np.nanmean(sim, axis=1), index=s.index)


def add_crane_features(trade, k=1.5, min_countries=3):
    out = trade.copy()
    if "unit_price" not in out.columns or "log_price" not in out.columns:
        out = standardize_trade_dataframe(out)
    out["date"] = pd.to_datetime(out["date"])
    out["hs_code"] = out["hs_code"].astype(str)
    out["country_code"] = out["country_code"].astype(str)
    out = out.sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)

    group_cols = ["hs_code", "date"]
    grp = out.groupby(group_cols)["log_price"]
    out["ht_country_count"] = grp.transform("count")
    out["ht_median_log_price"] = grp.transform("median")
    out["ht_mad_log_price"] = grp.transform(lambda x: robust_mad(x)).clip(lower=EPS)
    out["rel_z"] = (out["log_price"] - out["ht_median_log_price"]) / out["ht_mad_log_price"]

    rank = grp.rank(method="average")
    out["price_rank_pct"] = np.where(
        out["ht_country_count"] > 1,
        (rank - 1) / (out["ht_country_count"] - 1),
        np.nan,
    )
    out.loc[out["ht_country_count"] < min_countries, ["rel_z", "price_rank_pct"]] = np.nan

    out["loo_median_log_price"] = (
        out.groupby(group_cols)["log_price"].apply(leave_one_out_median_series).reset_index(level=group_cols, drop=True)
    )
    out["counterfactual_dev"] = out["log_price"] - out["loo_median_log_price"]
    out["counterfactual_dev_scaled"] = out["counterfactual_dev"] / out["ht_mad_log_price"]
    out["network_isolation_score"] = (
        out.groupby(group_cols)["log_price"].apply(isolation_score_series).reset_index(level=group_cols, drop=True)
    )

    out["total_hs_month_import"] = out.groupby(group_cols)["import_value"].transform("sum")
    out["import_share_country"] = out["import_value"] / out["total_hs_month_import"]
    out["hhi_hs_month"] = out.groupby(group_cols)["import_share_country"].transform(lambda x: float(np.sum(np.square(x))))
    out["concentration_risk"] = out["import_share_country"] * out["hhi_hs_month"]

    panel = out.groupby(["hs_code", "country_code"], group_keys=False)
    out["lag_date"] = panel["date"].shift(1)
    out["price_return"] = panel["log_price"].diff()
    out["rank_shift"] = out["price_rank_pct"] - panel["price_rank_pct"].shift(1)
    continuous_lag = out["date"].eq(out["lag_date"] + pd.DateOffset(months=1))
    out.loc[~continuous_lag, ["price_return", "rank_shift"]] = np.nan

    out["abs_rank_shift"] = out["rank_shift"].abs()
    out["import_growth"] = panel["import_value"].pct_change()
    out["weight_growth"] = panel["import_weight"].pct_change()
    out["log_import_value"] = np.log(out["import_value"])
    out["log_import_weight"] = np.log(out["import_weight"])
    out["abs_rel_z"] = out["rel_z"].abs()

    out["anomaly_abs"] = (out["rel_z"].abs() > k).astype(float)
    out["anomaly_up"] = (out["rel_z"] > k).astype(float)
    out["anomaly_down"] = (out["rel_z"] < -k).astype(float)

    for col in ["anomaly_abs", "anomaly_up", "anomaly_down", "abs_rel_z"]:
        out[f"{col}_roll3"] = panel[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    out["price_return_roll3"] = panel["price_return"].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    out["price_volatility_roll3"] = panel["price_return"].transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    out["import_growth_roll3"] = panel["import_growth"].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())

    out["return_sign"] = np.sign(out["price_return"])

    def majority_sign(x):
        nonzero = x.dropna()
        nonzero = nonzero[nonzero != 0]
        if len(nonzero) == 0:
            return np.nan
        return 1.0 if (nonzero > 0).mean() >= 0.5 else -1.0

    out["majority_sign"] = out.groupby(group_cols)["return_sign"].transform(majority_sign)
    out["sync_score"] = out.groupby(group_cols)["return_sign"].transform(
        lambda x: np.nan if x.dropna().empty else np.nanmean(x == out.loc[x.index, "majority_sign"])
    )
    out["idiosyncratic_current_score"] = out["abs_rel_z"] * (1 - out["sync_score"].fillna(0))

    stats = (
        out.groupby(group_cols, as_index=False)
        .agg(
            ht_median_for_shift=("log_price", "median"),
            ht_mad_for_shift=("log_price", robust_mad),
            ht_total_import_value=("import_value", "sum"),
        )
        .sort_values(["hs_code", "date"])
    )
    stats["ht_median_return"] = stats.groupby("hs_code")["ht_median_for_shift"].diff()
    stats["ht_mad_return"] = stats.groupby("hs_code")["ht_mad_for_shift"].diff()
    stats["total_import_return"] = stats.groupby("hs_code")["ht_total_import_value"].pct_change()
    stats["price_distribution_shift"] = (
        stats["ht_median_return"].abs() / (stats["ht_mad_for_shift"].abs() + EPS)
        + stats["ht_mad_return"].abs() / (stats["ht_mad_for_shift"].abs() + EPS)
    )
    out = out.merge(
        stats[
            [
                "hs_code",
                "date",
                "ht_median_return",
                "ht_mad_return",
                "total_import_return",
                "price_distribution_shift",
            ]
        ],
        on=group_cols,
        how="left",
    )
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out.replace([np.inf, -np.inf], np.nan)


def add_future_targets(df, k=1.5):
    out = df.copy().sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)
    panel = out.groupby(["hs_code", "country_code"], group_keys=False)
    out["next_date"] = panel["date"].shift(-1)
    out["next_rel_z"] = panel["rel_z"].shift(-1)
    out["next_price_rank_pct"] = panel["price_rank_pct"].shift(-1)
    out["next_network_isolation_score"] = panel["network_isolation_score"].shift(-1)
    out["has_next_month"] = out["next_date"].eq(out["date"] + pd.DateOffset(months=1))

    excess_abs = np.maximum(out["next_rel_z"].abs() - k, 0)
    excess_up = np.maximum(out["next_rel_z"] - k, 0)
    excess_down = np.maximum(-out["next_rel_z"] - k, 0)

    out["future_abs_anomaly"] = np.where(out["has_next_month"], excess_abs, np.nan)
    out["future_up_anomaly"] = np.where(out["has_next_month"], excess_up, np.nan)
    out["future_down_anomaly"] = np.where(out["has_next_month"], excess_down, np.nan)
    out["future_net_anomaly_amount"] = out["import_value"] * out["future_abs_anomaly"]
    out["future_up_anomaly_amount"] = out["import_value"] * out["future_up_anomaly"]
    out["future_down_anomaly_amount"] = out["import_value"] * out["future_down_anomaly"]
    return out


needed_feature_cols = {"rel_z", "network_isolation_score", "future_net_anomaly_amount"}
if needed_feature_cols.issubset(crane_or_trade.columns):
    crane = crane_or_trade.copy()
    print("Existing CRANE features detected. Reusing feature file.")
else:
    trade_clean = standardize_trade_dataframe(crane_or_trade)
    if not monthly_features.empty:
        trade_clean = trade_clean.merge(monthly_features, on="date", how="left")
    crane = add_crane_features(trade_clean, k=ANOMALY_K, min_countries=MIN_COUNTRIES_PER_HS_MONTH)
    crane = add_future_targets(crane, k=ANOMALY_K)

if "future_net_anomaly_amount" not in crane.columns:
    crane = add_future_targets(crane, k=ANOMALY_K)

if not monthly_features.empty:
    existing_external_cols = [c for c in monthly_features.columns if c != "date" and c in crane.columns]
    if not existing_external_cols:
        crane = crane.merge(monthly_features, on="date", how="left")

crane.to_csv("output_production/crane_features_production.csv", index=False, encoding="utf-8-sig")
print("crane shape:", crane.shape)
display(crane.head())
diagnose_hs_scope(crane, "CRANE feature frame")

# %% [markdown]
# ## 7. Data checks and visual coverage

# %%
def save_fig(path):
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    if not path.exists() or path.stat().st_size < 5000:
        raise RuntimeError(f"Figure failed visual sanity check: {path}")
    print("saved figure:", path, "bytes=", path.stat().st_size)


def plot_coverage_heatmap(df):
    coverage = df.pivot_table(
        index="hs_code",
        columns="year_month",
        values="country_code",
        aggfunc="nunique",
        fill_value=0,
    )
    plt.figure(figsize=(min(18, 0.35 * coverage.shape[1] + 5), max(4, 0.35 * coverage.shape[0] + 2)))
    sns.heatmap(coverage, cmap="YlGnBu", linewidths=0.2, linecolor="white")
    plt.title("Data coverage: number of supplier countries by HS-month")
    plt.xlabel("Month")
    plt.ylabel("HS code")
    save_fig("figures_production/01_coverage_heatmap.png")


def plot_target_distribution(df):
    y = df["future_net_anomaly_amount"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(["zero", "positive"], [(y == 0).mean(), (y > 0).mean()], color=["#6C757D", "#2A9D8F"])
    axes[0].set_title("Target zero/positive ratio")
    axes[0].set_ylim(0, 1)
    pos = y[y > 0]
    if len(pos) > 0:
        sns.histplot(np.log1p(pos), bins=50, ax=axes[1], color="#2A9D8F")
    axes[1].set_title("Positive target distribution, log1p scale")
    axes[1].set_xlabel("log1p(future anomaly amount)")
    save_fig("figures_production/02_target_distribution.png")


plot_coverage_heatmap(crane)
plot_target_distribution(crane)

print("Target summary:")
display(crane["future_net_anomaly_amount"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
print("Missing target ratio:", crane["future_net_anomaly_amount"].isna().mean())

# %% [markdown]
# ## 8. Model frame and time split

# %%
base_features = [
    "rel_z",
    "abs_rel_z",
    "counterfactual_dev",
    "counterfactual_dev_scaled",
    "price_rank_pct",
    "rank_shift",
    "abs_rank_shift",
    "network_isolation_score",
    "idiosyncratic_current_score",
    "anomaly_abs_roll3",
    "anomaly_up_roll3",
    "anomaly_down_roll3",
    "abs_rel_z_roll3",
    "price_return",
    "price_return_roll3",
    "price_volatility_roll3",
    "import_growth",
    "import_growth_roll3",
    "weight_growth",
    "import_share_country",
    "hhi_hs_month",
    "concentration_risk",
    "sync_score",
    "ht_median_return",
    "ht_mad_return",
    "total_import_return",
    "price_distribution_shift",
    "log_price",
    "log_import_value",
    "log_import_weight",
    "ht_country_count",
    "month_sin",
    "month_cos",
]

leakage_prefixes = ("next_", "future_")
external_numeric = [
    c
    for c in crane.columns
    if pd.api.types.is_numeric_dtype(crane[c])
    and c not in base_features
    and not c.startswith(leakage_prefixes)
    and c
    not in [
        "import_value",
        "import_weight",
        "unit_price",
        "year",
        "month",
        "has_next_month",
        "anomaly_abs",
        "anomaly_up",
        "anomaly_down",
    ]
]

feature_cols = list(dict.fromkeys(base_features + external_numeric))
feature_cols = [c for c in feature_cols if c in crane.columns]

zero_fill_features = [
    "rank_shift",
    "abs_rank_shift",
    "price_return",
    "price_return_roll3",
    "price_volatility_roll3",
    "import_growth",
    "import_growth_roll3",
    "weight_growth",
    "ht_median_return",
    "ht_mad_return",
    "total_import_return",
    "price_distribution_shift",
    "anomaly_abs_roll3",
    "anomaly_up_roll3",
    "anomaly_down_roll3",
    "abs_rel_z_roll3",
]

target_col = "future_net_anomaly_amount"
model_df = crane.copy()
for col in zero_fill_features:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(0)

model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=feature_cols + [target_col, "date"]).copy()

train_df = model_df[model_df["date"] < pd.to_datetime(TRAIN_END)].copy()
valid_df = model_df[(model_df["date"] >= pd.to_datetime(TRAIN_END)) & (model_df["date"] < pd.to_datetime(VALID_END))].copy()
test_df = model_df[model_df["date"] >= pd.to_datetime(VALID_END)].copy()

if train_df.empty or valid_df.empty or test_df.empty:
    dates = sorted(model_df["date"].dropna().unique())
    valid_cut = dates[int(len(dates) * 0.70)]
    test_cut = dates[int(len(dates) * 0.85)]
    train_df = model_df[model_df["date"] < valid_cut].copy()
    valid_df = model_df[(model_df["date"] >= valid_cut) & (model_df["date"] < test_cut)].copy()
    test_df = model_df[model_df["date"] >= test_cut].copy()
    print("Auto temporal split used:", valid_cut, test_cut)

material_threshold = float(train_df[target_col].quantile(MATERIAL_QUANTILE))
if material_threshold <= 0 and (train_df[target_col] > 0).any():
    material_threshold = float(train_df.loc[train_df[target_col] > 0, target_col].median())

for df in [train_df, valid_df, test_df]:
    df["material_anomaly"] = (df[target_col] >= material_threshold).astype(int)

assert train_df["date"].max() < valid_df["date"].min()
assert valid_df["date"].max() < test_df["date"].min()
assert not any(c.startswith(leakage_prefixes) for c in feature_cols)

print("Features:", len(feature_cols))
print(feature_cols)
print("Material threshold:", material_threshold)
for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    print(
        name,
        df.shape,
        df["date"].min(),
        df["date"].max(),
        "zero=",
        round((df[target_col] == 0).mean(), 4),
        "material_rate=",
        round(df["material_anomaly"].mean(), 4),
    )

# %% [markdown]
# ## 9. Evaluation helpers

# %%
def pct_rank_score(x):
    return pd.Series(x).rank(pct=True).fillna(0.5).to_numpy()


def precision_recall_at_k(ranked_df, label_col, amount_col, k):
    top = ranked_df.head(min(k, len(ranked_df)))
    total_pos = ranked_df[label_col].sum()
    return {
        f"precision_at_{k}": top[label_col].mean(),
        f"recall_at_{k}": np.nan if total_pos == 0 else top[label_col].sum() / total_pos,
        f"lift_at_{k}": np.nan if ranked_df[label_col].mean() == 0 else top[label_col].mean() / ranked_df[label_col].mean(),
        f"captured_amount_at_{k}": top[amount_col].sum(),
        f"zero_actual_count_at_{k}": int((top[amount_col] == 0).sum()),
    }


def evaluate_ranking(df, score, name, amount_like=False):
    score = np.asarray(score, dtype=float)
    if np.isfinite(score).any():
        score = np.where(np.isfinite(score), score, np.nanmedian(score[np.isfinite(score)]))
    else:
        score = np.zeros(len(df))
    label = df["material_anomaly"].to_numpy(dtype=int)
    y = df[target_col].to_numpy(dtype=float)
    ranked = df.assign(score=score).sort_values("score", ascending=False)
    out = {
        "model": name,
        "auc": roc_auc_score(label, score) if len(np.unique(label)) > 1 else np.nan,
        "average_precision": average_precision_score(label, score) if len(np.unique(label)) > 1 else np.nan,
        "base_rate": df["material_anomaly"].mean(),
        "amount_like": amount_like,
    }
    if amount_like:
        pred = np.maximum(score, 0)
        out["rmse"] = float(np.sqrt(mean_squared_error(y, pred)))
        out["mae"] = mean_absolute_error(y, pred)
        out["r2"] = r2_score(y, pred)
    else:
        out["rmse"] = np.nan
        out["mae"] = np.nan
        out["r2"] = np.nan
    for k in K_VALUES:
        out.update(precision_recall_at_k(ranked, "material_anomaly", target_col, k))
    return out


def calibration_by_decile(df, score, name):
    tmp = df.copy()
    tmp["score"] = np.asarray(score, dtype=float)
    tmp["score_decile"] = pd.qcut(tmp["score"].rank(method="first"), 10, labels=False) + 1
    out = (
        tmp.groupby("score_decile", as_index=False)
        .agg(
            n=("score_decile", "size"),
            mean_score=("score", "mean"),
            material_rate=("material_anomaly", "mean"),
            mean_actual_amount=(target_col, "mean"),
            captured_amount=(target_col, "sum"),
        )
        .sort_values("score_decile")
    )
    out["score_name"] = name
    return out


def add_structural_scores(df, amount_score_col):
    out = df.copy()
    out["pred_amount_score"] = pct_rank_score(out[amount_score_col])
    out["current_excess_amount"] = out["import_value"] * np.maximum(out["abs_rel_z"] - ANOMALY_K, 0)
    out["current_excess_score"] = pct_rank_score(out["current_excess_amount"])
    out["idiosyncratic_score"] = pct_rank_score(
        0.55 * out["abs_rel_z"].fillna(0)
        + 0.30 * out["network_isolation_score"].fillna(0)
        + 0.15 * out["abs_rank_shift"].fillna(0)
    )
    out["systemic_score"] = pct_rank_score(
        0.50 * out["sync_score"].fillna(0)
        + 0.30 * out["price_distribution_shift"].fillna(0)
        + 0.20 * out["ht_median_return"].abs().fillna(0)
    )
    out["persistence_score"] = pct_rank_score(out["anomaly_abs_roll3"].fillna(0))
    out["concentration_score"] = pct_rank_score(out["concentration_risk"].fillna(0))
    out["structural_risk_score"] = (
        0.35 * out["idiosyncratic_score"]
        + 0.20 * out["pred_amount_score"]
        + 0.20 * out["current_excess_score"]
        + 0.10 * out["persistence_score"]
        + 0.10 * out["concentration_score"]
        + 0.05 * (1 - out["systemic_score"])
    )
    out["risk_type"] = np.select(
        [
            (out["systemic_score"] >= 0.60) & (out["idiosyncratic_score"] >= 0.60),
            (out["systemic_score"] >= 0.60) & (out["idiosyncratic_score"] < 0.60),
            (out["systemic_score"] < 0.60) & (out["idiosyncratic_score"] >= 0.60),
        ],
        ["common_plus_country_specific", "common_shock", "country_specific_deviation"],
        default="low_or_mixed",
    )
    return out

# %% [markdown]
# ## 10. Models

# %%
X_train = train_df[feature_cols]
X_valid = valid_df[feature_cols]
X_test = test_df[feature_cols]
y_train = train_df[target_col].to_numpy(dtype=float)
y_valid = valid_df[target_col].to_numpy(dtype=float)
y_test = test_df[target_col].to_numpy(dtype=float)
y_train_cls = train_df["material_anomaly"].to_numpy(dtype=int)
y_valid_cls = valid_df["material_anomaly"].to_numpy(dtype=int)

# Baselines.
valid_current_excess = valid_df["import_value"] * np.maximum(valid_df["abs_rel_z"] - ANOMALY_K, 0)
test_current_excess = test_df["import_value"] * np.maximum(test_df["abs_rel_z"] - ANOMALY_K, 0)

# One-stage Tweedie amount model.
tweedie_records = []
tweedie_models = []
for power in [1.3, 1.5, 1.7, 1.9]:
    for leaves in [15, 31, 63]:
        mdl = lgb.LGBMRegressor(
            objective="tweedie",
            metric="tweedie",
            tweedie_variance_power=power,
            num_leaves=leaves,
            min_child_samples=30,
            learning_rate=0.03,
            n_estimators=2500,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        mdl.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="tweedie",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        pred_valid = np.maximum(mdl.predict(X_valid, num_iteration=mdl.best_iteration_), 0)
        rec = evaluate_ranking(valid_df, pred_valid, f"one_stage_tweedie_p{power}_l{leaves}", amount_like=True)
        rec.update({"power": power, "num_leaves": leaves, "best_iteration": mdl.best_iteration_})
        tweedie_records.append(rec)
        tweedie_models.append(mdl)

tweedie_cv = pd.DataFrame(tweedie_records).sort_values(
    ["average_precision", "precision_at_30", "r2"], ascending=[False, False, False]
)
best_tweedie = tweedie_models[int(tweedie_cv.index[0])]
valid_one_stage_amount = np.maximum(best_tweedie.predict(X_valid, num_iteration=best_tweedie.best_iteration_), 0)
test_one_stage_amount = np.maximum(best_tweedie.predict(X_test, num_iteration=best_tweedie.best_iteration_), 0)

display(tweedie_cv.head(10))

# %% [markdown]
# ## 11. Hurdle model
#
# Stage 1 estimates the probability of a material anomaly. Stage 2 estimates the
# amount conditional on material anomaly. Their product is the expected anomaly amount.

# %%
pos = max(y_train_cls.sum(), 1)
neg = max(len(y_train_cls) - y_train_cls.sum(), 1)
scale_pos_weight = neg / pos

clf_records = []
clf_models = []
for leaves in [15, 31, 63]:
    for min_child in [20, 50, 100]:
        clf = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=leaves,
            min_child_samples=min_child,
            learning_rate=0.03,
            n_estimators=2500,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(
            X_train,
            y_train_cls,
            eval_set=[(X_valid, y_valid_cls)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        prob_valid = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
        rec = evaluate_ranking(valid_df, prob_valid, f"classifier_l{leaves}_m{min_child}", amount_like=False)
        rec.update({"num_leaves": leaves, "min_child_samples": min_child, "best_iteration": clf.best_iteration_})
        clf_records.append(rec)
        clf_models.append(clf)

clf_cv = pd.DataFrame(clf_records).sort_values(
    ["average_precision", "precision_at_30", "auc"], ascending=[False, False, False]
)
best_clf = clf_models[int(clf_cv.index[0])]
valid_prob = best_clf.predict_proba(X_valid, num_iteration=best_clf.best_iteration_)[:, 1]
test_prob = best_clf.predict_proba(X_test, num_iteration=best_clf.best_iteration_)[:, 1]
display(clf_cv.head(10))

severity_train = train_df[train_df["material_anomaly"] == 1].copy()
severity_valid = valid_df[valid_df["material_anomaly"] == 1].copy()
if len(severity_train) < 50:
    severity_train = train_df[train_df[target_col] > 0].copy()
if len(severity_valid) < 30:
    severity_valid = valid_df[valid_df[target_col] > 0].copy()

sev_records = []
sev_models = []
for power in [1.3, 1.5, 1.7, 1.9]:
    for leaves in [15, 31]:
        sev = lgb.LGBMRegressor(
            objective="tweedie",
            metric="tweedie",
            tweedie_variance_power=power,
            num_leaves=leaves,
            min_child_samples=10,
            learning_rate=0.03,
            n_estimators=2500,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        sev.fit(
            severity_train[feature_cols],
            severity_train[target_col],
            eval_set=[(severity_valid[feature_cols], severity_valid[target_col])],
            eval_metric="tweedie",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        pred_valid = np.maximum(sev.predict(X_valid, num_iteration=sev.best_iteration_), 0)
        rec = evaluate_ranking(valid_df, pred_valid, f"severity_p{power}_l{leaves}", amount_like=True)
        rec.update({"power": power, "num_leaves": leaves, "best_iteration": sev.best_iteration_})
        sev_records.append(rec)
        sev_models.append(sev)

sev_cv = pd.DataFrame(sev_records).sort_values(["r2", "rmse"], ascending=[False, True])
best_sev = sev_models[int(sev_cv.index[0])]
valid_severity_amount = np.maximum(best_sev.predict(X_valid, num_iteration=best_sev.best_iteration_), 0)
test_severity_amount = np.maximum(best_sev.predict(X_test, num_iteration=best_sev.best_iteration_), 0)
display(sev_cv.head(10))

valid_hurdle_amount_raw = valid_prob * valid_severity_amount
test_hurdle_amount_raw = test_prob * test_severity_amount
scale = y_valid.sum() / valid_hurdle_amount_raw.sum() if valid_hurdle_amount_raw.sum() > 0 else 1.0
scale = float(np.clip(scale, 0.05, 20.0))
valid_hurdle_amount = valid_hurdle_amount_raw * scale
test_hurdle_amount = test_hurdle_amount_raw * scale
print("Hurdle calibration scale:", scale)

# %% [markdown]
# ## 12. Validation-selected hybrid warning score

# %%
valid_scored = valid_df.copy()
valid_scored["one_stage_expected_amount"] = valid_one_stage_amount
valid_scored["material_shock_probability"] = valid_prob
valid_scored["severity_amount_if_material"] = valid_severity_amount
valid_scored["hurdle_expected_amount"] = valid_hurdle_amount
valid_scored = add_structural_scores(valid_scored, "hurdle_expected_amount")

test_scored = test_df.copy()
test_scored["one_stage_expected_amount"] = test_one_stage_amount
test_scored["material_shock_probability"] = test_prob
test_scored["severity_amount_if_material"] = test_severity_amount
test_scored["hurdle_expected_amount"] = test_hurdle_amount
test_scored = add_structural_scores(test_scored, "hurdle_expected_amount")

valid_components = {
    "prob": pct_rank_score(valid_scored["material_shock_probability"]),
    "hurdle_amount": pct_rank_score(valid_scored["hurdle_expected_amount"]),
    "one_stage_amount": pct_rank_score(valid_scored["one_stage_expected_amount"]),
    "structural": valid_scored["structural_risk_score"].to_numpy(),
    "current": valid_scored["current_excess_score"].to_numpy(),
}

weight_rows = []
for w_prob in np.arange(0.0, 1.01, 0.1):
    for w_hurdle in np.arange(0.0, 1.01 - w_prob, 0.1):
        for w_struct in np.arange(0.0, 1.01 - w_prob - w_hurdle, 0.1):
            for w_current in np.arange(0.0, 1.01 - w_prob - w_hurdle - w_struct, 0.1):
                w_one = 1.0 - w_prob - w_hurdle - w_struct - w_current
                if w_one < -1e-9:
                    continue
                score = (
                    w_prob * valid_components["prob"]
                    + w_hurdle * valid_components["hurdle_amount"]
                    + w_struct * valid_components["structural"]
                    + w_current * valid_components["current"]
                    + w_one * valid_components["one_stage_amount"]
                )
                rec = evaluate_ranking(valid_scored, score, "hybrid_valid", amount_like=False)
                rec.update(
                    {
                        "w_prob": round(float(w_prob), 2),
                        "w_hurdle": round(float(w_hurdle), 2),
                        "w_structural": round(float(w_struct), 2),
                        "w_current": round(float(w_current), 2),
                        "w_one_stage": round(float(w_one), 2),
                    }
                )
                weight_rows.append(rec)

weight_search = pd.DataFrame(weight_rows).sort_values(
    ["average_precision", "precision_at_30", "captured_amount_at_30"],
    ascending=[False, False, False],
)
display(weight_search.head(20))

best_weights = weight_search.iloc[0][["w_prob", "w_hurdle", "w_structural", "w_current", "w_one_stage"]]
print("Best validation weights:")
display(best_weights)

test_components = {
    "prob": pct_rank_score(test_scored["material_shock_probability"]),
    "hurdle_amount": pct_rank_score(test_scored["hurdle_expected_amount"]),
    "one_stage_amount": pct_rank_score(test_scored["one_stage_expected_amount"]),
    "structural": test_scored["structural_risk_score"].to_numpy(),
    "current": test_scored["current_excess_score"].to_numpy(),
}

valid_scored["hybrid_warning_score"] = (
    best_weights["w_prob"] * valid_components["prob"]
    + best_weights["w_hurdle"] * valid_components["hurdle_amount"]
    + best_weights["w_structural"] * valid_components["structural"]
    + best_weights["w_current"] * valid_components["current"]
    + best_weights["w_one_stage"] * valid_components["one_stage_amount"]
)
test_scored["hybrid_warning_score"] = (
    best_weights["w_prob"] * test_components["prob"]
    + best_weights["w_hurdle"] * test_components["hurdle_amount"]
    + best_weights["w_structural"] * test_components["structural"]
    + best_weights["w_current"] * test_components["current"]
    + best_weights["w_one_stage"] * test_components["one_stage_amount"]
)

# %% [markdown]
# ## 13. Final validation and model comparison

# %%
comparison = pd.DataFrame(
    [
        evaluate_ranking(test_scored, test_scored["import_value"], "Baseline: import value", amount_like=True),
        evaluate_ranking(test_scored, test_scored["current_excess_amount"], "Baseline: current excess amount", amount_like=True),
        evaluate_ranking(test_scored, test_scored["one_stage_expected_amount"], "One-stage Tweedie expected amount", amount_like=True),
        evaluate_ranking(test_scored, test_scored["material_shock_probability"], "Hurdle stage 1 probability", amount_like=False),
        evaluate_ranking(test_scored, test_scored["hurdle_expected_amount"], "Hurdle expected amount", amount_like=True),
        evaluate_ranking(test_scored, test_scored["structural_risk_score"], "Structural risk score", amount_like=False),
        evaluate_ranking(test_scored, test_scored["hybrid_warning_score"], "Final hybrid warning score", amount_like=False),
    ]
).sort_values(["average_precision", "precision_at_30"], ascending=[False, False])

display(comparison)

calibration = pd.concat(
    [
        calibration_by_decile(test_scored, test_scored["material_shock_probability"], "material_probability"),
        calibration_by_decile(test_scored, test_scored["hurdle_expected_amount"], "hurdle_expected_amount"),
        calibration_by_decile(test_scored, test_scored["hybrid_warning_score"], "hybrid_warning_score"),
    ],
    ignore_index=True,
)
display(calibration)

monthly_backtest = []
for month, g in test_scored.groupby("date"):
    rec = {"date": month, "n": len(g), "base_rate": g["material_anomaly"].mean(), "actual_amount_sum": g[target_col].sum()}
    ranked = g.sort_values("hybrid_warning_score", ascending=False)
    for k in [5, 10, 20]:
        kk = min(k, len(ranked))
        rec[f"precision_at_{k}"] = ranked.head(kk)["material_anomaly"].mean()
        rec[f"captured_amount_at_{k}"] = ranked.head(kk)[target_col].sum()
    monthly_backtest.append(rec)
monthly_backtest = pd.DataFrame(monthly_backtest)
display(monthly_backtest)

# %% [markdown]
# ## 14. Visual diagnostics

# %%
def plot_model_comparison(comp):
    plot_cols = ["model", "average_precision", "precision_at_30", "precision_at_50"]
    tmp = comp[plot_cols].melt(id_vars="model", var_name="metric", value_name="value")
    plt.figure(figsize=(13, 6))
    sns.barplot(data=tmp, y="model", x="value", hue="metric")
    plt.title("Model comparison: ranking quality")
    plt.xlabel("Metric value")
    plt.ylabel("")
    save_fig("figures_production/03_model_comparison.png")


def plot_captured_amount(comp):
    tmp = comp[["model", "captured_amount_at_30", "captured_amount_at_50", "captured_amount_at_100"]].melt(
        id_vars="model", var_name="metric", value_name="captured_amount"
    )
    plt.figure(figsize=(13, 6))
    sns.barplot(data=tmp, y="model", x="captured_amount", hue="metric")
    plt.title("Captured actual anomaly amount by top-K")
    plt.xlabel("Captured actual amount")
    plt.ylabel("")
    save_fig("figures_production/04_captured_amount.png")


def plot_calibration(cal):
    g = sns.FacetGrid(cal, col="score_name", col_wrap=3, sharey=False, height=4)
    g.map_dataframe(sns.lineplot, x="score_decile", y="material_rate", marker="o", color="#2A9D8F")
    g.set_titles("{col_name}")
    g.set_axis_labels("Score decile", "Observed material anomaly rate")
    plt.savefig("figures_production/05_calibration_decile.png", dpi=180, bbox_inches="tight")
    plt.show()
    path = Path("figures_production/05_calibration_decile.png")
    if path.stat().st_size < 5000:
        raise RuntimeError("Calibration figure failed visual sanity check")


def plot_risk_map(df):
    plt.figure(figsize=(9, 7))
    sizes = 30 + 350 * pct_rank_score(df["hurdle_expected_amount"])
    scatter = plt.scatter(
        df["systemic_score"],
        df["idiosyncratic_score"],
        s=sizes,
        c=df["hybrid_warning_score"],
        cmap="viridis",
        alpha=0.65,
        edgecolor="none",
    )
    plt.axvline(0.60, color="gray", linestyle="--", linewidth=1)
    plt.axhline(0.60, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Systemic/common-shock score")
    plt.ylabel("Country-specific relative-deviation score")
    plt.title("CRANE risk map")
    plt.colorbar(scatter, label="Hybrid warning score")
    save_fig("figures_production/06_risk_map.png")


def feature_importance(model, name):
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=imp.head(25), x="gain", y="feature", color="#2A9D8F")
    plt.title(f"Feature importance: {name}")
    plt.xlabel("Gain")
    plt.ylabel("")
    save_fig(f"figures_production/07_feature_importance_{name}.png")
    return imp


def plot_top_case_timeline(df_all, top_df):
    row = top_df.iloc[0]
    case = df_all[(df_all["hs_code"].astype(str) == str(row["hs_code"])) & (df_all["country_code"].astype(str) == str(row["country_code"]))].copy()
    if case.empty:
        return
    case = case.sort_values("date")
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(case["date"], case["rel_z"], marker="o", color="#2A9D8F", label="rel_z")
    ax1.axhline(ANOMALY_K, color="#E76F51", linestyle="--", linewidth=1)
    ax1.axhline(-ANOMALY_K, color="#E76F51", linestyle="--", linewidth=1)
    ax1.set_ylabel("Relative price deviation, rel_z")
    ax1.set_title(f"Top case timeline: HS {row['hs_code']} / {row['country_code']}")
    ax2 = ax1.twinx()
    ax2.bar(case["date"], case["import_value"], alpha=0.22, color="#6C757D", label="import value")
    ax2.set_ylabel("Import value")
    fig.tight_layout()
    plt.savefig("figures_production/08_top_case_timeline.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_network_snapshot(df_all, top_df):
    row = top_df.iloc[0]
    snap = df_all[(df_all["hs_code"].astype(str) == str(row["hs_code"])) & (df_all["date"] == row["date"])].copy()
    if len(snap) < 3:
        return
    snap = snap.sort_values("log_price")
    vals = snap["log_price"].to_numpy()
    scale = max(float(robust_mad(vals)), EPS)
    sim = np.exp(-np.abs(vals[:, None] - vals[None, :]) / scale)
    labels = snap["country_code"].astype(str).tolist()
    plt.figure(figsize=(max(7, 0.35 * len(labels)), max(6, 0.35 * len(labels))))
    sns.heatmap(sim, xticklabels=labels, yticklabels=labels, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"Relative price similarity network heatmap: HS {row['hs_code']}, {row['date'].date()}")
    save_fig("figures_production/09_network_similarity_heatmap.png")


plot_model_comparison(comparison)
plot_captured_amount(comparison)
plot_calibration(calibration)
plot_risk_map(test_scored)
importance_one_stage = feature_importance(best_tweedie, "one_stage_tweedie")
importance_classifier = feature_importance(best_clf, "classifier")
importance_severity = feature_importance(best_sev, "severity")

top_warning_preview = test_scored.sort_values(["hybrid_warning_score", "hurdle_expected_amount"], ascending=False).reset_index(drop=True)
plot_top_case_timeline(crane, top_warning_preview)
plot_network_snapshot(crane, top_warning_preview)

# %% [markdown]
# ## 15. Final warning list and outputs

# %%
top_warning = test_scored.sort_values(["hybrid_warning_score", "hurdle_expected_amount"], ascending=False).reset_index(drop=True)
top_warning["warning_rank"] = np.arange(1, len(top_warning) + 1)

warning_cols = [
    "warning_rank",
    "date",
    "next_date",
    "hs_code",
    "hs_name",
    "country_code",
    "country_name",
    "import_value",
    "import_weight",
    "unit_price",
    "rel_z",
    "next_rel_z",
    "price_rank_pct",
    "rank_shift",
    "network_isolation_score",
    "current_excess_amount",
    "material_shock_probability",
    "severity_amount_if_material",
    "hurdle_expected_amount",
    "one_stage_expected_amount",
    "hybrid_warning_score",
    "structural_risk_score",
    "risk_type",
    "future_net_anomaly_amount",
    "material_anomaly",
]

display(top_warning[warning_cols].head(50))

risk_type_summary = (
    top_warning.groupby("risk_type", as_index=False)
    .agg(
        n=("risk_type", "size"),
        material_rate=("material_anomaly", "mean"),
        actual_amount_mean=(target_col, "mean"),
        actual_amount_sum=(target_col, "sum"),
        avg_warning_score=("hybrid_warning_score", "mean"),
    )
    .sort_values("actual_amount_sum", ascending=False)
)
display(risk_type_summary)

top_warning.to_csv("output_production/crane_final_top_warning.csv", index=False, encoding="utf-8-sig")
comparison.to_csv("output_production/crane_final_model_comparison.csv", index=False, encoding="utf-8-sig")
weight_search.to_csv("output_production/crane_final_weight_search.csv", index=False, encoding="utf-8-sig")
calibration.to_csv("output_production/crane_final_calibration_decile.csv", index=False, encoding="utf-8-sig")
monthly_backtest.to_csv("output_production/crane_final_monthly_backtest.csv", index=False, encoding="utf-8-sig")
tweedie_cv.to_csv("output_production/crane_one_stage_tweedie_cv.csv", index=False, encoding="utf-8-sig")
clf_cv.to_csv("output_production/crane_classifier_cv.csv", index=False, encoding="utf-8-sig")
sev_cv.to_csv("output_production/crane_severity_cv.csv", index=False, encoding="utf-8-sig")
importance_one_stage.to_csv("output_production/feature_importance_one_stage_tweedie.csv", index=False, encoding="utf-8-sig")
importance_classifier.to_csv("output_production/feature_importance_classifier.csv", index=False, encoding="utf-8-sig")
importance_severity.to_csv("output_production/feature_importance_severity.csv", index=False, encoding="utf-8-sig")

print("Saved production outputs:")
for path in sorted(Path("output_production").glob("*.csv")):
    print("-", path)
print("Saved figures:")
for path in sorted(Path("figures_production").glob("*.png")):
    print("-", path)

# %% [markdown]
# ## 16. Exploratory HS-to-HS Risk Propagation Network
#
# 이 섹션은 최종 경보점수를 HS 단위 월별 위험 시계열로 압축한 뒤, 한 HS의 위험 상승이 다른 HS의 위험 상승보다 1-3개월 선행하는지를 탐색합니다.
#
# 중요한 점:
#
# - 이 분석은 인과관계 증명이 아니라 **전파 후보 탐색**입니다.
# - 전체 HS 데이터에서는 HS 코드가 많아 pairwise 계산이 커지므로, 실제 충격금액/수입금액/경보점수 기준 상위 HS만 시각화합니다.
# - 결과는 후속 분석에서 공급망 연결, 산업분류, 품목군 지식과 결합해 해석해야 합니다.

# %%
if RISK_PROPAGATION_ENABLED:
    propagation_df = test_scored.copy()
    propagation_df["date"] = pd.to_datetime(propagation_df["date"])
    propagation_df["month_id"] = propagation_df["date"].dt.to_period("M").astype(str)
    propagation_df["hs_code"] = propagation_df["hs_code"].astype(str)

    hs_month = (
        propagation_df.groupby(["hs_code", "date"], as_index=False)
        .agg(
            hs_warning_mean=("hybrid_warning_score", "mean"),
            hs_prob_mean=("material_shock_probability", "mean"),
            hs_structural_mean=("structural_risk_score", "mean"),
            hs_material_rate=("material_anomaly", "mean"),
            hs_actual_amount=(target_col, "sum"),
            hs_import_value=("import_value", "sum"),
            n_country=("country_code", "nunique"),
        )
    )

    hs_counts = hs_month.groupby("hs_code")["date"].nunique()
    valid_hs = hs_counts[hs_counts >= RISK_PROPAGATION_MIN_MONTHS].index
    hs_month = hs_month[hs_month["hs_code"].isin(valid_hs)].copy()

    node_rank = (
        hs_month.groupby("hs_code", as_index=False)
        .agg(
            mean_warning=("hs_warning_mean", "mean"),
            total_actual_amount=("hs_actual_amount", "sum"),
            total_import_value=("hs_import_value", "sum"),
            mean_material_rate=("hs_material_rate", "mean"),
            n_month=("date", "nunique"),
        )
    )

    if RISK_PROPAGATION_NODE_SELECTION == "actual_amount":
        node_rank = node_rank.sort_values(["total_actual_amount", "mean_warning"], ascending=False)
    elif RISK_PROPAGATION_NODE_SELECTION == "import_value":
        node_rank = node_rank.sort_values(["total_import_value", "mean_warning"], ascending=False)
    elif RISK_PROPAGATION_NODE_SELECTION == "warning":
        node_rank = node_rank.sort_values(["mean_warning", "total_actual_amount"], ascending=False)
    else:
        raise ValueError("RISK_PROPAGATION_NODE_SELECTION must be actual_amount, import_value, or warning.")

    selected_hs = node_rank.head(RISK_PROPAGATION_MAX_NODES)["hs_code"].tolist()
    hs_month = hs_month[hs_month["hs_code"].isin(selected_hs)].copy()

    print("HS propagation nodes used:", len(selected_hs))
    display(node_rank.head(RISK_PROPAGATION_MAX_NODES))

# %% [markdown]
# ### 16.1 HS-Month Risk Matrix
#
# 각 HS 코드의 월별 평균 경보점수(`hs_warning_mean`)를 행렬로 만듭니다. 행은 월, 열은 HS 코드입니다. 결측 월은 해당 HS의 평균값으로 채워서 상관계산이 가능하게 합니다.

# %%
if RISK_PROPAGATION_ENABLED and len(hs_month) > 0:
    risk_col = "hs_warning_mean"
    risk_mat = (
        hs_month.pivot_table(index="date", columns="hs_code", values=risk_col, aggfunc="mean")
        .sort_index()
    )
    risk_mat = risk_mat.apply(lambda x: x.fillna(x.mean()), axis=0)
    risk_mat = risk_mat.dropna(axis=1, how="all")
    print("risk_mat shape:", risk_mat.shape)
    display(risk_mat.head())

    plt.figure(figsize=(min(18, 0.35 * risk_mat.shape[1] + 5), 6))
    sns.heatmap(risk_mat.T, cmap="YlGnBu", cbar_kws={"label": "HS warning score"})
    plt.title("HS-month warning score matrix")
    plt.xlabel("Month")
    plt.ylabel("HS code")
    save_fig("figures_production/10_hs_month_warning_matrix.png")

# %% [markdown]
# ### 16.2 Lagged Correlation Edges
#
# `from_hs`의 위험점수가 `lag`개월 전에 높았을 때 `to_hs`의 현재 위험점수와 얼마나 같이 움직이는지 계산합니다. 양의 상관이 높으면 “전파 후보 edge”로 봅니다.

# %%
if RISK_PROPAGATION_ENABLED and "risk_mat" in globals() and risk_mat.shape[1] >= 2:
    edges = []
    hs_list = list(risk_mat.columns)

    for lag in range(1, RISK_PROPAGATION_MAX_LAG + 1):
        lead = risk_mat.shift(lag)
        current = risk_mat
        for h_from in hs_list:
            x = lead[h_from]
            for h_to in hs_list:
                if h_from == h_to:
                    continue
                y = current[h_to]
                tmp = pd.concat([x, y], axis=1).dropna()
                if len(tmp) < max(5, RISK_PROPAGATION_MIN_MONTHS - lag):
                    continue
                corr = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
                if np.isfinite(corr):
                    edges.append(
                        {
                            "from_hs": h_from,
                            "to_hs": h_to,
                            "lag": lag,
                            "lagged_corr": corr,
                            "abs_corr": abs(corr),
                        }
                    )

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        strong_edges = pd.DataFrame(columns=["from_hs", "to_hs", "lag", "lagged_corr", "abs_corr"])
        print("No lagged HS edges could be computed.")
    else:
        strong_edges = edge_df[edge_df["lagged_corr"] >= RISK_PROPAGATION_CORR_THRESHOLD].copy()
        display(edge_df.sort_values("lagged_corr", ascending=False).head(30))
        print("Strong positive lagged links:", len(strong_edges))
        display(strong_edges.sort_values("lagged_corr", ascending=False).head(30))

# %% [markdown]
# ### 16.3 Sender and Receiver Summary
#
# `out_degree`가 높은 HS는 다른 HS 위험을 선행하는 후보, `in_degree`가 높은 HS는 다른 HS 위험을 뒤따르는 후보로 해석합니다.

# %%
if RISK_PROPAGATION_ENABLED and "strong_edges" in globals():
    if strong_edges.empty:
        sender_receiver = pd.DataFrame(
            columns=["hs_code", "out_degree", "in_degree", "mean_out_corr", "mean_in_corr"]
        )
        print("No strong edges above threshold. Try lowering RISK_PROPAGATION_CORR_THRESHOLD.")
    else:
        sender = (
            strong_edges.groupby("from_hs")
            .agg(out_degree=("to_hs", "count"), mean_out_corr=("lagged_corr", "mean"))
            .rename_axis("hs_code")
            .reset_index()
        )
        receiver = (
            strong_edges.groupby("to_hs")
            .agg(in_degree=("from_hs", "count"), mean_in_corr=("lagged_corr", "mean"))
            .rename_axis("hs_code")
            .reset_index()
        )
        sender_receiver = (
            pd.merge(sender, receiver, on="hs_code", how="outer")
            .fillna({"out_degree": 0, "in_degree": 0, "mean_out_corr": 0, "mean_in_corr": 0})
            .sort_values(["out_degree", "mean_out_corr"], ascending=False)
            .reset_index(drop=True)
        )
        display(sender_receiver.head(40))

# %% [markdown]
# ### 16.4 Propagation Network Visualization
#
# 강한 lagged correlation edge만 방향그래프로 그립니다. 노드 크기와 색은 평균 경보점수이며, edge 두께는 lagged correlation 크기입니다.

# %%
if RISK_PROPAGATION_ENABLED and "strong_edges" in globals():
    hs_node = (
        hs_month.groupby("hs_code", as_index=False)
        .agg(
            mean_warning=(risk_col, "mean"),
            total_actual_amount=("hs_actual_amount", "sum"),
            total_import_value=("hs_import_value", "sum"),
        )
    )
    node_info = hs_node.set_index("hs_code").to_dict("index")

    G = nx.DiGraph()
    for hs in hs_node["hs_code"]:
        G.add_node(hs)
    for _, row in strong_edges.iterrows():
        G.add_edge(row["from_hs"], row["to_hs"], weight=row["lagged_corr"], lag=row["lag"])

    fig, ax = plt.subplots(figsize=(13, 10))
    if G.number_of_edges() == 0:
        ax.text(
            0.5,
            0.5,
            "No strong HS-to-HS links above threshold",
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.axis("off")
    else:
        pos = nx.spring_layout(G, seed=123, k=0.8)
        node_sizes = [500 + 3000 * node_info.get(node, {}).get("mean_warning", 0) for node in G.nodes()]
        node_colors = [node_info.get(node, {}).get("mean_warning", 0) for node in G.nodes()]
        edge_widths = [1.5 + 5 * G[u][v]["weight"] for u, v in G.edges()]

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap="viridis",
            alpha=0.85,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            alpha=0.45,
            arrows=True,
            arrowsize=18,
            edge_color="gray",
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        fig.colorbar(nodes, ax=ax, label="Mean HS warning score")
        ax.axis("off")

    ax.set_title(
        "Exploratory HS risk propagation network\n"
        f"Edges: lagged correlation >= {RISK_PROPAGATION_CORR_THRESHOLD}, "
        f"lag 1-{RISK_PROPAGATION_MAX_LAG} months"
    )
    plt.tight_layout()
    plt.savefig("figures_production/11_hs_risk_propagation_network.png", dpi=180, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### 16.5 Lag Summary and Strongest Relation Heatmap
#
# lag별 강한 연결 수를 보고, 각 HS pair에서 가장 높은 lagged correlation을 heatmap으로 확인합니다.

# %%
if RISK_PROPAGATION_ENABLED and "edge_df" in globals() and not edge_df.empty:
    lag_summary = (
        edge_df.query("lagged_corr > 0")
        .groupby("lag", as_index=False)
        .agg(
            mean_positive_corr=("lagged_corr", "mean"),
            max_corr=("lagged_corr", "max"),
            n_edges_above_threshold=("lagged_corr", lambda x: (x >= RISK_PROPAGATION_CORR_THRESHOLD).sum()),
        )
    )
    display(lag_summary)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=lag_summary, x="lag", y="n_edges_above_threshold", color="#2A9D8F")
    plt.title("Number of strong HS-to-HS risk links by lag")
    plt.xlabel("Lag in months")
    plt.ylabel("Number of strong links")
    save_fig("figures_production/12_hs_links_by_lag.png")

    best_edges = (
        edge_df.sort_values("lagged_corr", ascending=False)
        .groupby(["from_hs", "to_hs"], as_index=False)
        .first()
    )
    heat = best_edges.pivot_table(index="from_hs", columns="to_hs", values="lagged_corr", aggfunc="max")
    plt.figure(figsize=(max(10, 0.35 * heat.shape[1]), max(8, 0.35 * heat.shape[0])))
    sns.heatmap(
        heat,
        cmap="YlGnBu",
        center=0,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Max lagged correlation"},
    )
    plt.title("HS-to-HS maximum lagged risk correlation")
    plt.xlabel("Receiving HS")
    plt.ylabel("Leading HS")
    save_fig("figures_production/13_hs_lagged_corr_heatmap.png")

# %% [markdown]
# ### 16.6 Save Propagation Outputs
#
# 전파 네트워크 분석 결과를 CSV로 저장합니다. 이 결과는 발표자료에서 “품목 간 위험 동조/전파 후보”를 보여주는 보조 분석으로 사용할 수 있습니다.

# %%
if RISK_PROPAGATION_ENABLED:
    if "edge_df" not in globals():
        edge_df = pd.DataFrame(columns=["from_hs", "to_hs", "lag", "lagged_corr", "abs_corr"])
    if "strong_edges" not in globals():
        strong_edges = pd.DataFrame(columns=["from_hs", "to_hs", "lag", "lagged_corr", "abs_corr"])
    if "sender_receiver" not in globals():
        sender_receiver = pd.DataFrame(columns=["hs_code", "out_degree", "in_degree", "mean_out_corr", "mean_in_corr"])

    edge_df.to_csv("output_production/crane_hs_lagged_risk_edges_all.csv", index=False, encoding="utf-8-sig")
    strong_edges.to_csv("output_production/crane_hs_lagged_risk_edges_strong.csv", index=False, encoding="utf-8-sig")
    sender_receiver.to_csv("output_production/crane_hs_sender_receiver_summary.csv", index=False, encoding="utf-8-sig")

    print("Saved HS propagation outputs:")
    print("- output_production/crane_hs_lagged_risk_edges_all.csv")
    print("- output_production/crane_hs_lagged_risk_edges_strong.csv")
    print("- output_production/crane_hs_sender_receiver_summary.csv")

# %% [markdown]
# ## 17. Interpretation template
#
# Use this section after running the notebook.
#
# - Best ranking model: check `crane_final_model_comparison.csv`.
# - Practical warning list: use `crane_final_top_warning.csv`, sorted by `hybrid_warning_score`.
# - Amount interpretation: use `hurdle_expected_amount` and `one_stage_expected_amount` as expected money-impact references, not as exact forecasts.
# - Structural interpretation: use `risk_type`, `idiosyncratic_score`, `systemic_score`, and `current_excess_amount`.
# - Validation: report AUC, AP, Precision@K, Lift@K, captured amount@K, monthly backtest stability, and calibration decile plots.
#
# A good result should satisfy three checks:
#
# 1. `Final hybrid warning score` beats simple `import value` on AP and Precision@K.
# 2. `Structural risk score` or the final hybrid score has high Precision@30/50.
# 3. Captured amount@K remains close to the one-stage amount model, so precision is not gained by ignoring large shocks.
