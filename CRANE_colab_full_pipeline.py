# %% [markdown]
# # CRANE Colab Full Pipeline
#
# Cross-Country Relative Anomaly Network features for early warning of
# HS-country-month import unit price anomaly shocks.
#
# Data source:
# - Korea Customs Service, item-country import/export performance (GW)
# - Public Data Portal OpenAPI: https://www.data.go.kr/data/15100475/openapi.do
#
# Target:
# - future_net_anomaly_amount = current_import_value * max(abs(next_rel_z) - k, 0)
#
# Model:
# - LightGBM Gradient Boosting with Tweedie objective.

# %%
import importlib.util
import os
import sys
import time
import xml.etree.ElementTree as ET
from getpass import getpass
from urllib.parse import unquote


def ensure_package(import_name, pip_name=None):
    """Install a package in Colab if it is missing."""
    if importlib.util.find_spec(import_name) is None:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


ensure_package("lightgbm")
ensure_package("shap")

# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import shap

from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 123
EPS = 1e-6

os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

# %% [markdown]
# ## 1. User configuration
#
# `HS_CODES` can be 2, 4, 6, or 10 digit HS codes. In the prior notebook,
# `"8542"` and `"8507"` were used as a small test set. If you use many HS
# codes and months, check the Public Data Portal daily traffic limit.
#
# The API documentation marks `cntyCd` as required, but the prior notebook
# successfully collected all countries by omitting it. The default below keeps
# that behavior. If the API returns a missing-country-code error, set
# `COUNTRY_CODES` to a list such as `["CN", "US", "JP", "VN", "TW", "DE"]`.

# %%
COLLECT_FROM_API = True
LOCAL_TRADE_CSV = None

START_YYYYMM = "202001"
END_YYYYMM = "202412"
HS_CODES = ["8542", "8507"]
COUNTRY_CODES = [None]

ANOMALY_K = 1.5
MIN_COUNTRIES_PER_HS_MONTH = 3

TEST_START = "2024-01-01"
VALID_START = "2023-01-01"

USE_CATEGORICAL_IDS = False
TOP_LABEL_QUANTILE = 0.90

# %% [markdown]
# ## 2. Public Data Portal API collection
#
# Get the service key from:
# 1. https://www.data.go.kr
# 2. Log in
# 3. Search for "Korea Customs item-country trade performance (GW)"
# 4. Click the API usage request button
# 5. Copy the service key into the Colab prompt below

# %%
def make_yearly_periods(start_yyyymm, end_yyyymm):
    """Split requests into periods of at most one year."""
    start_year = int(start_yyyymm[:4])
    end_year = int(end_yyyymm[:4])
    periods = []
    for year in range(start_year, end_year + 1):
        start_month = int(start_yyyymm[4:6]) if year == start_year else 1
        end_month = int(end_yyyymm[4:6]) if year == end_year else 12
        periods.append((f"{year}{start_month:02d}", f"{year}{end_month:02d}"))
    return periods


def call_customs_api_xml(
    service_key,
    start_yyyymm,
    end_yyyymm,
    hs_code=None,
    country_code=None,
    page_no=1,
    num_rows=10000,
    timeout=90,
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

    response = requests.get(base_url, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    return response.text


def parse_customs_xml(xml_text):
    root = ET.fromstring(xml_text)
    result_code = root.findtext(".//resultCode")
    result_msg = root.findtext(".//resultMsg")

    if result_code not in (None, "00"):
        raise RuntimeError(f"API resultCode={result_code}, resultMsg={result_msg}")

    total_count_text = root.findtext(".//totalCount")
    total_count = int(total_count_text) if total_count_text else None

    rows = []
    for item in root.findall(".//item"):
        rows.append({child.tag: child.text for child in item})
    return pd.DataFrame(rows), total_count


def collect_one_job(
    service_key,
    hs_code,
    start_yyyymm,
    end_yyyymm,
    country_code=None,
    sleep_sec=0.35,
    max_retries=3,
):
    all_pages = []
    page_no = 1
    num_rows = 10000

    while True:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                xml_text = call_customs_api_xml(
                    service_key=service_key,
                    start_yyyymm=start_yyyymm,
                    end_yyyymm=end_yyyymm,
                    hs_code=hs_code,
                    country_code=country_code,
                    page_no=page_no,
                    num_rows=num_rows,
                )
                df_page, total_count = parse_customs_xml(xml_text)
                break
            except Exception as err:
                last_error = err
                time.sleep(sleep_sec * attempt)
        else:
            raise RuntimeError(f"Max retries failed: {last_error}")

        if df_page.empty:
            break

        df_page["query_hs_code"] = hs_code
        df_page["query_country_code"] = country_code if country_code else ""
        df_page["query_start_yyyymm"] = start_yyyymm
        df_page["query_end_yyyymm"] = end_yyyymm
        df_page["query_page_no"] = page_no
        all_pages.append(df_page)

        if total_count is None or page_no * num_rows >= total_count:
            break

        page_no += 1
        time.sleep(sleep_sec)

    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


def collect_customs_data(
    service_key,
    hs_codes,
    start_yyyymm,
    end_yyyymm,
    country_codes=(None,),
):
    periods = make_yearly_periods(start_yyyymm, end_yyyymm)
    all_data = []
    failures = []

    for hs_code in hs_codes:
        for country_code in country_codes:
            for start, end in periods:
                label_country = country_code if country_code else "ALL"
                print(f"Collecting hs={hs_code}, country={label_country}, period={start}-{end}")
                try:
                    part = collect_one_job(
                        service_key=service_key,
                        hs_code=hs_code,
                        country_code=country_code,
                        start_yyyymm=start,
                        end_yyyymm=end,
                    )
                    print("  rows:", len(part))
                    if not part.empty:
                        all_data.append(part)
                except Exception as err:
                    print("  failed:", err)
                    failures.append(
                        {
                            "hs_code": hs_code,
                            "country_code": country_code,
                            "start_yyyymm": start,
                            "end_yyyymm": end,
                            "error": str(err),
                        }
                    )

    raw = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    failed = pd.DataFrame(failures)
    return raw, failed


# %% [markdown]
# ## 3. Standardize raw customs data

# %%
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def standardize_customs(df):
    if df is None or df.empty:
        raise ValueError("Input data is empty.")

    required_clean = {"date", "hs_code", "country_code", "import_value", "import_weight"}
    if required_clean.issubset(df.columns):
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["hs_code"] = out["hs_code"].astype(str).str.strip()
        out["country_code"] = out["country_code"].astype(str).str.strip()
        out["import_value"] = pd.to_numeric(out["import_value"], errors="coerce")
        out["import_weight"] = pd.to_numeric(out["import_weight"], errors="coerce")
        if "country_name" not in out.columns:
            out["country_name"] = out["country_code"]
        if "hs_name" not in out.columns:
            out["hs_name"] = out["hs_code"]
    else:
        rename_map = {
            "year": "year_month",
            "hsCd": "hs_code",
            "statCd": "country_code",
            "statCdCntnKor1": "country_name",
            "statKor": "hs_name",
            "impDlr": "import_value",
            "impWgt": "import_weight",
            "expDlr": "export_value",
            "expWgt": "export_weight",
            "balPayments": "trade_balance",
        }
        out = df.rename(columns=rename_map).copy()

        required = [
            "year_month",
            "hs_code",
            "country_code",
            "country_name",
            "hs_name",
            "import_value",
            "import_weight",
        ]
        missing = [col for col in required if col not in out.columns]
        if missing:
            raise KeyError(f"Missing required raw columns: {missing}")

        out["year_month"] = out["year_month"].astype(str).str.strip()
        out["year_month"] = (
            out["year_month"]
            .str.replace(".", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.strip()
        )
        out = out[out["year_month"].str.match(r"^\d{6}$", na=False)].copy()

        out["date"] = pd.to_datetime(out["year_month"] + "01", format="%Y%m%d")
        out["hs_code"] = out["hs_code"].astype(str).str.strip()
        out["country_code"] = out["country_code"].astype(str).str.strip()
        out["country_name"] = out["country_name"].astype(str).str.strip()
        out["hs_name"] = out["hs_name"].astype(str).str.strip()
        out["import_value"] = clean_numeric(out["import_value"])
        out["import_weight"] = clean_numeric(out["import_weight"])

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


# %% [markdown]
# ## 4. Load or collect data

# %%
if COLLECT_FROM_API:
    raw_key = getpass("Public Data Portal service key: ").strip()
    key_candidates = [unquote(raw_key), raw_key]

    customs_raw = pd.DataFrame()
    customs_failed = pd.DataFrame()
    last_error = None
    for key in key_candidates:
        try:
            customs_raw, customs_failed = collect_customs_data(
                service_key=key,
                hs_codes=HS_CODES,
                start_yyyymm=START_YYYYMM,
                end_yyyymm=END_YYYYMM,
                country_codes=COUNTRY_CODES,
            )
            if not customs_raw.empty:
                break
        except Exception as err:
            last_error = err
            print("Key attempt failed:", err)

    if customs_raw.empty:
        raise RuntimeError(
            "API collection failed. Check service key approval and COUNTRY_CODES. "
            f"Last error: {last_error}"
        )

    customs_raw.to_csv("data_raw/customs_raw.csv", index=False, encoding="utf-8-sig")
    customs_failed.to_csv("data_raw/customs_failed_jobs.csv", index=False, encoding="utf-8-sig")
    trade_clean = standardize_customs(customs_raw)
else:
    if LOCAL_TRADE_CSV is None:
        raise ValueError("Set LOCAL_TRADE_CSV or set COLLECT_FROM_API=True.")
    local_df = pd.read_csv(LOCAL_TRADE_CSV)
    trade_clean = standardize_customs(local_df)

trade_clean.to_csv("data_processed/trade_clean.csv", index=False, encoding="utf-8-sig")
print("trade_clean shape:", trade_clean.shape)
display(trade_clean.head())
display(trade_clean[["import_value", "import_weight", "unit_price"]].describe())

# %% [markdown]
# ## 5. CRANE feature engineering

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
    mean_similarity = np.nanmean(sim, axis=1)
    isolation = 1.0 - mean_similarity
    return pd.Series(isolation, index=s.index)


def add_crane_features(df, k=1.5, min_countries=3, eps=EPS):
    out = df.copy().sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)

    group_cols = ["hs_code", "date"]
    grp = out.groupby(group_cols)["log_price"]
    out["ht_country_count"] = grp.transform("count")
    out["ht_median_log_price"] = grp.transform("median")
    out["ht_mad_log_price"] = grp.transform(lambda x: robust_mad(x)).clip(lower=eps)
    out["rel_z"] = (out["log_price"] - out["ht_median_log_price"]) / out["ht_mad_log_price"]

    rank = grp.rank(method="average")
    out["price_rank_pct"] = np.where(
        out["ht_country_count"] > 1,
        (rank - 1) / (out["ht_country_count"] - 1),
        np.nan,
    )
    out.loc[out["ht_country_count"] < min_countries, ["rel_z", "price_rank_pct"]] = np.nan

    out["loo_median_log_price"] = (
        out.groupby(group_cols)["log_price"]
        .apply(leave_one_out_median_series)
        .reset_index(level=group_cols, drop=True)
    )
    out["counterfactual_dev"] = out["log_price"] - out["loo_median_log_price"]
    out["counterfactual_dev_scaled"] = out["counterfactual_dev"] / out["ht_mad_log_price"]

    out["network_isolation_score"] = (
        out.groupby(group_cols)["log_price"]
        .apply(isolation_score_series)
        .reset_index(level=group_cols, drop=True)
    )

    out["total_hs_month_import"] = out.groupby(group_cols)["import_value"].transform("sum")
    out["import_share_country"] = out["import_value"] / out["total_hs_month_import"]
    out["hhi_hs_month"] = out.groupby(group_cols)["import_share_country"].transform(
        lambda x: float(np.sum(np.square(x)))
    )
    out["concentration_risk"] = out["import_share_country"] * out["hhi_hs_month"]

    out = out.sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)
    panel = out.groupby(["hs_code", "country_code"], group_keys=False)
    out["lag_date"] = panel["date"].shift(1)
    out["price_return"] = panel["log_price"].diff()
    out["rank_shift"] = out["price_rank_pct"] - panel["price_rank_pct"].shift(1)

    expected_lag = out["lag_date"] + pd.DateOffset(months=1)
    continuous_lag = out["date"].eq(expected_lag)
    out.loc[~continuous_lag, ["price_return", "rank_shift"]] = np.nan

    out["abs_rank_shift"] = out["rank_shift"].abs()
    out["import_growth"] = panel["import_value"].pct_change()
    out["weight_growth"] = panel["import_weight"].pct_change()
    out["log_import_value"] = np.log(out["import_value"])
    out["log_import_weight"] = np.log(out["import_weight"])

    out["anomaly_abs"] = (out["rel_z"].abs() > k).astype(float)
    out["anomaly_up"] = (out["rel_z"] > k).astype(float)
    out["anomaly_down"] = (out["rel_z"] < -k).astype(float)
    out["abs_rel_z"] = out["rel_z"].abs()

    for col in ["anomaly_abs", "anomaly_up", "anomaly_down", "abs_rel_z"]:
        out[f"{col}_roll3"] = panel[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    out["price_return_roll3"] = panel["price_return"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    out["price_volatility_roll3"] = panel["price_return"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    )
    out["import_growth_roll3"] = panel["import_growth"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )

    out["return_sign"] = np.sign(out["price_return"])

    def majority_sign(x):
        nonzero = x.dropna()
        nonzero = nonzero[nonzero != 0]
        if len(nonzero) == 0:
            return np.nan
        return 1.0 if (nonzero > 0).mean() >= 0.5 else -1.0

    majority = out.groupby(group_cols)["return_sign"].transform(majority_sign)
    out["majority_sign"] = majority
    out["sync_score"] = out.groupby(group_cols)["return_sign"].transform(
        lambda x: np.nan if x.dropna().empty else np.nanmean(x == majority.loc[x.index])
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
        stats["ht_median_return"].abs() / (stats["ht_mad_for_shift"].abs() + eps)
        + stats["ht_mad_return"].abs() / (stats["ht_mad_for_shift"].abs() + eps)
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


crane = add_crane_features(
    trade_clean,
    k=ANOMALY_K,
    min_countries=MIN_COUNTRIES_PER_HS_MONTH,
)

print("crane shape:", crane.shape)
print("rel_z missing ratio:", crane["rel_z"].isna().mean())
display(
    crane[
        [
            "date",
            "hs_code",
            "country_code",
            "unit_price",
            "rel_z",
            "counterfactual_dev_scaled",
            "price_rank_pct",
            "network_isolation_score",
            "import_share_country",
            "hhi_hs_month",
        ]
    ].head(20)
)

# %% [markdown]
# ## 6. Next-month weighted relative anomaly target

# %%
def add_future_targets(df, k=1.5):
    out = df.copy().sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)
    panel = out.groupby(["hs_code", "country_code"], group_keys=False)

    out["next_date"] = panel["date"].shift(-1)
    out["next_rel_z"] = panel["rel_z"].shift(-1)
    out["next_price_rank_pct"] = panel["price_rank_pct"].shift(-1)
    out["next_network_isolation_score"] = panel["network_isolation_score"].shift(-1)

    expected_next = out["date"] + pd.DateOffset(months=1)
    out["has_next_month"] = out["next_date"].eq(expected_next)

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


crane = add_future_targets(crane, k=ANOMALY_K)

target_col = "future_net_anomaly_amount"
print(crane[target_col].describe())
print("target zero ratio:", (crane[target_col] == 0).mean())
print("target missing ratio:", crane[target_col].isna().mean())

plt.figure(figsize=(9, 4))
positive_y = crane.loc[crane[target_col] > 0, target_col]
if len(positive_y) > 0:
    sns.histplot(np.log1p(positive_y), bins=50)
    plt.title("Positive target distribution: log1p(future_net_anomaly_amount)")
    plt.xlabel("log1p target")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Model frame and time split

# %%
numeric_features = [
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
categorical_features = ["hs_code", "country_code"] if USE_CATEGORICAL_IDS else []
feature_cols = numeric_features + categorical_features

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

model_df = crane.copy()
for col in zero_fill_features:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(0)

model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=feature_cols + [target_col, "date"]).copy()

train_df = model_df[model_df["date"] < pd.to_datetime(VALID_START)].copy()
valid_df = model_df[
    (model_df["date"] >= pd.to_datetime(VALID_START))
    & (model_df["date"] < pd.to_datetime(TEST_START))
].copy()
test_df = model_df[model_df["date"] >= pd.to_datetime(TEST_START)].copy()

if valid_df.empty:
    dates = sorted(model_df["date"].dropna().unique())
    valid_cut = dates[int(len(dates) * 0.70)]
    test_cut = dates[int(len(dates) * 0.85)]
    train_df = model_df[model_df["date"] < valid_cut].copy()
    valid_df = model_df[(model_df["date"] >= valid_cut) & (model_df["date"] < test_cut)].copy()
    test_df = model_df[model_df["date"] >= test_cut].copy()
    print("Auto split used:", valid_cut, test_cut)

print("train:", train_df.shape, train_df["date"].min(), train_df["date"].max())
print("valid:", valid_df.shape, valid_df["date"].min(), valid_df["date"].max())
print("test :", test_df.shape, test_df["date"].min(), test_df["date"].max())
print("features:", len(feature_cols))


def make_top_label(y, threshold=None, quantile=0.90):
    y = np.asarray(y, dtype=float)
    if threshold is None:
        threshold = float(np.quantile(y, quantile))
        if threshold <= 0 and np.any(y > 0):
            threshold = float(np.quantile(y[y > 0], 0.50))
    return (y >= threshold).astype(int), threshold


train_top, top_threshold = make_top_label(train_df[target_col], quantile=TOP_LABEL_QUANTILE)
valid_df["actual_top_anomaly"] = make_top_label(valid_df[target_col], top_threshold)[0]
test_df["actual_top_anomaly"] = make_top_label(test_df[target_col], top_threshold)[0]

print("top threshold:", top_threshold)
print("valid top rate:", valid_df["actual_top_anomaly"].mean())
print("test top rate :", test_df["actual_top_anomaly"].mean())

# %% [markdown]
# ## 8. LightGBM Tweedie Gradient Boosting

# %%
def make_lgbm_matrix(df):
    X = df[feature_cols].copy()
    for col in categorical_features:
        X[col] = X[col].astype("category")
    return X


def evaluate_predictions(y_true, pred, top_label, name):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 0)
    top_label = np.asarray(top_label, dtype=int)

    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    auc = roc_auc_score(top_label, pred) if len(np.unique(top_label)) > 1 else np.nan
    ap = average_precision_score(top_label, pred) if len(np.unique(top_label)) > 1 else np.nan

    rank_df = pd.DataFrame({"y": y_true, "pred": pred, "top": top_label}).sort_values(
        "pred", ascending=False
    )
    base = rank_df["top"].mean()

    def precision_at(k):
        if len(rank_df) == 0:
            return np.nan
        return rank_df.head(min(k, len(rank_df)))["top"].mean()

    def recall_at(k):
        denom = rank_df["top"].sum()
        if denom == 0:
            return np.nan
        return rank_df.head(min(k, len(rank_df)))["top"].sum() / denom

    out = {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "auc_top": auc,
        "average_precision": ap,
        "base_rate": base,
        "precision_at_10": precision_at(10),
        "precision_at_30": precision_at(30),
        "precision_at_50": precision_at(50),
        "precision_at_100": precision_at(100),
        "recall_at_10": recall_at(10),
        "recall_at_30": recall_at(30),
        "recall_at_50": recall_at(50),
        "recall_at_100": recall_at(100),
    }
    for k in [10, 30, 50, 100]:
        p = out[f"precision_at_{k}"]
        out[f"lift_at_{k}"] = p / base if base > 0 else np.nan
    return out


X_train = make_lgbm_matrix(train_df)
y_train = train_df[target_col].to_numpy(dtype=float)
X_valid = make_lgbm_matrix(valid_df)
y_valid = valid_df[target_col].to_numpy(dtype=float)
X_test = make_lgbm_matrix(test_df)
y_test = test_df[target_col].to_numpy(dtype=float)

grid = []
for tweedie_power in [1.1, 1.3, 1.5, 1.7, 1.9]:
    for num_leaves in [15, 31, 63]:
        grid.append(
            {
                "tweedie_variance_power": tweedie_power,
                "num_leaves": num_leaves,
                "min_child_samples": 30,
                "learning_rate": 0.03,
                "n_estimators": 2500,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            }
        )

records = []
models = []

for idx, params in enumerate(grid, start=1):
    model = lgb.LGBMRegressor(
        objective="tweedie",
        metric="tweedie",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="tweedie",
        categorical_feature=categorical_features or "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
    rec = evaluate_predictions(
        y_valid,
        valid_pred,
        valid_df["actual_top_anomaly"].to_numpy(),
        name=f"lgbm_tweedie_{idx}",
    )
    rec.update(params)
    rec["best_iteration"] = model.best_iteration_
    records.append(rec)
    models.append(model)
    print(
        idx,
        "power=",
        params["tweedie_variance_power"],
        "leaves=",
        params["num_leaves"],
        "AP=",
        round(rec["average_precision"], 4),
        "P@30=",
        round(rec["precision_at_30"], 4),
        "RMSE=",
        round(rec["rmse"], 2),
    )

cv_results = pd.DataFrame(records).sort_values(
    ["average_precision", "precision_at_30", "rmse"],
    ascending=[False, False, True],
)
display(cv_results.head(10))

best_idx = int(cv_results.index[0])
best_model = models[best_idx]
test_pred = np.maximum(best_model.predict(X_test, num_iteration=best_model.best_iteration_), 0)

test_metrics = pd.DataFrame(
    [
        evaluate_predictions(
            y_test,
            test_pred,
            test_df["actual_top_anomaly"].to_numpy(),
            "LightGBM Tweedie Gradient Boosting",
        )
    ]
)
display(test_metrics)

# %% [markdown]
# ## 9. Interpretation and warning outputs

# %%
test_result = test_df.copy().reset_index(drop=True)
test_result["predicted_net_anomaly_amount"] = test_pred
test_result["risk_rank"] = (
    test_result["predicted_net_anomaly_amount"].rank(method="first", ascending=False).astype(int)
)


def pct_rank(s):
    return s.rank(pct=True).fillna(0.5)


test_result["pred_amount_score"] = pct_rank(test_result["predicted_net_anomaly_amount"])
test_result["idiosyncratic_score"] = pct_rank(
    0.55 * test_result["abs_rel_z"].fillna(0)
    + 0.30 * test_result["network_isolation_score"].fillna(0)
    + 0.15 * test_result["abs_rank_shift"].fillna(0)
)
test_result["systemic_score"] = pct_rank(
    0.50 * test_result["sync_score"].fillna(0)
    + 0.30 * test_result["price_distribution_shift"].fillna(0)
    + 0.20 * test_result["ht_median_return"].abs().fillna(0)
)
test_result["persistence_score"] = pct_rank(test_result["anomaly_abs_roll3"].fillna(0))
test_result["concentration_score"] = pct_rank(test_result["concentration_risk"].fillna(0))

test_result["structural_risk_score"] = (
    0.35 * test_result["idiosyncratic_score"]
    + 0.25 * test_result["pred_amount_score"]
    + 0.15 * test_result["persistence_score"]
    + 0.15 * test_result["concentration_score"]
    + 0.10 * (1 - test_result["systemic_score"])
)

test_result["risk_type"] = np.select(
    [
        (test_result["systemic_score"] >= 0.60) & (test_result["idiosyncratic_score"] >= 0.60),
        (test_result["systemic_score"] >= 0.60) & (test_result["idiosyncratic_score"] < 0.60),
        (test_result["systemic_score"] < 0.60) & (test_result["idiosyncratic_score"] >= 0.60),
    ],
    ["common_plus_country_specific", "common_shock", "country_specific_deviation"],
    default="low_or_mixed",
)

top_warning = test_result.sort_values(
    ["predicted_net_anomaly_amount", "structural_risk_score"], ascending=False
).reset_index(drop=True)
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
    "unit_price",
    "rel_z",
    "price_rank_pct",
    "rank_shift",
    "network_isolation_score",
    "import_share_country",
    "hhi_hs_month",
    "concentration_risk",
    "sync_score",
    "price_distribution_shift",
    "future_net_anomaly_amount",
    "predicted_net_anomaly_amount",
    "structural_risk_score",
    "risk_type",
]

display(top_warning[warning_cols].head(30))

importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_gain": best_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_model.booster_.feature_importance(importance_type="split"),
    }
).sort_values("importance_gain", ascending=False)
display(importance.head(30))

plt.figure(figsize=(9, 7))
plot_df = test_result.copy()
sizes = 30 + 300 * plot_df["pred_amount_score"].clip(0, 1)
scatter = plt.scatter(
    plot_df["systemic_score"],
    plot_df["idiosyncratic_score"],
    s=sizes,
    c=plot_df["structural_risk_score"],
    cmap="viridis",
    alpha=0.65,
    edgecolor="none",
)
plt.axvline(0.60, color="gray", linestyle="--", linewidth=1)
plt.axhline(0.60, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Systemic or common-shock score")
plt.ylabel("Country-specific relative deviation score")
plt.title("CRANE risk map")
plt.colorbar(scatter, label="Structural risk score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(data=importance.head(20), x="importance_gain", y="feature", color="#2A9D8F")
plt.title("LightGBM feature importance by gain")
plt.xlabel("Importance gain")
plt.ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. SHAP summary

# %%
sample_n = min(3000, len(X_test))
if sample_n > 0:
    shap_sample = X_test.sample(sample_n, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, plot_type="bar", max_display=20)

# %% [markdown]
# ## 11. Save artifacts

# %%
crane.to_csv("output/crane_features.csv", index=False, encoding="utf-8-sig")
cv_results.to_csv("output/lgbm_tweedie_cv_results.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv("output/lgbm_tweedie_test_metrics.csv", index=False, encoding="utf-8-sig")
top_warning.to_csv("output/crane_top_warning.csv", index=False, encoding="utf-8-sig")
importance.to_csv("output/lgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

print("Saved:")
print("- output/crane_features.csv")
print("- output/lgbm_tweedie_cv_results.csv")
print("- output/lgbm_tweedie_test_metrics.csv")
print("- output/crane_top_warning.csv")
print("- output/lgbm_feature_importance.csv")
