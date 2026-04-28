# %% [markdown]
# # CRANE Upgrade: Hurdle Model + Structural Ensemble
#
# This notebook upgrades the one-stage Tweedie model in two ways:
#
# 1. Separate "will a material anomaly happen?" from "how large is the amount?"
# 2. Combine predictive amount, material-shock probability, current structural
#    anomaly, and current excess-relative-price amount into one warning score.
#
# Input:
# - `crane_features.csv`, created by `CRANE_colab_full_pipeline.ipynb`
#
# Outputs:
# - `output_upgrade/crane_upgrade_top_warning.csv`
# - `output_upgrade/crane_upgrade_model_comparison.csv`
# - `output_upgrade/crane_upgrade_weight_search.csv`
# - `output_upgrade/crane_upgrade_calibration_decile.csv`

# %%
import importlib.util
import os
import sys


def ensure_package(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


ensure_package("sklearn", "scikit-learn")
ensure_package("lightgbm")

# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
ANOMALY_K = 1.5
TRAIN_END = "2023-01-01"
VALID_END = "2024-01-01"
MATERIAL_QUANTILE = 0.90

os.makedirs("output_upgrade", exist_ok=True)

# %% [markdown]
# ## 1. Load CRANE features

# %%
FEATURE_PATH = "crane_features.csv"

crane = pd.read_csv(FEATURE_PATH, parse_dates=["date", "next_date"])
print("crane shape:", crane.shape)
print("date range:", crane["date"].min(), "~", crane["date"].max())
display(crane.head())

# %% [markdown]
# ## 2. Build model frame

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
    model_df[col] = model_df[col].fillna(0)

model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=numeric_features + [target_col, "date"]).copy()

train_df = model_df[model_df["date"] < pd.to_datetime(TRAIN_END)].copy()
valid_df = model_df[
    (model_df["date"] >= pd.to_datetime(TRAIN_END))
    & (model_df["date"] < pd.to_datetime(VALID_END))
].copy()
test_df = model_df[model_df["date"] >= pd.to_datetime(VALID_END)].copy()

material_threshold = float(train_df[target_col].quantile(MATERIAL_QUANTILE))
if material_threshold <= 0 and (train_df[target_col] > 0).any():
    material_threshold = float(train_df.loc[train_df[target_col] > 0, target_col].median())

for df in [train_df, valid_df, test_df]:
    df["material_anomaly"] = (df[target_col] >= material_threshold).astype(int)

print("material threshold:", material_threshold)
for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    y = df[target_col]
    print(
        name,
        df.shape,
        df["date"].min(),
        df["date"].max(),
        "zero=",
        round((y == 0).mean(), 4),
        "material_rate=",
        round(df["material_anomaly"].mean(), 4),
        "mean_y=",
        round(y.mean(), 2),
    )

# %% [markdown]
# ## 3. Utility functions

# %%
def pct_rank_desc(x):
    s = pd.Series(x)
    return s.rank(pct=True).fillna(0.5).to_numpy()


def evaluate_score(df, score, name, amount_col=target_col, label_col="material_anomaly"):
    score = np.asarray(score, dtype=float)
    score = np.where(np.isfinite(score), score, np.nanmedian(score[np.isfinite(score)]))
    y = df[amount_col].to_numpy(dtype=float)
    label = df[label_col].to_numpy(dtype=int)

    auc = roc_auc_score(label, score) if len(np.unique(label)) > 1 else np.nan
    ap = average_precision_score(label, score) if len(np.unique(label)) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y, np.maximum(score, 0))))
    mae = mean_absolute_error(y, np.maximum(score, 0))
    r2 = r2_score(y, np.maximum(score, 0))

    ranked = df.assign(score=score).sort_values("score", ascending=False)

    def precision_at(k):
        return ranked.head(min(k, len(ranked)))[label_col].mean()

    def recall_at(k):
        denom = ranked[label_col].sum()
        if denom == 0:
            return np.nan
        return ranked.head(min(k, len(ranked)))[label_col].sum() / denom

    out = {
        "model": name,
        "auc": auc,
        "average_precision": ap,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "base_rate": df[label_col].mean(),
    }
    for k in [10, 30, 50, 100, 200]:
        out[f"precision_at_{k}"] = precision_at(k)
        out[f"recall_at_{k}"] = recall_at(k)
        out[f"actual_amount_sum_at_{k}"] = ranked.head(min(k, len(ranked)))[amount_col].sum()
    return out


def add_structural_scores(df, amount_score_col):
    out = df.copy()
    out["pred_amount_score"] = pct_rank_desc(out[amount_score_col])
    out["current_excess_amount"] = out["import_value"] * np.maximum(out["abs_rel_z"] - ANOMALY_K, 0)
    out["current_excess_score"] = pct_rank_desc(out["current_excess_amount"])
    out["idiosyncratic_score"] = pct_rank_desc(
        0.55 * out["abs_rel_z"].fillna(0)
        + 0.30 * out["network_isolation_score"].fillna(0)
        + 0.15 * out["abs_rank_shift"].fillna(0)
    )
    out["systemic_score"] = pct_rank_desc(
        0.50 * out["sync_score"].fillna(0)
        + 0.30 * out["price_distribution_shift"].fillna(0)
        + 0.20 * out["ht_median_return"].abs().fillna(0)
    )
    out["persistence_score"] = pct_rank_desc(out["anomaly_abs_roll3"].fillna(0))
    out["concentration_score"] = pct_rank_desc(out["concentration_risk"].fillna(0))
    out["structural_risk_score_v2"] = (
        0.35 * out["idiosyncratic_score"]
        + 0.20 * out["pred_amount_score"]
        + 0.20 * out["current_excess_score"]
        + 0.10 * out["persistence_score"]
        + 0.10 * out["concentration_score"]
        + 0.05 * (1 - out["systemic_score"])
    )
    out["risk_type_v2"] = np.select(
        [
            (out["systemic_score"] >= 0.60) & (out["idiosyncratic_score"] >= 0.60),
            (out["systemic_score"] >= 0.60) & (out["idiosyncratic_score"] < 0.60),
            (out["systemic_score"] < 0.60) & (out["idiosyncratic_score"] >= 0.60),
        ],
        ["common_plus_country_specific", "common_shock", "country_specific_deviation"],
        default="low_or_mixed",
    )
    return out


def feature_importance_table(model, features):
    return pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

# %% [markdown]
# ## 4. Stage 1: material anomaly classifier

# %%
X_train = train_df[numeric_features]
X_valid = valid_df[numeric_features]
X_test = test_df[numeric_features]

y_train_amount = train_df[target_col].to_numpy(dtype=float)
y_valid_amount = valid_df[target_col].to_numpy(dtype=float)
y_test_amount = test_df[target_col].to_numpy(dtype=float)

y_train_cls = train_df["material_anomaly"].to_numpy(dtype=int)
y_valid_cls = valid_df["material_anomaly"].to_numpy(dtype=int)
y_test_cls = test_df["material_anomaly"].to_numpy(dtype=int)

pos = max(y_train_cls.sum(), 1)
neg = max(len(y_train_cls) - y_train_cls.sum(), 1)
scale_pos_weight = neg / pos
print("scale_pos_weight:", round(scale_pos_weight, 3))

clf_grid = []
for num_leaves in [15, 31, 63]:
    for min_child_samples in [20, 50, 100]:
        clf_grid.append(
            {
                "num_leaves": num_leaves,
                "min_child_samples": min_child_samples,
                "learning_rate": 0.03,
                "n_estimators": 2500,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            }
        )

clf_records = []
clf_models = []
for idx, params in enumerate(clf_grid, start=1):
    clf = lgb.LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    clf.fit(
        X_train,
        y_train_cls,
        eval_set=[(X_valid, y_valid_cls)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    valid_prob = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
    rec = evaluate_score(valid_df, valid_prob, f"classifier_{idx}")
    rec.update(params)
    rec["best_iteration"] = clf.best_iteration_
    clf_records.append(rec)
    clf_models.append(clf)

clf_results = pd.DataFrame(clf_records).sort_values(
    ["average_precision", "precision_at_30", "auc"],
    ascending=[False, False, False],
)
display(clf_results.head(10))

best_clf = clf_models[int(clf_results.index[0])]
valid_prob = best_clf.predict_proba(X_valid, num_iteration=best_clf.best_iteration_)[:, 1]
test_prob = best_clf.predict_proba(X_test, num_iteration=best_clf.best_iteration_)[:, 1]

# %% [markdown]
# ## 5. Stage 2: severity amount model conditional on material anomaly

# %%
severity_train = train_df[train_df["material_anomaly"] == 1].copy()
severity_valid = valid_df[valid_df["material_anomaly"] == 1].copy()

if len(severity_valid) < 30:
    severity_valid = valid_df[valid_df[target_col] > 0].copy()
if len(severity_train) < 50:
    severity_train = train_df[train_df[target_col] > 0].copy()

print("severity train/valid:", severity_train.shape, severity_valid.shape)

sev_grid = []
for power in [1.3, 1.5, 1.7, 1.9]:
    for num_leaves in [15, 31]:
        sev_grid.append(
            {
                "tweedie_variance_power": power,
                "num_leaves": num_leaves,
                "min_child_samples": 10,
                "learning_rate": 0.03,
                "n_estimators": 2500,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
            }
        )

sev_records = []
sev_models = []
for idx, params in enumerate(sev_grid, start=1):
    reg = lgb.LGBMRegressor(
        objective="tweedie",
        metric="tweedie",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    reg.fit(
        severity_train[numeric_features],
        severity_train[target_col],
        eval_set=[(severity_valid[numeric_features], severity_valid[target_col])],
        eval_metric="tweedie",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    pred_valid_sev = np.maximum(reg.predict(X_valid, num_iteration=reg.best_iteration_), 0)
    rec = evaluate_score(valid_df, pred_valid_sev, f"severity_{idx}")
    rec.update(params)
    rec["best_iteration"] = reg.best_iteration_
    sev_records.append(rec)
    sev_models.append(reg)

sev_results = pd.DataFrame(sev_records).sort_values(
    ["r2", "rmse", "average_precision"],
    ascending=[False, True, False],
)
display(sev_results.head(10))

best_sev = sev_models[int(sev_results.index[0])]
valid_severity_amount = np.maximum(
    best_sev.predict(X_valid, num_iteration=best_sev.best_iteration_), 0
)
test_severity_amount = np.maximum(
    best_sev.predict(X_test, num_iteration=best_sev.best_iteration_), 0
)

# %% [markdown]
# ## 6. Hurdle expected amount and hybrid warning score

# %%
valid_hurdle_amount_raw = valid_prob * valid_severity_amount
test_hurdle_amount_raw = test_prob * test_severity_amount

calibration_factor = 1.0
if valid_hurdle_amount_raw.sum() > 0:
    calibration_factor = y_valid_amount.sum() / valid_hurdle_amount_raw.sum()
calibration_factor = float(np.clip(calibration_factor, 0.05, 20.0))

valid_hurdle_amount = valid_hurdle_amount_raw * calibration_factor
test_hurdle_amount = test_hurdle_amount_raw * calibration_factor

valid_scored = valid_df.copy()
valid_scored["material_shock_probability"] = valid_prob
valid_scored["severity_amount_if_material"] = valid_severity_amount
valid_scored["hurdle_expected_amount"] = valid_hurdle_amount
valid_scored = add_structural_scores(valid_scored, "hurdle_expected_amount")

test_scored = test_df.copy()
test_scored["material_shock_probability"] = test_prob
test_scored["severity_amount_if_material"] = test_severity_amount
test_scored["hurdle_expected_amount"] = test_hurdle_amount
test_scored = add_structural_scores(test_scored, "hurdle_expected_amount")

valid_components = {
    "prob": pct_rank_desc(valid_scored["material_shock_probability"]),
    "amount": pct_rank_desc(valid_scored["hurdle_expected_amount"]),
    "structural": valid_scored["structural_risk_score_v2"].to_numpy(),
    "current": valid_scored["current_excess_score"].to_numpy(),
}

weight_records = []
for w_prob in np.arange(0.0, 1.01, 0.1):
    for w_amount in np.arange(0.0, 1.01 - w_prob, 0.1):
        for w_struct in np.arange(0.0, 1.01 - w_prob - w_amount, 0.1):
            w_current = 1.0 - w_prob - w_amount - w_struct
            if w_current < -1e-9:
                continue
            score = (
                w_prob * valid_components["prob"]
                + w_amount * valid_components["amount"]
                + w_struct * valid_components["structural"]
                + w_current * valid_components["current"]
            )
            rec = evaluate_score(valid_scored, score, "hybrid_valid")
            rec.update(
                {
                    "w_prob": round(float(w_prob), 2),
                    "w_amount": round(float(w_amount), 2),
                    "w_structural": round(float(w_struct), 2),
                    "w_current": round(float(w_current), 2),
                }
            )
            weight_records.append(rec)

weight_search = pd.DataFrame(weight_records).sort_values(
    ["average_precision", "precision_at_30", "actual_amount_sum_at_30"],
    ascending=[False, False, False],
)
display(weight_search.head(15))

best_weights = weight_search.iloc[0][["w_prob", "w_amount", "w_structural", "w_current"]]
print("best weights:")
display(best_weights)

test_components = {
    "prob": pct_rank_desc(test_scored["material_shock_probability"]),
    "amount": pct_rank_desc(test_scored["hurdle_expected_amount"]),
    "structural": test_scored["structural_risk_score_v2"].to_numpy(),
    "current": test_scored["current_excess_score"].to_numpy(),
}

test_scored["hybrid_warning_score"] = (
    best_weights["w_prob"] * test_components["prob"]
    + best_weights["w_amount"] * test_components["amount"]
    + best_weights["w_structural"] * test_components["structural"]
    + best_weights["w_current"] * test_components["current"]
)

valid_scored["hybrid_warning_score"] = (
    best_weights["w_prob"] * valid_components["prob"]
    + best_weights["w_amount"] * valid_components["amount"]
    + best_weights["w_structural"] * valid_components["structural"]
    + best_weights["w_current"] * valid_components["current"]
)

# %% [markdown]
# ## 7. Compare upgraded scores

# %%
comparison_records = []

valid_baseline_current = valid_scored["current_excess_amount"].to_numpy()
test_baseline_current = test_scored["current_excess_amount"].to_numpy()

comparison_records.append(
    evaluate_score(test_scored, test_scored["hurdle_expected_amount"], "Hurdle expected amount")
)
comparison_records.append(
    evaluate_score(test_scored, test_scored["material_shock_probability"], "Stage-1 material probability")
)
comparison_records.append(
    evaluate_score(test_scored, test_scored["structural_risk_score_v2"], "Structural risk v2")
)
comparison_records.append(
    evaluate_score(test_scored, test_scored["current_excess_amount"], "Current abs(rel_z) excess x value")
)
comparison_records.append(
    evaluate_score(test_scored, test_scored["hybrid_warning_score"], "Upgraded hybrid warning score")
)

model_comparison = pd.DataFrame(comparison_records).sort_values(
    ["average_precision", "precision_at_30"],
    ascending=[False, False],
)
display(model_comparison)

# %% [markdown]
# ## 8. Top warning table and calibration

# %%
top_warning = test_scored.sort_values(
    ["hybrid_warning_score", "hurdle_expected_amount"],
    ascending=False,
).reset_index(drop=True)
top_warning["upgrade_warning_rank"] = np.arange(1, len(top_warning) + 1)

warning_cols = [
    "upgrade_warning_rank",
    "date",
    "next_date",
    "hs_code",
    "hs_name",
    "country_code",
    "country_name",
    "import_value",
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
    "hybrid_warning_score",
    "structural_risk_score_v2",
    "risk_type_v2",
    "future_net_anomaly_amount",
    "material_anomaly",
]
display(top_warning[warning_cols].head(40))

top_warning["score_decile"] = pd.qcut(
    top_warning["hybrid_warning_score"].rank(method="first"),
    10,
    labels=False,
) + 1
calibration_decile = (
    top_warning.groupby("score_decile", as_index=False)
    .agg(
        n=("score_decile", "size"),
        mean_score=("hybrid_warning_score", "mean"),
        mean_probability=("material_shock_probability", "mean"),
        mean_expected_amount=("hurdle_expected_amount", "mean"),
        mean_actual_amount=(target_col, "mean"),
        material_rate=("material_anomaly", "mean"),
        actual_amount_sum=(target_col, "sum"),
    )
    .sort_values("score_decile")
)
display(calibration_decile)

plt.figure(figsize=(9, 5))
sns.barplot(data=calibration_decile, x="score_decile", y="material_rate", color="#2A9D8F")
plt.axhline(test_scored["material_anomaly"].mean(), color="gray", linestyle="--", linewidth=1)
plt.title("Material anomaly rate by upgraded warning-score decile")
plt.xlabel("Warning-score decile")
plt.ylabel("Observed material anomaly rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Save outputs

# %%
classifier_importance = feature_importance_table(best_clf, numeric_features)
severity_importance = feature_importance_table(best_sev, numeric_features)

clf_results.to_csv("output_upgrade/crane_upgrade_classifier_cv.csv", index=False, encoding="utf-8-sig")
sev_results.to_csv("output_upgrade/crane_upgrade_severity_cv.csv", index=False, encoding="utf-8-sig")
weight_search.to_csv("output_upgrade/crane_upgrade_weight_search.csv", index=False, encoding="utf-8-sig")
model_comparison.to_csv("output_upgrade/crane_upgrade_model_comparison.csv", index=False, encoding="utf-8-sig")
top_warning.to_csv("output_upgrade/crane_upgrade_top_warning.csv", index=False, encoding="utf-8-sig")
calibration_decile.to_csv("output_upgrade/crane_upgrade_calibration_decile.csv", index=False, encoding="utf-8-sig")
classifier_importance.to_csv(
    "output_upgrade/crane_upgrade_classifier_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)
severity_importance.to_csv(
    "output_upgrade/crane_upgrade_severity_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

print("Saved upgraded outputs to output_upgrade/")
