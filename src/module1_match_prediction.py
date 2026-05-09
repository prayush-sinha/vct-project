"""
Module 1 — Match Outcome Prediction
=====================================
Predicts whether a team wins a given map using aggregated team-level features.

Models
------
  • Logistic Regression (baseline)
  • XGBoost classifier
  • LightGBM classifier (with Optuna hyperparameter tuning)

Evaluation: Accuracy, F1-macro, ROC-AUC, calibration curve
Explainability: SHAP summary + beeswarm plots
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.calibration import CalibrationDisplay
import xgboost as xgb
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
MODELS_DIR  = Path(__file__).parent.parent / "models"
FIG_DIR     = Path(__file__).parent.parent / "outputs" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "kills_mean", "deaths_mean", "assists_mean",
    "acs_mean", "adr_mean",
    "KD_ratio_mean", "KDA_ratio_mean",
    "kill_per_round_mean", "death_per_round_mean",
    "entry_success_rate_mean", "entry_diff_mean",
    "clutch_per_round_mean", "clutch_weight_mean",
    "multikill_score_mean",
    "impact_score_mean",
    "util_score_mean",
    "econ_rating_mean",
    "avg_bank_mean", "eco_rounds_pct_mean", "full_buy_pct_mean",
    # opponent mirror features
    "kills_mean_opp", "deaths_mean_opp", "acs_mean_opp",
    "KD_ratio_mean_opp", "impact_score_mean_opp",
    "entry_success_rate_mean_opp", "clutch_per_round_mean_opp",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    team_df = pd.read_parquet(DATA_DIR / "team_features.parquet")

    # Keep only columns that exist
    cols = [c for c in FEATURE_COLS if c in team_df.columns]
    X = team_df[cols].fillna(team_df[cols].median())
    y = team_df["match_outcome"].astype(int)
    print(f"[Module1] Dataset: {X.shape[0]:,} samples, {X.shape[1]} features | "
          f"Win rate: {y.mean():.2%}")
    return X, y, cols


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_idx = list(cv.split(X, y))
    tr_idx, te_idx = split_idx[-1]   # hold out last fold for detailed analysis
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    results = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                                   random_state=42)),
    ])
    lr_pipe.fit(X_tr, y_tr)
    lr_proba = lr_pipe.predict_proba(X_te)[:, 1]
    lr_pred  = lr_pipe.predict(X_te)
    results["Logistic Regression"] = {
        "model": lr_pipe,
        "proba": lr_proba,
        "pred":  lr_pred,
        "acc":   accuracy_score(y_te, lr_pred),
        "f1":    f1_score(y_te, lr_pred, average="macro"),
        "auc":   roc_auc_score(y_te, lr_proba),
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)
    xgb_proba = xgb_model.predict_proba(X_te)[:, 1]
    xgb_pred  = xgb_model.predict(X_te)
    results["XGBoost"] = {
        "model": xgb_model,
        "proba": xgb_proba,
        "pred":  xgb_pred,
        "acc":   accuracy_score(y_te, xgb_pred),
        "f1":    f1_score(y_te, xgb_pred, average="macro"),
        "auc":   roc_auc_score(y_te, xgb_proba),
    }

    # ── LightGBM (manually tuned) ─────────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        num_leaves=31, subsample=0.85, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=42, n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    lgb_proba = lgb_model.predict_proba(X_te)[:, 1]
    lgb_pred  = lgb_model.predict(X_te)
    results["LightGBM"] = {
        "model": lgb_model,
        "proba": lgb_proba,
        "pred":  lgb_pred,
        "acc":   accuracy_score(y_te, lgb_pred),
        "f1":    f1_score(y_te, lgb_pred, average="macro"),
        "auc":   roc_auc_score(y_te, lgb_proba),
    }

    # ── Cross-validated AUC ───────────────────────────────────────────────────
    print("\n── 5-Fold CV ROC-AUC ─────────────────────────────────────────────")
    for name, res in results.items():
        cv_scores = cross_val_score(res["model"], X, y,
                                    cv=cv, scoring="roc_auc", n_jobs=-1)
        res["cv_auc"] = cv_scores
        print(f"  {name:22s}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Print classification reports ──────────────────────────────────────────
    print("\n── Test-fold Classification Reports ─────────────────────────────")
    for name, res in results.items():
        print(f"\n{name}")
        print(classification_report(y_te, res["pred"], target_names=["Loss", "Win"]))

    return results, X_te, y_te, feature_names


def plot_comparison(results: dict, y_te, fig_dir: Path):
    """ROC curves + calibration + CV-AUC comparison on a single figure."""
    colours = {"Logistic Regression": "#4C72B0",
               "XGBoost":            "#DD8452",
               "LightGBM":           "#55A868"}

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # ROC curves
    ax_roc = fig.add_subplot(gs[0])
    for name, res in results.items():
        RocCurveDisplay.from_predictions(
            y_te, res["proba"],
            name=f"{name} (AUC={res['auc']:.3f})",
            ax=ax_roc, color=colours[name],
        )
    ax_roc.set_title("ROC Curves — Match Outcome Prediction", fontsize=11, fontweight="bold")
    ax_roc.legend(fontsize=8)

    # Calibration curves
    ax_cal = fig.add_subplot(gs[1])
    for name, res in results.items():
        CalibrationDisplay.from_predictions(
            y_te, res["proba"], n_bins=10,
            name=name, ax=ax_cal, color=colours[name],
        )
    ax_cal.set_title("Calibration Curves", fontsize=11, fontweight="bold")
    ax_cal.legend(fontsize=8)

    # CV-AUC boxplot
    ax_box = fig.add_subplot(gs[2])
    box_data = [res["cv_auc"] for res in results.values()]
    bp = ax_box.boxplot(box_data, patch_artist=True, labels=list(results.keys()),
                        medianprops=dict(color="black", linewidth=2))
    for patch, (name, _) in zip(bp["boxes"], results.items()):
        patch.set_facecolor(colours[name])
        patch.set_alpha(0.7)
    ax_box.set_title("5-Fold CV ROC-AUC Distribution", fontsize=11, fontweight="bold")
    ax_box.set_ylabel("ROC-AUC")
    ax_box.tick_params(axis="x", labelsize=8)
    ax_box.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "module1_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Module1] Saved → {path}")


def plot_feature_importance(results: dict, feature_names: list, fig_dir: Path):
    """XGBoost feature importance + LightGBM feature importance side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, name in zip(axes, ["XGBoost", "LightGBM"]):
        model = results[name]["model"]
        imp = model.feature_importances_
        fi = pd.Series(imp, index=feature_names).sort_values(ascending=False).head(15)
        fi.sort_values().plot.barh(ax=ax, color="#4C72B0", alpha=0.8)
        ax.set_title(f"{name} — Top-15 Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "module1_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module1] Saved → {path}")


def plot_shap(results: dict, X_te: pd.DataFrame, fig_dir: Path):
    """SHAP beeswarm plot for the best model (LightGBM)."""
    model = results["LightGBM"]["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # For binary classification lgb returns a list; take class-1 values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_te, plot_type="dot",
                      show=False, max_display=20)
    plt.title("SHAP Feature Impact — LightGBM Match Predictor", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module1_shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module1] Saved → {path}")


def run():
    X, y, feature_names = load_data()
    results, X_te, y_te, feature_names = train_and_evaluate(X, y, feature_names)

    # Save best model (LightGBM)
    joblib.dump(results["LightGBM"]["model"], MODELS_DIR / "lgbm_match_predictor.pkl")
    joblib.dump(results["XGBoost"]["model"],  MODELS_DIR / "xgb_match_predictor.pkl")
    joblib.dump(results["Logistic Regression"]["model"], MODELS_DIR / "lr_match_predictor.pkl")
    print(f"\n[Module1] Models saved to {MODELS_DIR}")

    plot_comparison(results, y_te, FIG_DIR)
    plot_feature_importance(results, feature_names, FIG_DIR)
    plot_shap(results, X_te, FIG_DIR)

    # Summary table
    print("\n── Final Test-set Metrics ────────────────────────────────────────")
    print(f"{'Model':22s}  {'Accuracy':>9}  {'F1-macro':>9}  {'ROC-AUC':>9}")
    for name, res in results.items():
        print(f"{name:22s}  {res['acc']:9.4f}  {res['f1']:9.4f}  {res['auc']:9.4f}")

    return results


if __name__ == "__main__":
    run()
