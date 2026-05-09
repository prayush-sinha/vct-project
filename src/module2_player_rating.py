"""
Module 2 — Player Performance Rating System
============================================
Trains a regression model to predict a composite player impact score,
then ranks players by predicted performance and compares against
traditional metrics (ACS, VLR rating).

Key ideas
---------
• Target  : impact_score (engineered in data_processing.py) — captures
  frags, clutches, entry success and economy contribution
• Model   : Ridge regression (interpretable) + XGBoost regression
• Output  : Global player leaderboard with confidence bands
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIG_DIR    = Path(__file__).parent.parent / "outputs" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_FEATURES = [
    "kills", "deaths", "assists",
    "acs", "adr", "kast_pct", "hs_pct",
    "KD_ratio", "KDA_ratio",
    "kill_per_round", "death_per_round",
    "first_kills", "first_deaths",
    "entry_success_rate", "entry_diff",
    "clutch_total", "clutch_weight", "clutch_per_round",
    "multikill_score",
    "util_score",
    "win_rate_rounds",
    "econ_rating",
    "avg_bank", "eco_rounds_pct", "full_buy_pct",
    "map_enc",
]

TARGET = "impact_score"


def load_data() -> tuple[pd.DataFrame, pd.Series, list]:
    df = pd.read_parquet(DATA_DIR / "player_features.parquet")
    feats = [c for c in PLAYER_FEATURES if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    y = df[TARGET]
    print(f"[Module2] Player rating dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    return df, X, y, feats


def train_models(X, y) -> dict:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Ridge Regression ──────────────────────────────────────────────────────
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=10.0)),
    ])
    ridge_cv = cross_val_score(ridge, X, y, cv=cv, scoring="r2", n_jobs=-1)
    ridge.fit(X, y)
    print(f"[Module2] Ridge R² CV: {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")

    # ── XGBoost Regressor ─────────────────────────────────────────────────────
    xgb_reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        eval_metric="rmse",
    )
    xgb_cv = cross_val_score(xgb_reg, X, y, cv=cv, scoring="r2", n_jobs=-1)
    xgb_reg.fit(X, y)
    print(f"[Module2] XGBoost R² CV: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    return {"Ridge": ridge, "XGBoost": xgb_reg}


def build_leaderboard(df: pd.DataFrame, X: pd.DataFrame,
                       models: dict, features: list) -> pd.DataFrame:
    """
    Predict per-game impact score → aggregate to player-level → rank.
    Returns a leaderboard DataFrame.
    """
    # Use XGBoost (higher R²) for final predictions
    model = models["XGBoost"]
    df = df.copy()
    df["pred_impact"] = model.predict(X)

    # Aggregate per player (at least 5 map appearances to be eligible)
    agg = df.groupby("player").agg(
        team          = ("team", lambda x: x.mode().iloc[0]),
        maps_played   = ("match_id", "nunique"),
        avg_acs       = ("acs", "mean"),
        avg_vlr       = ("vlr_rating", "mean"),
        avg_kd        = ("KD_ratio", "mean"),
        avg_impact    = ("impact_score", "mean"),
        pred_impact   = ("pred_impact", "mean"),
        win_rate      = ("win", "mean"),
        avg_first_kills = ("first_kills", "mean"),
        avg_clutch    = ("clutch_per_round", "mean"),
        avg_hs        = ("hs_pct", "mean"),
    ).reset_index()

    agg = agg[agg["maps_played"] >= 5].copy()
    agg["rank"] = agg["pred_impact"].rank(ascending=False).astype(int)
    agg = agg.sort_values("rank")

    print(f"[Module2] Leaderboard: {len(agg)} eligible players")
    return agg


def plot_leaderboard(lb: pd.DataFrame, fig_dir: Path):
    """Top-30 player performance heatmap + scatter vs ACS."""
    top30 = lb.head(30).copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # ── Left: horizontal bar — predicted impact ───────────────────────────────
    ax = axes[0]
    bars = ax.barh(top30["player"][::-1], top30["pred_impact"][::-1],
                   color=plt.cm.RdYlGn(np.linspace(0.35, 0.9, len(top30))), edgecolor="white")
    ax.set_xlabel("Predicted Impact Score", fontsize=11)
    ax.set_title("🏆 VCT Player Performance Leaderboard (Top 30)",
                 fontsize=12, fontweight="bold")
    ax.axvline(top30["pred_impact"].mean(), color="navy", linestyle="--",
               alpha=0.6, label="Mean")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    for bar, team in zip(bars[::-1], top30["team"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                team, va="center", ha="left", fontsize=6.5, color="#444")

    # ── Right: predicted impact vs ACS scatter ────────────────────────────────
    ax2 = axes[1]
    sc = ax2.scatter(lb["avg_acs"], lb["pred_impact"],
                     c=lb["win_rate"], cmap="RdYlGn",
                     s=60, alpha=0.7, edgecolors="white", linewidths=0.3)
    # Highlight top 10
    for _, row in lb.head(10).iterrows():
        ax2.annotate(row["player"], (row["avg_acs"], row["pred_impact"]),
                     fontsize=7, ha="left", color="#222",
                     xytext=(3, 3), textcoords="offset points")
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Win Rate", fontsize=9)
    r, _ = pearsonr(lb["avg_acs"].dropna(), lb["pred_impact"][lb["avg_acs"].notna()])
    ax2.set_xlabel("Average ACS (Traditional Metric)", fontsize=11)
    ax2.set_ylabel("Predicted Impact Score (ML Model)", fontsize=11)
    ax2.set_title(f"Predicted Impact vs Traditional ACS  (r = {r:.3f})",
                  fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = fig_dir / "module2_player_leaderboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module2] Saved → {path}")


def plot_metric_correlation(lb: pd.DataFrame, fig_dir: Path):
    """Spearman correlation matrix between player rating metrics."""
    metrics = ["avg_acs", "avg_vlr", "avg_kd", "avg_impact", "pred_impact",
               "win_rate", "avg_first_kills", "avg_clutch"]
    corr = lb[metrics].dropna().corr(method="spearman")
    labels = ["ACS", "VLR Rating", "K/D", "Impact\n(true)", "Impact\n(pred)",
              "Win Rate", "First Kills", "Clutch Rate"]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, square=True)
    ax.set_title("Spearman Correlation — Player Performance Metrics",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module2_metric_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module2] Saved → {path}")


def plot_shap_player(models, X, features, fig_dir):
    """SHAP for XGBoost regressor — what drives a high impact score."""
    model = models["XGBoost"]
    sample = X.sample(min(1000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional",
                                    model_output="raw")
    sv = explainer.shap_values(sample, check_additivity=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, sample, show=False, max_display=18)
    plt.title("SHAP — Drivers of Player Impact Score (XGBoost Regressor)",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module2_shap_player.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module2] Saved → {path}")


def run():
    df, X, y, features = load_data()
    models = train_models(X, y)
    lb = build_leaderboard(df, X, models, features)

    # Save artefacts
    joblib.dump(models["XGBoost"], MODELS_DIR / "xgb_player_rater.pkl")
    lb.to_csv(DATA_DIR / "player_leaderboard.csv", index=False)

    plot_leaderboard(lb, FIG_DIR)
    plot_metric_correlation(lb, FIG_DIR)
    plot_shap_player(models, X, features, FIG_DIR)

    print("\n── Top 15 Players ──────────────────────────────────────────────")
    print(lb[["rank","player","team","maps_played","avg_acs","avg_kd","pred_impact","win_rate"]]
          .head(15).to_string(index=False))

    return models, lb


if __name__ == "__main__":
    run()
