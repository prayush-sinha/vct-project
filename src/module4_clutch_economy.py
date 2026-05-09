"""
Module 4 — Clutch & Round Economy Analysis
============================================
Two complementary analyses:

A) Clutch Success Probability
   Predicts whether a player wins a clutch situation (1vN) using per-game
   features. Target: has at least 1 clutch win in the map.

B) Economy → Round Win Impact
   Quantifies how spending strategy (full-buy vs eco vs force) influences
   round win rate, controlling for team quality.
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

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import shap

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIG_DIR    = Path(__file__).parent.parent / "outputs" / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# A — CLUTCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

CLUTCH_FEATURES = [
    "kill_per_round", "death_per_round", "KD_ratio",
    "entry_success_rate", "hs_pct", "kast_pct",
    "multikill_score", "clutch_weight",
    "econ_rating", "win_rate_rounds",
    "impact_score",
    "avg_bank", "full_buy_pct",
    "map_enc",
]


def build_clutch_dataset(df: pd.DataFrame):
    """
    Target: binary — player had at least one successful clutch in the map.
    A 'clutch' here = any 1v1/1v2/…/1v5 won.
    """
    d = df.copy()
    d["clutch_any"] = (d["clutch_total"] > 0).astype(int)

    feats = [c for c in CLUTCH_FEATURES if c in d.columns]
    X = d[feats].fillna(d[feats].median())
    y = d["clutch_any"]
    print(f"[Module4-A] Clutch dataset: {len(X):,} rows | "
          f"Clutch rate: {y.mean():.2%}")
    return X, y, feats


def train_clutch_predictor(X, y, feats) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))])
    lr_auc = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    lr.fit(X, y)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    xgb_auc = cross_val_score(xgb_clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    xgb_clf.fit(X, y)

    print(f"[Module4-A] LR  ROC-AUC CV: {lr_auc.mean():.4f} ± {lr_auc.std():.4f}")
    print(f"[Module4-A] XGB ROC-AUC CV: {xgb_auc.mean():.4f} ± {xgb_auc.std():.4f}")

    return {"LR": lr, "XGBoost": xgb_clf}


def plot_clutch_shap(model, X, feats, fig_dir):
    sample = X.sample(min(800, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    sv_1 = sv[1] if isinstance(sv, list) else sv

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv_1, sample, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP — Clutch Success Drivers (XGBoost)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module4_clutch_shap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module4-A] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# B — ECONOMY IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def economy_round_analysis(df: pd.DataFrame, fig_dir: Path):
    """
    Visualise:
    1. Win rate by economy tier (eco / force / full-buy)
    2. Average bank size for winners vs losers
    3. Full-buy % vs win-rate scatter (team level)
    """
    d = df.copy()
    # Bin eco_rounds_pct into tiers
    d["eco_tier"] = pd.cut(
        d["eco_rounds_pct"],
        bins=[-0.01, 0.15, 0.35, 0.60, 1.01],
        labels=["Mostly Full-Buy", "Mixed Economy", "High Eco", "Pure Eco"]
    )
    d["full_buy_tier"] = pd.cut(
        d["full_buy_pct"],
        bins=[-0.01, 0.30, 0.55, 0.80, 1.01],
        labels=["Low Full-Buy", "Moderate", "High Full-Buy", "Always Full"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Win rate by eco_tier
    tier_win = d.groupby("eco_tier", observed=True)["win"].mean().reset_index()
    tier_win.columns = ["tier", "win_rate"]
    axes[0].bar(tier_win["tier"], tier_win["win_rate"],
                color=["#4CAF50", "#FFC107", "#FF7043", "#F44336"],
                alpha=0.85, edgecolor="white")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="navy", linestyle="--", alpha=0.5)
    axes[0].set_title("Win Rate by Economy Tier", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_xlabel("Economy Tier")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(tier_win["win_rate"]):
        axes[0].text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=9)

    # Panel 2: avg bank for W vs L
    bank_summary = d.groupby("win")["avg_bank"].median().reset_index()
    axes[1].bar(["Loss", "Win"], bank_summary["avg_bank"],
                color=["#E84855", "#3E92CC"], alpha=0.85, edgecolor="white")
    axes[1].set_title("Median Bank Size — Win vs Loss", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Average Bank (credits)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(bank_summary["avg_bank"]):
        axes[1].text(i, v + 50, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: full_buy_pct vs win_rate scatter (per player)
    scatter_d = d.groupby("player").agg(
        full_buy_pct=("full_buy_pct", "mean"),
        win_rate=("win", "mean"),
        n=("win", "count"),
    ).reset_index()
    scatter_d = scatter_d[scatter_d["n"] >= 5]
    axes[2].scatter(scatter_d["full_buy_pct"], scatter_d["win_rate"],
                    c=scatter_d["win_rate"], cmap="RdYlGn",
                    s=scatter_d["n"].clip(5, 40) * 1.5,
                    alpha=0.65, edgecolors="white", linewidths=0.3)
    # Trend line
    valid = scatter_d[["full_buy_pct", "win_rate"]].dropna()
    valid = valid[np.isfinite(valid["full_buy_pct"]) & np.isfinite(valid["win_rate"])]
    if len(valid) > 2:
        try:
            z = np.polyfit(valid["full_buy_pct"], valid["win_rate"], 1)
            p = np.poly1d(z)
            xs = np.linspace(valid["full_buy_pct"].min(), valid["full_buy_pct"].max(), 100)
            axes[2].plot(xs, p(xs), "r--", lw=1.5, alpha=0.7, label="Trend")
        except Exception:
            pass
    axes[2].set_title("Full-Buy % vs Player Win Rate", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Full-Buy Round Fraction")
    axes[2].set_ylabel("Win Rate")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Economy & Round Impact Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = fig_dir / "module4_economy_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module4-B] Saved → {path}")


def clutch_by_scenario(df: pd.DataFrame, fig_dir: Path):
    """Visualise clutch success rate broken down by scenario (1v1..1v5)."""
    clutch_map = {
        "clutch_1v1": "1v1", "clutch_1v2": "1v2",
        "clutch_1v3": "1v3", "clutch_1v4": "1v4", "clutch_1v5": "1v5",
    }
    totals = {label: df[col].sum() for col, label in clutch_map.items()
              if col in df.columns}

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3E92CC", "#4CAF50", "#FFC107", "#FF7043", "#E84855"]
    bars = ax.bar(totals.keys(), totals.values(), color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("Total Clutch Wins by Scenario (All Matches)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Clutch Wins")
    ax.set_xlabel("Clutch Scenario")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, totals.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module4_clutch_scenarios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module4-B] Saved → {path}")


def run():
    df = pd.read_parquet(DATA_DIR / "player_features.parquet")

    # ── A: Clutch predictor ────────────────────────────────────────────────────
    print("\n═══ Module 4-A: Clutch Prediction ═══")
    X_c, y_c, feats_c = build_clutch_dataset(df)
    clutch_models = train_clutch_predictor(X_c, y_c, feats_c)
    plot_clutch_shap(clutch_models["XGBoost"], X_c, feats_c, FIG_DIR)
    joblib.dump(clutch_models["XGBoost"], MODELS_DIR / "xgb_clutch.pkl")

    # ── B: Economy analysis ────────────────────────────────────────────────────
    print("\n═══ Module 4-B: Economy Analysis ═══")
    economy_round_analysis(df, FIG_DIR)
    clutch_by_scenario(df, FIG_DIR)

    print("\n✅ Module 4 complete.")
    return clutch_models


if __name__ == "__main__":
    run()
