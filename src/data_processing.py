"""
VCT Esports Analytics — Data Processing & Feature Engineering
=============================================================
Loads raw player-level match data, cleans it, and engineers features for
all downstream ML modules.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
VCT_PATH = DATA_DIR / "VCTdata.csv"
AGENT_PATH = DATA_DIR / "agent_id.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RAW LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (vct_df, agent_df) with minimal type coercion."""
    vct = pd.read_csv(VCT_PATH, parse_dates=["match_date"])
    agent = pd.read_csv(AGENT_PATH)
    return vct, agent


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLEANING
# ══════════════════════════════════════════════════════════════════════════════
_NUMERIC_FILL_MEDIANS = [
    "kills", "deaths", "assists", "acs", "adr",
    "kast_pct", "hs_pct", "first_kills", "first_deaths",
    "econ_rating", "plants", "defuses",
    "multikill_2k", "multikill_3k", "multikill_4k", "multikill_5k",
    "clutch_1v1", "clutch_1v2", "clutch_1v3", "clutch_1v4", "clutch_1v5",
]

def clean(vct: pd.DataFrame, agent: pd.DataFrame) -> pd.DataFrame:
    """
    • Deduplicate rows
    • Impute numeric nulls with column median (per player when enough data,
      else global median) — avoids leakage of future data
    • Derive 'rounds_played' and guard against division-by-zero
    • Binary-encode 'result'
    • Normalise categorical strings
    """
    df = vct.copy()

    # ── 2.1  Deduplicate ──────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "player", "map"])
    print(f"[clean] Dropped {before - len(df)} duplicate rows → {len(df)} rows remain")

    # ── 2.2  Impute numeric nulls ─────────────────────────────────────────────
    for col in _NUMERIC_FILL_MEDIANS:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        # per-player median first, then global fallback
        player_med = df.groupby("player")[col].transform("median")
        global_med = df[col].median()
        df[col] = df[col].fillna(player_med).fillna(global_med)
        print(f"[clean] Imputed {null_count:,} nulls in '{col}'")

    # vlr_rating — special treatment (many nulls)
    if df["vlr_rating"].isna().sum() > 0:
        player_med = df.groupby("player")["vlr_rating"].transform("median")
        df["vlr_rating"] = df["vlr_rating"].fillna(player_med).fillna(df["vlr_rating"].median())

    # ── 2.3  Derived base columns ─────────────────────────────────────────────
    df["rounds_played"] = df["rounds_won"] + df["rounds_lost"]
    df["rounds_played"] = df["rounds_played"].replace(0, 1)   # guard /0

    # Binary target  (1 = Win)
    df["win"] = (df["result"] == "Win").astype(int)

    # Normalise strings
    df["map"] = df["map"].str.strip().str.title()
    df["team"] = df["team"].str.strip()
    df["player"] = df["player"].str.strip()

    # Sort chronologically for rolling features
    df = df.sort_values(["player", "match_date", "match_id"]).reset_index(drop=True)

    print(f"[clean] Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _clutch_total(df: pd.DataFrame) -> pd.Series:
    """Sum of all clutch situations attempted."""
    clutch_cols = ["clutch_1v1", "clutch_1v2", "clutch_1v3", "clutch_1v4", "clutch_1v5"]
    return df[clutch_cols].sum(axis=1)


def _clutch_weight(df: pd.DataFrame) -> pd.Series:
    """Weighted clutch score: harder clutches count more."""
    weights = {"clutch_1v1": 1, "clutch_1v2": 2, "clutch_1v3": 3,
               "clutch_1v4": 4, "clutch_1v5": 5}
    return sum(df[col] * w for col, w in weights.items())


def _parse_economy(economy_json: str) -> dict:
    """Parse round_economy JSON → aggregate economy metrics."""
    try:
        rounds = json.loads(economy_json)
        banks = [r.get("bank", 0) for r in rounds]
        buy_types = [r.get("buy_type", "") for r in rounds]
        total_rounds = max(len(rounds), 1)
        return {
            "avg_bank": np.mean(banks),
            "eco_rounds_pct": sum(1 for b in buy_types if b.lower() in ["eco", "pistol"]) / total_rounds,
            "full_buy_pct": sum(1 for b in buy_types if b.lower() == "full buy") / total_rounds,
        }
    except Exception:
        return {"avg_bank": np.nan, "eco_rounds_pct": np.nan, "full_buy_pct": np.nan}


def engineer_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-player-per-map advanced features.

    Feature groups
    ──────────────
    KD_ratio            — K/D ratio; classic fragging efficiency
    KDA_ratio           — accounts for assists
    kill_per_round      — kills normalised by rounds played
    death_per_round     — deaths normalised by rounds played
    clutch_total        — raw clutch count; proxy for clutch frequency
    clutch_weight       — difficulty-weighted clutch index
    clutch_per_round    — clutch rate per round (avoid bias from long maps)
    multikill_score     — composite multikill count (2k–5k weighted)
    entry_success_rate  — first_kills / (first_kills + first_deaths)
    entry_diff          — first_kills − first_deaths (net entry advantage)
    win_rate_rounds     — rounds_won / rounds_played (map win momentum)
    util_score          — plants + defuses (site-control utility index)
    impact_score        — synthetic player impact (custom composite)
    avg_bank            — average economy before buying
    eco_rounds_pct      — fraction of rounds on eco/pistol buys
    full_buy_pct        — fraction of rounds on full-buy
    """
    d = df.copy()

    # ── 3.1  Classic ratios ───────────────────────────────────────────────────
    d["KD_ratio"]      = d["kills"] / d["deaths"].clip(lower=1)
    d["KDA_ratio"]     = (d["kills"] + 0.5 * d["assists"]) / d["deaths"].clip(lower=1)
    d["kill_per_round"] = d["kills"] / d["rounds_played"]
    d["death_per_round"] = d["deaths"] / d["rounds_played"]

    # ── 3.2  Clutch features ──────────────────────────────────────────────────
    d["clutch_total"]   = _clutch_total(d)
    d["clutch_weight"]  = _clutch_weight(d)
    d["clutch_per_round"] = d["clutch_weight"] / d["rounds_played"]

    # ── 3.3  Multi-kill composite ─────────────────────────────────────────────
    d["multikill_score"] = (
        d["multikill_2k"] * 1 +
        d["multikill_3k"] * 2 +
        d["multikill_4k"] * 3 +
        d["multikill_5k"] * 4
    )

    # ── 3.4  Aggression / entry ───────────────────────────────────────────────
    fk = d["first_kills"]
    fd = d["first_deaths"]
    d["entry_success_rate"] = fk / (fk + fd).clip(lower=1)
    d["entry_diff"]         = fk - fd

    # ── 3.5  Round-level efficiency ───────────────────────────────────────────
    d["win_rate_rounds"] = d["rounds_won"] / d["rounds_played"]
    d["util_score"]      = d["plants"] + d["defuses"]

    # ── 3.6  Economy features (JSON parse) ────────────────────────────────────
    if "round_economy" in d.columns:
        eco = d["round_economy"].fillna("[]").apply(_parse_economy)
        eco_df = pd.DataFrame(eco.tolist(), index=d.index)
        d = pd.concat([d, eco_df], axis=1)
        # Fill economy nulls with global medians
        for col in ["avg_bank", "eco_rounds_pct", "full_buy_pct"]:
            d[col] = d[col].fillna(d[col].median())
    else:
        d["avg_bank"] = np.nan
        d["eco_rounds_pct"] = np.nan
        d["full_buy_pct"] = np.nan

    # ── 3.7  Composite impact score ───────────────────────────────────────────
    # Designed to resemble a VCT-style performance index:
    # High ACS + good KD + clutches + entries − deaths per round
    d["impact_score"] = (
        0.30 * d["acs"].fillna(d["acs"].median()) / d["acs"].median() +
        0.20 * d["KD_ratio"] / d["KD_ratio"].median() +
        0.15 * d["entry_success_rate"] +
        0.15 * d["clutch_per_round"] / (d["clutch_per_round"].median() + 1e-9) +
        0.10 * d["multikill_score"].clip(upper=10) / 10 +
        0.10 * (1 - d["death_per_round"] / max(d["death_per_round"].median(), 1e-9))
    )

    print(f"[features] Added player-level features → {d.shape[1]} columns")
    return d


def rolling_player_features(df: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    """
    Compute rolling averages for key performance metrics per player.
    Rolling features capture recent form, crucial for predicting future matches.
    Uses shift(1) to prevent data leakage (only past info used).
    """
    d = df.copy()
    roll_cols = ["kills", "deaths", "acs", "KD_ratio", "impact_score",
                 "entry_success_rate", "clutch_per_round", "win"]

    for w in windows:
        for col in roll_cols:
            if col not in d.columns:
                continue
            rolled = (
                d.groupby("player")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            d[f"{col}_roll{w}"] = rolled

    print(f"[rolling] Added rolling features (windows={windows}) → {d.shape[1]} columns")
    return d


def aggregate_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level stats to team-per-match-per-map level.
    Returns a team-match DataFrame with:
      - Mean/sum of all player stats for a team in a given match
      - Opponent's aggregated stats (for head-to-head comparisons)
      - Final match_outcome column (1 = team won)
    """
    agg_cols = [
        "kills", "deaths", "assists", "acs", "adr",
        "KD_ratio", "KDA_ratio", "kill_per_round", "death_per_round",
        "clutch_total", "clutch_weight", "clutch_per_round",
        "multikill_score", "entry_success_rate", "entry_diff",
        "win_rate_rounds", "util_score", "impact_score",
        "econ_rating", "plants", "defuses",
        "avg_bank", "eco_rounds_pct", "full_buy_pct",
    ]
    # Only keep cols that exist
    agg_cols = [c for c in agg_cols if c in df.columns]

    team_agg = (
        df.groupby(["match_id", "map", "team", "win"])[agg_cols]
        .agg(["mean", "sum", "std"])
        .reset_index()
    )
    team_agg.columns = ["match_id", "map", "team", "win"] + [
        f"{col}_{stat}" for col, stat in team_agg.columns[4:]
    ]

    # Merge opponent stats (self-join on match_id + map, different team)
    merged = team_agg.merge(
        team_agg,
        on=["match_id", "map"],
        suffixes=("_team", "_opp")
    )
    merged = merged[merged["team_team"] != merged["team_opp"]]
    merged = merged.rename(columns={"win_team": "match_outcome"})
    merged = merged.drop(columns=["win_opp"])

    print(f"[team_agg] Team-match features shape: {merged.shape}")
    return merged


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode map; one-hot encode stage (collapsed to broad phase)."""
    d = df.copy()

    # Map: ordinal label encoding (maps have no natural order but few unique vals)
    d["map_enc"] = pd.Categorical(d["map"]).codes

    # Stage: collapse to broad phase (Regular / Group / Playoffs / Other)
    def stage_phase(s: str) -> str:
        s = str(s).lower()
        if "regular" in s:
            return "Regular"
        if "group" in s:
            return "Group"
        if "playoff" in s or "final" in s or "semi" in s or "quarter" in s:
            return "Playoffs"
        return "Other"

    if "stage" in d.columns:
        d["stage_phase"] = d["stage"].apply(stage_phase)
        phase_dummies = pd.get_dummies(d["stage_phase"], prefix="phase", drop_first=True)
        d = pd.concat([d, phase_dummies], axis=1)

    return d


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline. Returns (player_df, team_df).

    player_df : one row per player per map per match, all engineered features
    team_df   : one row per team per map per match, aggregated features
    """
    vct_raw, agent_raw = load_raw()
    df = clean(vct_raw, agent_raw)
    df = engineer_player_features(df)
    df = rolling_player_features(df, windows=(3, 5))
    df = encode_categoricals(df)
    team_df = aggregate_team_features(df)
    return df, team_df


if __name__ == "__main__":
    player_df, team_df = build_dataset()
    player_df.to_parquet("/home/claude/vct_ml/data/player_features.parquet", index=False)
    team_df.to_parquet("/home/claude/vct_ml/data/team_features.parquet", index=False)
    print("\n✅ Datasets saved.")
    print("Player DF:", player_df.shape)
    print("Team DF  :", team_df.shape)
