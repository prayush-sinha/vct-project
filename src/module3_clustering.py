"""
Module 3 — Player Playstyle Clustering
========================================
Groups professional VCT players into behavioural archetypes using
unsupervised learning, then visualises the clusters in 2-D.

Pipeline
--------
1. Aggregate player-level features (career averages, min 5 maps)
2. Scale features → PCA whitening
3. KMeans (k determined by Elbow + Silhouette)
4. Label each cluster with a meaningful esports archetype name
5. Visualise with PCA-2D and t-SNE-2D
6. DBSCAN noise detection to find outlier players
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIG_DIR    = Path(__file__).parent.parent / "outputs" / "figures"

CLUSTER_FEATURES = [
    # Fragging profile
    "kill_per_round", "death_per_round", "KD_ratio",
    "acs",
    # Entry / aggression
    "entry_success_rate", "entry_diff",
    "first_kills",
    # Clutch / impact
    "clutch_per_round", "clutch_weight",
    "multikill_score",
    # Support / utility
    "util_score", "assists",
    "defuses", "plants",
    # Economy
    "econ_rating", "eco_rounds_pct", "full_buy_pct",
    # Headshot precision
    "hs_pct",
    # KAST (survival)
    "kast_pct",
]

# Esports-meaningful archetype labels (applied after inspecting cluster centroids)
ARCHETYPE_LABELS = {
    0: "⚔️ Entry Fragger",
    1: "🧠 Strategic IGL/Support",
    2: "🎯 Clutch Specialist",
    3: "🛡️ Anchor / Sentinel",
    4: "🚀 Hybrid Duelist",
}
ARCHETYPE_COLORS = {
    0: "#E84855",   # red — aggressive
    1: "#3E92CC",   # blue — strategic
    2: "#F4A261",   # orange — clutch
    3: "#57A773",   # green — defensive
    4: "#A663CC",   # purple — hybrid
}


def load_player_profiles(min_maps: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    df = pd.read_parquet(DATA_DIR / "player_features.parquet")
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]

    # Career averages per player
    profile = df.groupby("player")[feats + ["win"]].agg(
        {**{f: "mean" for f in feats}, "win": "mean"}
    ).reset_index().rename(columns={"win": "win_rate"})

    # Map count + most-common team per player
    n_maps = df.groupby("player").size().rename("n_maps_played").reset_index()
    team_mode = (
        df.groupby("player")["team"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    profile = profile.merge(n_maps, on="player")
    profile = profile.merge(team_mode, on="player")
    profile = profile[profile["n_maps_played"] >= min_maps].copy()
    profile = profile.dropna(subset=feats)

    print(f"[Module3] {len(profile)} players eligible for clustering ({min_maps}+ maps)")
    return df, profile, feats


def optimal_k(X_scaled: np.ndarray, k_range=(2, 9)) -> int:
    """Elbow + Silhouette to find optimal K."""
    inertias, silhouettes = [], []
    ks = range(k_range[0], k_range[1] + 1)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, "o-", color="#E84855")
    axes[0].set(xlabel="K", ylabel="Inertia", title="Elbow Method")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ks, silhouettes, "s-", color="#3E92CC")
    axes[1].set(xlabel="K", ylabel="Silhouette Score", title="Silhouette Analysis")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "module3_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_k = ks[int(np.argmax(silhouettes))]
    print(f"[Module3] Optimal K = {best_k} (Silhouette = {max(silhouettes):.4f})")
    return best_k


def fit_clusters(X_scaled: np.ndarray, profile: pd.DataFrame, feats: list, k: int):
    """Fit KMeans, assign archetype labels by centroid inspection."""
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    labels = km.fit_predict(X_scaled)
    profile = profile.copy()
    profile["cluster_raw"] = labels

    # Centroid DataFrame for interpretation
    centroids = pd.DataFrame(km.cluster_centers_, columns=feats)
    print("\n── Cluster Centroids (z-scores) ───────────────────────────────")
    key_cols = ["kill_per_round", "entry_success_rate", "clutch_per_round",
                "util_score", "econ_rating", "hs_pct", "kast_pct"]
    key_cols = [c for c in key_cols if c in centroids.columns]
    print(centroids[key_cols].round(2).to_string())

    # Auto-map cluster IDs to archetypes based on highest distinguishing feature
    # We use a heuristic: rank clusters on selected axes
    archetype_map = _assign_archetypes(centroids, k)
    profile["cluster"] = profile["cluster_raw"].map(archetype_map)
    profile["archetype"] = profile["cluster"].map(ARCHETYPE_LABELS)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    print(f"\n[Module3] KMeans (k={k}) → Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    return km, profile, archetype_map


def _assign_archetypes(centroids: pd.DataFrame, k: int) -> dict:
    """
    Map raw cluster IDs to our archetype IDs (0-4) using a priority heuristic.
    Works even when k < 5 by merging unused archetypes.
    """
    scores = pd.DataFrame(index=range(len(centroids)))
    if "kill_per_round" in centroids.columns:
        scores["entry_score"] = (centroids["kill_per_round"] +
                                 centroids.get("entry_success_rate", 0))
    if "util_score" in centroids.columns:
        scores["support_score"] = centroids["util_score"] + centroids.get("assists", 0)
    if "clutch_per_round" in centroids.columns:
        scores["clutch_score"] = centroids["clutch_per_round"]
    if "kast_pct" in centroids.columns:
        scores["anchor_score"] = centroids["kast_pct"] - centroids.get("kill_per_round", 0)
    if "hs_pct" in centroids.columns:
        scores["hybrid_score"] = centroids["hs_pct"]

    score_cols = list(scores.columns)
    archetype_names = ["entry_score", "support_score", "clutch_score", "anchor_score", "hybrid_score"]
    archetype_ids   = [0, 1, 2, 3, 4]

    mapping = {}
    assigned = set()
    for arch_name, arch_id in zip(archetype_names, archetype_ids):
        if arch_name not in scores.columns:
            continue
        remaining = [i for i in scores.index if i not in assigned]
        if not remaining:
            break
        best = scores.loc[remaining, arch_name].idxmax()
        mapping[best] = arch_id
        assigned.add(best)
    # Any cluster left → hybrid
    for i in range(len(centroids)):
        if i not in mapping:
            mapping[i] = 4
    return mapping


def add_pca_tsne(X_scaled: np.ndarray, profile: pd.DataFrame) -> pd.DataFrame:
    """Add 2-D PCA and t-SNE coordinates to the profile DataFrame."""
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    profile["pca1"] = pca_coords[:, 0]
    profile["pca2"] = pca_coords[:, 1]
    print(f"[Module3] PCA explains {pca.explained_variance_ratio_.sum():.2%} of variance")

    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=42, init="pca", learning_rate="auto")
    tsne_coords = tsne.fit_transform(X_scaled)
    profile["tsne1"] = tsne_coords[:, 0]
    profile["tsne2"] = tsne_coords[:, 1]
    return profile, pca


def detect_outliers(X_scaled: np.ndarray, profile: pd.DataFrame) -> pd.DataFrame:
    """DBSCAN to flag statistical outliers / one-of-a-kind players."""
    db = DBSCAN(eps=1.5, min_samples=3)
    noise = db.fit_predict(X_scaled)
    profile["is_outlier"] = (noise == -1)
    n_out = profile["is_outlier"].sum()
    print(f"[Module3] DBSCAN found {n_out} outlier players")
    return profile


def plot_clusters(profile: pd.DataFrame, fig_dir: Path):
    """2-panel figure: PCA scatter + t-SNE scatter, coloured by archetype."""
    archetypes_present = sorted(profile["cluster"].unique())
    palette = {a: ARCHETYPE_COLORS.get(a, "#888") for a in archetypes_present}

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("VCT Player Playstyle Archetypes", fontsize=15, fontweight="bold", y=1.01)

    for ax, (x_col, y_col, title) in zip(axes, [
        ("pca1", "pca2", "PCA Projection"),
        ("tsne1", "tsne2", "t-SNE Projection"),
    ]):
        for arch_id in archetypes_present:
            sub = profile[profile["cluster"] == arch_id]
            label = ARCHETYPE_LABELS.get(arch_id, f"Cluster {arch_id}")
            ax.scatter(sub[x_col], sub[y_col],
                       c=palette[arch_id], label=label,
                       s=60, alpha=0.7, edgecolors="white", linewidths=0.3)
            # Annotate top 3 impact players per cluster
            if "pred_impact" not in sub.columns and "n_maps_played" in sub.columns:
                top3 = sub.nlargest(3, "n_maps_played")
            else:
                try:
                    lb = pd.read_csv(DATA_DIR / "player_leaderboard.csv")
                    top3 = sub[sub["player"].isin(lb.head(30)["player"])].head(3)
                except Exception:
                    top3 = sub.head(3)
            for _, row in top3.iterrows():
                ax.annotate(row["player"],
                            (row[x_col], row[y_col]),
                            fontsize=6, ha="left",
                            xytext=(3, 3), textcoords="offset points",
                            color="#333")

        # Outlier overlay
        out = profile[profile["is_outlier"]]
        ax.scatter(out[x_col], out[y_col], marker="x", c="black",
                   s=40, linewidths=1.2, zorder=5, label="Outlier")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_xlabel(x_col.upper())
        ax.set_ylabel(y_col.upper())
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = fig_dir / "module3_cluster_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module3] Saved → {path}")


def plot_cluster_radar(profile: pd.DataFrame, feats: list, fig_dir: Path):
    """Radar / spider chart showing centroid profile per archetype."""
    radar_feats = ["kill_per_round", "entry_success_rate", "clutch_per_round",
                   "util_score_norm", "kast_pct_norm", "hs_pct_norm"]

    # Normalise 0-1 for radar
    d = profile.copy()
    for col in ["util_score", "kast_pct", "hs_pct"]:
        if col in d.columns:
            mn, mx = d[col].min(), d[col].max()
            d[f"{col}_norm"] = (d[col] - mn) / (mx - mn + 1e-9)

    radar_feats = [f for f in radar_feats if f in d.columns]
    labels_clean = [f.replace("_norm", "").replace("_", " ").title() for f in radar_feats]
    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    archetypes = sorted(d["cluster"].unique())
    n_arch = len(archetypes)
    fig, axes = plt.subplots(1, n_arch, figsize=(4 * n_arch, 5),
                             subplot_kw=dict(polar=True))
    if n_arch == 1:
        axes = [axes]

    for ax, arch_id in zip(axes, archetypes):
        sub = d[d["cluster"] == arch_id]
        vals = sub[radar_feats].mean().tolist()
        vals += vals[:1]
        color = ARCHETYPE_COLORS.get(arch_id, "#888")
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_clean, fontsize=8)
        ax.set_yticks([])
        label = ARCHETYPE_LABELS.get(arch_id, f"Cluster {arch_id}")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=12)

    fig.suptitle("Archetype Radar Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "module3_radar_archetypes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Module3] Saved → {path}")


def print_cluster_stats(profile: pd.DataFrame):
    print("\n── Cluster Summary ─────────────────────────────────────────────")
    for arch_id, label in sorted(ARCHETYPE_LABELS.items()):
        sub = profile[profile["cluster"] == arch_id]
        if len(sub) == 0:
            continue
        print(f"\n  {label}  (n={len(sub)})")
        print(f"    KD: {sub['KD_ratio'].mean():.2f} | "
              f"ACS: {sub['acs'].mean():.0f} | "
              f"Clutch/round: {sub['clutch_per_round'].mean():.3f} | "
              f"Win rate: {sub['win_rate'].mean():.2%}")
        print(f"    Top players: {', '.join(sub.nlargest(5,'n_maps_played')['player'].tolist())}")


def run():
    df, profile, feats = load_player_profiles(min_maps=5)

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(profile[feats])

    # Optimal k
    k = optimal_k(X_scaled, k_range=(2, 8))
    k = min(k, 5)  # cap at 5 to keep archetypes interpretable

    # Cluster
    km, profile, archetype_map = fit_clusters(X_scaled, profile, feats, k)
    profile = detect_outliers(X_scaled, profile)
    profile, pca = add_pca_tsne(X_scaled, profile)

    # Save
    joblib.dump(km, MODELS_DIR / "kmeans_clusters.pkl")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "cluster_pca.pkl")
    profile.to_csv(DATA_DIR / "player_clusters.csv", index=False)

    plot_clusters(profile, FIG_DIR)
    plot_cluster_radar(profile, feats, FIG_DIR)
    print_cluster_stats(profile)

    return km, profile


if __name__ == "__main__":
    run()
