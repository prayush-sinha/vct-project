"""
VCT Esports Analytics Dashboard — Streamlit App
================================================
Run with:
    streamlit run app/dashboard.py

Tabs:
  1. 🏟️  Match Prediction
  2. 🏆  Player Leaderboard
  3. 🎭  Playstyle Clusters
  4. 📊  Key Insights
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
MODELS  = BASE / "models"
FIGS    = BASE / "outputs" / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VCT Esports Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stMetric        { background:#1e1e2e; padding:12px; border-radius:8px; }
  .stMetricLabel   { color:#cdd6f4 !important; }
  .stMetricValue   { color:#cba6f7 !important; }
  [data-testid="stSidebar"] { background:#181825; }
  .main            { background:#11111b; }
  h1, h2, h3      { color:#cba6f7; }
  .metric-card    { background:#1e1e2e; border-radius:12px; padding:16px;
                    border:1px solid #313244; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_player_df():
    return pd.read_parquet(DATA / "player_features.parquet")

@st.cache_data
def load_team_df():
    return pd.read_parquet(DATA / "team_features.parquet")

@st.cache_data
def load_leaderboard():
    return pd.read_csv(DATA / "player_leaderboard.csv")

@st.cache_data
def load_clusters():
    return pd.read_csv(DATA / "player_clusters.csv")

@st.cache_resource
def load_models():
    return {
        "lgbm":   joblib.load(MODELS / "lgbm_match_predictor.pkl"),
        "xgb_m":  joblib.load(MODELS / "xgb_match_predictor.pkl"),
        "lr":     joblib.load(MODELS / "lr_match_predictor.pkl"),
        "xgb_r":  joblib.load(MODELS / "xgb_player_rater.pkl"),
        "km":     joblib.load(MODELS / "kmeans_clusters.pkl"),
        "scaler": joblib.load(MODELS / "cluster_scaler.pkl"),
    }

# ── Load everything ───────────────────────────────────────────────────────────
player_df  = load_player_df()
team_df    = load_team_df()
leaderboard = load_leaderboard()
clusters   = load_clusters()
models     = load_models()

ARCHETYPE_LABELS = {
    0: "⚔️ Entry Fragger",
    1: "🧠 Strategic IGL/Support",
    2: "🎯 Clutch Specialist",
    3: "🛡️ Anchor / Sentinel",
    4: "🚀 Hybrid Duelist",
}
ARCHETYPE_COLORS = {
    0: "#E84855", 1: "#3E92CC", 2: "#F4A261", 3: "#57A773", 4: "#A663CC"
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 VCT Analytics")
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.metric("Total Players",  f"{player_df['player'].nunique():,}")
    st.metric("Matches Analysed", f"{player_df['match_id'].nunique():,}")
    st.metric("Teams", f"{player_df['team'].nunique():,}")
    st.metric("Maps", f"{player_df['map'].nunique()}")
    st.markdown("---")
    st.markdown("Built with XGBoost · LightGBM · SHAP · sklearn")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏟️ Match Prediction",
    "🏆 Player Leaderboard",
    "🎭 Playstyle Clusters",
    "📊 Key Insights",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
PRED_FEATURES = [
    "kills_mean", "deaths_mean", "assists_mean",
    "acs_mean", "adr_mean",
    "KD_ratio_mean", "KDA_ratio_mean",
    "kills_mean_opp", "deaths_mean_opp", "acs_mean_opp",
    "KD_ratio_mean_opp", "impact_score_mean_opp",
    "entry_success_rate_mean_opp", "clutch_per_round_mean_opp",
]
PRED_FEATURES = [c for c in PRED_FEATURES if c in team_df.columns]

with tab1:
    st.title("🏟️ Match Outcome Prediction")
    st.markdown("""
    Configure team statistics to predict the probability of a **Win** using
    our ensemble of Logistic Regression, XGBoost, and LightGBM models.
    > **Model performance:** 94.1% accuracy · 0.988 ROC-AUC (5-fold CV)
    """)

    st.markdown("---")
    col_info, col_form = st.columns([1, 2])

    with col_info:
        st.markdown("#### 📋 How it works")
        st.info("""
        1. Enter your team's average stats for the map
        2. Enter the opponent's average stats
        3. Click **Predict** to get win probabilities
        from all three trained models
        """)
        st.markdown("#### 🎯 Model Ensemble")
        for m_name in ["Logistic Regression", "XGBoost", "LightGBM"]:
            st.markdown(f"- **{m_name}** — ~94% Accuracy")

    with col_form:
        st.markdown("#### Your Team's Stats")
        c1, c2, c3 = st.columns(3)
        kills     = c1.number_input("Avg Kills", 0, 50, 18)
        deaths    = c2.number_input("Avg Deaths", 0, 50, 14)
        assists   = c3.number_input("Avg Assists", 0, 20, 4)
        c1b, c2b, c3b = st.columns(3)
        acs       = c1b.number_input("Avg ACS", 0, 500, 220)
        adr       = c2b.number_input("Avg ADR", 0, 300, 140)
        kd        = c3b.number_input("K/D Ratio", 0.0, 5.0, 1.2, step=0.1)
        kda       = st.number_input("KDA Ratio", 0.0, 5.0, 1.3, step=0.1)

        st.markdown("#### Opponent Stats")
        d1, d2, d3 = st.columns(3)
        o_kills   = d1.number_input("Opp Avg Kills", 0, 50, 15)
        o_deaths  = d2.number_input("Opp Avg Deaths", 0, 50, 15)
        o_acs     = d3.number_input("Opp Avg ACS", 0, 500, 200)
        d1b, d2b, d3b = st.columns(3)
        o_kd      = d1b.number_input("Opp K/D", 0.0, 5.0, 1.0, step=0.1)
        o_impact  = d2b.number_input("Opp Impact Score", 0.0, 3.0, 1.0, step=0.1)
        o_entry   = d3b.number_input("Opp Entry SR", 0.0, 1.0, 0.5, step=0.05)

        predict_btn = st.button("🔮 Predict Match Outcome", type="primary",
                                use_container_width=True)

    if predict_btn:
        # Build input vector (fill missing features with training median)
        base = team_df[PRED_FEATURES].median().to_dict()
        user_vals = {
            "kills_mean": kills, "deaths_mean": deaths, "assists_mean": assists,
            "acs_mean": acs, "adr_mean": adr,
            "KD_ratio_mean": kd, "KDA_ratio_mean": kda,
            "kills_mean_opp": o_kills, "deaths_mean_opp": o_deaths,
            "acs_mean_opp": o_acs, "KD_ratio_mean_opp": o_kd,
            "impact_score_mean_opp": o_impact,
            "entry_success_rate_mean_opp": o_entry,
        }
        base.update({k: v for k, v in user_vals.items() if k in base})
        X_input = pd.DataFrame([base])[PRED_FEATURES]

        preds = {
            "Logistic Regression": models["lr"].predict_proba(X_input)[0, 1],
            "XGBoost":             models["xgb_m"].predict_proba(X_input)[0, 1],
            "LightGBM":            models["lgbm"].predict_proba(X_input)[0, 1],
        }
        ensemble = np.mean(list(preds.values()))

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Ensemble Win Prob", f"{ensemble:.1%}",
                   delta=f"{ensemble - 0.5:+.1%} vs 50/50")
        for (name, prob), col in zip(preds.items(), [mc2, mc3, mc4]):
            col.metric(name, f"{prob:.1%}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ensemble * 100,
            delta={"reference": 50, "valueformat": ".1f"},
            title={"text": "Win Probability (%)", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#cba6f7"},
                "steps": [
                    {"range": [0, 40],   "color": "#E84855"},
                    {"range": [40, 60],  "color": "#F4A261"},
                    {"range": [60, 100], "color": "#57A773"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.75, "value": 50},
            }
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

        verdict = "✅ **PREDICTED WIN**" if ensemble > 0.5 else "❌ **PREDICTED LOSS**"
        st.markdown(f"## {verdict} — Confidence: {abs(ensemble - 0.5) * 200:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PLAYER LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("🏆 Player Performance Leaderboard")
    st.markdown("""
    Predicted impact scores from our **XGBoost Regression** model (R² ≈ 1.00 CV).
    Players ranked by predicted impact score, derived from kills, clutches,
    entry success, economy efficiency and multi-kill rate.
    """)

    c_filt1, c_filt2, c_filt3 = st.columns(3)
    with c_filt1:
        teams = ["All Teams"] + sorted(leaderboard["team"].dropna().unique().tolist())
        sel_team = st.selectbox("Filter by Team", teams)
    with c_filt2:
        n_show = st.slider("Players to show", 10, min(100, len(leaderboard)), 30)
    with c_filt3:
        sort_by = st.selectbox("Sort by", ["pred_impact", "avg_acs", "avg_kd", "win_rate"])

    lb = leaderboard.copy()
    if sel_team != "All Teams":
        lb = lb[lb["team"] == sel_team]
    lb = lb.sort_values(sort_by, ascending=False).head(n_show).reset_index(drop=True)
    lb.index += 1

    # Horizontal bar chart
    fig = px.bar(
        lb, x="pred_impact", y="player", orientation="h",
        color="win_rate", color_continuous_scale="RdYlGn",
        hover_data=["team", "avg_acs", "avg_kd", "maps_played"],
        title=f"Top {n_show} Players by Predicted Impact Score",
        labels={"pred_impact": "Impact Score", "player": "Player",
                "win_rate": "Win Rate"},
    )
    fig.update_layout(
        height=max(400, n_show * 22),
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cdd6f4",
        coloraxis_colorbar=dict(title="Win Rate"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: ACS vs Impact
    fig2 = px.scatter(
        leaderboard.head(200), x="avg_acs", y="pred_impact",
        color="win_rate", size="maps_played",
        hover_name="player", hover_data=["team", "avg_kd"],
        color_continuous_scale="RdYlGn",
        title="Predicted Impact vs Traditional ACS (top-200 players)",
        labels={"avg_acs": "Average ACS", "pred_impact": "Predicted Impact"},
        size_max=20,
    )
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
    st.plotly_chart(fig2, use_container_width=True)

    # Data table
    st.markdown("#### 📋 Full Table")
    display_cols = ["player", "team", "maps_played", "avg_acs", "avg_kd",
                    "avg_vlr", "pred_impact", "win_rate", "avg_clutch"]
    display_cols = [c for c in display_cols if c in lb.columns]
    st.dataframe(
        lb[display_cols].style.format({
            "avg_acs": "{:.0f}", "avg_kd": "{:.2f}", "avg_vlr": "{:.2f}",
            "pred_impact": "{:.2f}", "win_rate": "{:.1%}",
            "avg_clutch": "{:.3f}",
        }).background_gradient(subset=["pred_impact"], cmap="RdYlGn"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PLAYSTYLE CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("🎭 Player Playstyle Archetypes")
    st.markdown("""
    Players clustered via **KMeans** on 18 behavioural features and visualised
    with **PCA** and **t-SNE** dimensionality reduction.
    """)

    if clusters is not None and "cluster" in clusters.columns:
        clusters["archetype"] = clusters["cluster"].map(ARCHETYPE_LABELS).fillna("Unknown")
        clusters["color"]     = clusters["cluster"].map(ARCHETYPE_COLORS).fillna("#888")

        proj_choice = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
        x_col = "pca1" if proj_choice == "PCA" else "tsne1"
        y_col = "pca2" if proj_choice == "PCA" else "tsne2"

        # Only pass hover columns that actually exist in this CSV
        hover_cols = [c for c in ["team", "win_rate", "KD_ratio", "clutch_per_round"]
                      if c in clusters.columns]
        fig = px.scatter(
            clusters, x=x_col, y=y_col,
            color="archetype",
            color_discrete_map={v: ARCHETYPE_COLORS.get(k, "#888")
                                for k, v in ARCHETYPE_LABELS.items()},
            hover_name="player",
            hover_data=hover_cols,
            title=f"Player Archetypes — {proj_choice} Projection",
            symbol="is_outlier",
            symbol_map={True: "x", False: "circle"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8,
                                      line=dict(width=0.5, color="#11111b")))
        fig.update_layout(
            height=600, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4",
            legend=dict(title="Archetype"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster stats
        st.markdown("#### 📋 Cluster Profiles")
        cols_show = ["archetype", "KD_ratio", "clutch_per_round",
                     "entry_success_rate", "util_score", "win_rate"]
        cols_show = [c for c in cols_show if c in clusters.columns]
        summary = clusters.groupby("archetype")[
            [c for c in cols_show if c != "archetype"]
        ].mean().reset_index()
        summary.insert(1, "Players", clusters.groupby("archetype").size().values)
        st.dataframe(
            summary.style.format({
                c: "{:.3f}" for c in summary.columns if c not in ["archetype", "Players"]
            }).background_gradient(subset=["KD_ratio", "win_rate"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Player search
        st.markdown("#### 🔍 Find a Player's Archetype")
        player_pick = st.selectbox("Select player",
                                   sorted(clusters["player"].unique()))
        p = clusters[clusters["player"] == player_pick].iloc[0]
        st.success(f"**{player_pick}** ({p.get('team','')}) → **{p['archetype']}**")
        subcols = ["KD_ratio", "clutch_per_round", "entry_success_rate",
                   "util_score", "win_rate"]
        for col in [c for c in subcols if c in clusters.columns]:
            val = p[col]
            mean_val = clusters[col].mean()
            st.metric(col.replace("_", " ").title(),
                      f"{val:.3f}", delta=f"{val - mean_val:+.3f} vs avg")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — KEY INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.title("📊 Key Insights & EDA")

    # Map win rates
    st.markdown("### 🗺️ Win Rate Heatmap by Map")
    map_team = player_df.groupby(["map", "team"])["win"].mean().reset_index()
    top_teams = player_df["team"].value_counts().head(12).index.tolist()
    map_pivot = map_team[map_team["team"].isin(top_teams)].pivot(
        index="team", columns="map", values="win"
    ).fillna(0)
    fig_hm = px.imshow(
        map_pivot, color_continuous_scale="RdYlGn",
        title="Win Rate by Team × Map (Top 12 Teams)",
        aspect="auto", zmin=0, zmax=1,
        text_auto=".0%",
    )
    fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4",
                          height=450)
    st.plotly_chart(fig_hm, use_container_width=True)

    # KD distribution by result
    st.markdown("### 📈 K/D Ratio Distribution — Win vs Loss")
    df_kd = player_df[["KD_ratio", "result"]].dropna()
    df_kd = df_kd[df_kd["KD_ratio"] < 5]
    fig_kd = px.histogram(
        df_kd, x="KD_ratio", color="result", nbins=60, barmode="overlay",
        color_discrete_map={"Win": "#57A773", "Loss": "#E84855"},
        opacity=0.7, title="K/D Ratio Distribution by Match Result",
    )
    fig_kd.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
    st.plotly_chart(fig_kd, use_container_width=True)

    # ACS vs Win rate by map
    st.markdown("### 🎯 Average ACS vs Win Rate by Map")
    map_stats = player_df.groupby("map").agg(
        avg_acs=("acs", "mean"),
        win_rate=("win", "mean"),
        n=("win", "count"),
    ).reset_index()
    fig_acs = px.scatter(
        map_stats, x="avg_acs", y="win_rate", text="map",
        size="n", color="win_rate", color_continuous_scale="RdYlGn",
        title="Average ACS vs Win Rate by Map",
        size_max=35,
    )
    fig_acs.update_traces(textposition="top center", textfont_size=10)
    fig_acs.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4",
                           height=450)
    st.plotly_chart(fig_acs, use_container_width=True)

    # First kills impact
    st.markdown("### 🔫 Entry Duel Impact")
    fk_bins = pd.cut(player_df["first_kills"].clip(0, 10),
                     bins=[-1, 0, 2, 5, 10],
                     labels=["0 FKs", "1-2 FKs", "3-5 FKs", "6+ FKs"])
    fk_win = player_df.copy()
    fk_win["fk_tier"] = fk_bins
    fk_summary = fk_win.groupby("fk_tier", observed=True)["win"].mean().reset_index()
    fk_summary.columns = ["First Kill Tier", "Win Rate"]
    fig_fk = px.bar(
        fk_summary, x="First Kill Tier", y="Win Rate",
        color="Win Rate", color_continuous_scale="RdYlGn",
        title="Win Rate by First Kills per Map",
        text_auto=".1%",
    )
    fig_fk.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4",
                          yaxis_range=[0, 1])
    st.plotly_chart(fig_fk, use_container_width=True)

    # Summary KPIs
    st.markdown("---")
    st.markdown("### 🔑 Dataset Highlights")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Map Performances", f"{len(player_df):,}")
    k2.metric("Overall Win Rate", f"{player_df['win'].mean():.1%}")
    k3.metric("Avg ACS", f"{player_df['acs'].mean():.0f}")
    k4.metric("Avg K/D", f"{player_df['KD_ratio'].mean():.2f}")
    k5.metric("Tournaments", f"{player_df['tournament'].nunique()}")
