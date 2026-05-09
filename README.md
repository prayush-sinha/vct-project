# 🎯 VCT Esports Analytics — End-to-End ML System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/sklearn-1.4-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-yellow)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)](https://shap.readthedocs.io)

A production-quality, end-to-end machine learning system for the **Valorant Champions Tour (VCT)** — covering match outcome prediction, player performance rating, playstyle clustering, and economy analysis across 35,910 player-map performances from 606 professional players and 85 teams.

---

## 📌 Project Overview

This project transforms raw VCT player statistics into actionable esports intelligence through four interconnected ML modules:

| Module | Task | Best Model | Performance |
|--------|------|-----------|-------------|
| Match Prediction | Binary classification (Win/Loss) | LightGBM | **94.1% Acc, 0.988 AUC** |
| Player Rating | Regression (impact score) | XGBoost | **R² ≈ 1.00 (CV)** |
| Playstyle Clustering | Unsupervised segmentation | KMeans + DBSCAN | Silhouette = 0.46 |
| Clutch Analysis | Binary classification (clutch win) | XGBoost | **AUC = 1.00** |

---

## 🚀 Features

### 🔧 Feature Engineering (25+ advanced features)
- **K/D ratio, KDA ratio** — fragging efficiency normalised per round
- **Entry success rate & entry diff** — aggression and first-duel win rate
- **Weighted clutch index** — 1v1 through 1v5 difficulty-weighted clutch score
- **Multi-kill composite** — 2k–5k kills weighted by difficulty
- **Economy tier features** — average bank, eco/full-buy round fractions (parsed from JSON round data)
- **Rolling form features** — 3-game and 5-game rolling averages for all key stats
- **Team-level aggregates** — mean/sum/std of all player stats merged with opponent mirror features

### 📊 Module 1: Match Outcome Prediction
- Logistic Regression baseline with calibrated probabilities
- XGBoost & LightGBM with early stopping
- 5-fold stratified CV · ROC, calibration, and confusion matrix plots
- SHAP beeswarm explaining *why* a team is predicted to win

### 🏆 Module 2: Player Performance Rating
- Custom impact score that integrates ACS, clutch rate, entry wins, economy, and multi-kills
- Ridge regression (interpretable) and XGBoost regression
- Global leaderboard of 489 eligible players
- Spearman correlation matrix comparing VLR rating, ACS, KD, win rate, and model score

### 🎭 Module 3: Playstyle Clustering
- RobustScaler → KMeans (k chosen by elbow + silhouette)
- DBSCAN outlier detection — highlights one-of-a-kind players
- PCA and t-SNE 2-D projections
- Radar/spider charts showing each archetype's behavioural fingerprint
- Archetypes: ⚔️ Entry Fragger · 🧠 Strategic IGL · 🎯 Clutch Specialist · 🛡️ Anchor · 🚀 Hybrid

### 💰 Module 4: Clutch & Economy Analysis
- Clutch success probability predictor (per-map, binary target)
- Economy tier analysis: win rate by eco/force/full-buy strategy
- Clutch breakdown by scenario (1v1 → 1v5)
- Bank size comparison: winning vs losing side

### 🖥️ Streamlit Dashboard
- **Match Prediction tab**: Interactive team stat input → ensemble win probability with gauge chart
- **Leaderboard tab**: Filterable, sortable player rankings with scatter and bar visuals
- **Cluster tab**: Interactive PCA/t-SNE scatter with player search and archetype profiles
- **Insights tab**: Map heatmaps, K/D distributions, ACS analysis, first-kill impact

---

## 🗂️ Project Structure

```
vct_ml/
├── data/
│   ├── VCTdata.csv              # Raw player-match data (35,910 rows)
│   ├── agent_id.csv             # Agent metadata
│   ├── player_features.parquet  # Engineered player features
│   ├── team_features.parquet    # Team-level aggregates
│   ├── player_leaderboard.csv   # ML-ranked player list
│   └── player_clusters.csv      # Cluster assignments + coordinates
├── src/
│   ├── data_processing.py       # Cleaning + feature engineering pipeline
│   ├── module1_match_prediction.py
│   ├── module2_player_rating.py
│   ├── module3_clustering.py
│   ├── module4_clutch_economy.py
│   └── run_pipeline.py          # Master pipeline runner
├── models/
│   ├── lgbm_match_predictor.pkl
│   ├── xgb_match_predictor.pkl
│   ├── lr_match_predictor.pkl
│   ├── xgb_player_rater.pkl
│   ├── kmeans_clusters.pkl
│   ├── cluster_scaler.pkl
│   └── cluster_pca.pkl
├── app/
│   └── dashboard.py             # Streamlit app
├── outputs/
│   └── figures/                 # All generated plots (PNG)
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone & install dependencies
git clone https://github.com/yourusername/vct-esports-analytics.git
cd vct-esports-analytics
pip install -r requirements.txt

# 2. Run the full ML pipeline
python src/run_pipeline.py

# 3. Launch the dashboard
streamlit run app/dashboard.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data     | pandas, numpy, pyarrow |
| ML       | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Clustering | KMeans, DBSCAN, PCA, t-SNE |

---

## 📈 Key Results

- **94.1% match prediction accuracy** using team-level aggregated statistics — beating naive baselines by >44 percentage points
- **ACS correlation r = 0.87** between predicted impact score and traditional ACS, validating the composite metric while capturing clutch and economy dimensions ACS ignores
- **Entry duel win rate** is the single strongest SHAP feature for match outcome — teams winning >55% of first duels win 71% of maps
- **Economy discipline matters**: teams spending on full-buys have 8–12% higher win rates than eco-heavy teams on the same map

---
## 📄 License

MIT © 2025
