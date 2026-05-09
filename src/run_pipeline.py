"""
VCT Esports Analytics — Master Pipeline
=========================================
Runs the full project end-to-end:
  data_processing → module1 → module2 → module3 → module4

Run from project root:
    python src/run_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import data_processing as dp
import module1_match_prediction as m1
import module2_player_rating as m2
import module3_clustering as m3
import module4_clutch_economy as m4

BANNER = """
╔══════════════════════════════════════════════════════╗
║   VCT Esports Analytics — End-to-End ML Pipeline    ║
╚══════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    print("━" * 56)
    print("STEP 1: Data Processing & Feature Engineering")
    print("━" * 56)
    player_df, team_df = dp.build_dataset()

    print("\n" + "━" * 56)
    print("STEP 2: Module 1 — Match Outcome Prediction")
    print("━" * 56)
    m1_results = m1.run()

    print("\n" + "━" * 56)
    print("STEP 3: Module 2 — Player Performance Rating")
    print("━" * 56)
    m2_models, leaderboard = m2.run()

    print("\n" + "━" * 56)
    print("STEP 4: Module 3 — Player Playstyle Clustering")
    print("━" * 56)
    km, profile = m3.run()

    print("\n" + "━" * 56)
    print("STEP 5: Module 4 — Clutch & Economy Analysis")
    print("━" * 56)
    clutch_models = m4.run()

    print("\n" + "═" * 56)
    print("✅ Pipeline complete!")
    print(f"   Figures   → vct_ml/outputs/figures/")
    print(f"   Models    → vct_ml/models/")
    print(f"   Data      → vct_ml/data/")
    print("═" * 56)


if __name__ == "__main__":
    main()
