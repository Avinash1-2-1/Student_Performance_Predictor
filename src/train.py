"""
train.py
--------
Loads the dataset, trains the StudentPerformanceModel, prints evaluation
metrics, saves diagnostic plots, and persists the trained model to disk.

Usage:
    python3 src/train.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from model import StudentPerformanceModel, FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "models")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} records loaded.\n")

    print("Training model...")
    model = StudentPerformanceModel()
    metrics = model.train(df)

    print("\n=== Regression metrics (final_score prediction) ===")
    for k, v in metrics["regression"].items():
        print(f"  {k}: {v:.3f}")

    print("\n=== Classification metrics (pass/fail prediction) ===")
    for k, v in metrics["classification"].items():
        print(f"  {k}: {v:.3f}")

    print("\n=== Feature influence (regression coefficients, standardized) ===")
    for feat, coef in metrics["feature_importance"].items():
        print(f"  {feat}: {coef:.3f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ---- Diagnostic plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, feat in zip(axes, FEATURES):
        ax.scatter(df[feat], df["final_score"], alpha=0.3, s=15, color="#4C72B0")
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Final Score")
        ax.set_title(f"Final Score vs {feat.replace('_', ' ').title()}")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "feature_relationships.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Diagnostic plot saved to {plot_path}")


if __name__ == "__main__":
    main()
