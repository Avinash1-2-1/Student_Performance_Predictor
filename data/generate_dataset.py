"""
generate_dataset.py
--------------------
Generates a synthetic but realistic dataset of student records based on:
    - study_hours_per_week   (numeric, 0-40)
    - attendance_percent     (numeric, 40-100)
    - previous_score         (numeric, 0-100)  -> score from prior exam/term

Target:
    - final_score             (numeric, 0-100)   -> continuous performance score
    - passed                  (0/1)              -> derived: 1 if final_score >= 40

The relationship is built with realistic weights + noise so the model
has something meaningful to learn (not a trivial linear identity).
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_STUDENTS = 1000

def generate_dataset(n=N_STUDENTS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # ---- Feature generation ----
    study_hours = np.clip(rng.normal(loc=15, scale=7, size=n), 0, 40)
    attendance = np.clip(rng.normal(loc=80, scale=12, size=n), 40, 100)
    previous_score = np.clip(rng.normal(loc=65, scale=15, size=n), 0, 100)

    # ---- Target generation (weighted combination + noise) ----
    # Weights reflect real-world intuition:
    #   previous performance matters most, then study hours, then attendance
    noise = rng.normal(loc=0, scale=6, size=n)

    final_score = (
        0.35 * previous_score +
        1.3 * study_hours +
        0.25 * attendance +
        noise
    )

    # Rescale/clip to a realistic 0-100 exam score range
    final_score = np.clip(final_score, 0, 100)

    passed = (final_score >= 40).astype(int)

    df = pd.DataFrame({
        "study_hours_per_week": np.round(study_hours, 1),
        "attendance_percent": np.round(attendance, 1),
        "previous_score": np.round(previous_score, 1),
        "final_score": np.round(final_score, 1),
        "passed": passed
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} student records -> {out_path}")
    print(df.describe())
