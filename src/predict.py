"""
predict.py
----------
Command-line tool to predict a student's final score and pass/fail outcome
using the trained model.

Usage (interactive):
    python3 src/predict.py

Usage (arguments):
    python3 src/predict.py --study_hours 12 --attendance 85 --previous_score 70
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(__file__))
from model import StudentPerformanceModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run `python3 src/train.py` first.")
        sys.exit(1)
    model = StudentPerformanceModel()
    model.load(MODEL_PATH)
    return model


def run_prediction(model, study_hours, attendance, previous_score):
    result = model.predict(study_hours, attendance, previous_score)
    print("\n--- Prediction Result ---")
    print(f"  Predicted final score : {result['predicted_final_score']} / 100")
    print(f"  Pass probability      : {result['pass_probability'] * 100:.1f}%")
    print(f"  Likely outcome        : {'PASS' if result['will_pass'] else 'FAIL'}")
    return result


def interactive_mode(model):
    print("Enter student details to predict performance:")
    study_hours = float(input("  Study hours per week (0-40): "))
    attendance = float(input("  Attendance percentage (0-100): "))
    previous_score = float(input("  Previous exam score (0-100): "))
    run_prediction(model, study_hours, attendance, previous_score)


def main():
    parser = argparse.ArgumentParser(description="Predict student performance.")
    parser.add_argument("--study_hours", type=float, help="Study hours per week")
    parser.add_argument("--attendance", type=float, help="Attendance percentage")
    parser.add_argument("--previous_score", type=float, help="Previous exam score")
    args = parser.parse_args()

    model = load_model()

    if args.study_hours is not None and args.attendance is not None and args.previous_score is not None:
        run_prediction(model, args.study_hours, args.attendance, args.previous_score)
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
