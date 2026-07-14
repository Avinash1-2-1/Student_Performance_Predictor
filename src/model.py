"""
model.py
--------
Defines the StudentPerformanceModel class:
    - a regression model (Linear Regression) to predict final_score
    - a classification model (Logistic Regression) to predict pass/fail
    - helper methods for training, evaluation, saving, and loading
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

FEATURES = ["study_hours_per_week", "attendance_percent", "previous_score"]


class StudentPerformanceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.regressor = LinearRegression()
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        X = df[FEATURES]
        y_reg = df["final_score"]
        y_clf = df["passed"]

        X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.regressor.fit(X_train_scaled, yreg_train)
        self.classifier.fit(X_train_scaled, yclf_train)
        self.is_trained = True

        # ---- Evaluation ----
        reg_preds = self.regressor.predict(X_test_scaled)
        clf_preds = self.classifier.predict(X_test_scaled)

        metrics = {
            "regression": {
                "MAE": mean_absolute_error(yreg_test, reg_preds),
                "RMSE": np.sqrt(mean_squared_error(yreg_test, reg_preds)),
                "R2": r2_score(yreg_test, reg_preds),
            },
            "classification": {
                "Accuracy": accuracy_score(yclf_test, clf_preds),
                "Precision": precision_score(yclf_test, clf_preds, zero_division=0),
                "Recall": recall_score(yclf_test, clf_preds, zero_division=0),
                "F1": f1_score(yclf_test, clf_preds, zero_division=0),
            },
            "feature_importance": dict(zip(FEATURES, self.regressor.coef_))
        }

        return metrics

    def predict(self, study_hours: float, attendance: float, previous_score: float):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() or load() first.")

        X_new = pd.DataFrame(
            [[study_hours, attendance, previous_score]],
            columns=FEATURES
        )
        X_scaled = self.scaler.transform(X_new)

        predicted_score = float(np.clip(self.regressor.predict(X_scaled)[0], 0, 100))
        pass_probability = float(self.classifier.predict_proba(X_scaled)[0][1])
        will_pass = bool(self.classifier.predict(X_scaled)[0])

        return {
            "predicted_final_score": round(predicted_score, 2),
            "will_pass": will_pass,
            "pass_probability": round(pass_probability, 4),
        }

    def save(self, path: str):
        joblib.dump({
            "scaler": self.scaler,
            "regressor": self.regressor,
            "classifier": self.classifier,
            "is_trained": self.is_trained
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.regressor = data["regressor"]
        self.classifier = data["classifier"]
        self.is_trained = data["is_trained"]
        return self
