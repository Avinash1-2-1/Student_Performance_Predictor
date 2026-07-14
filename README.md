# Student Performance Predictor

A machine learning project that predicts a student's **final score** and
**pass/fail outcome** based on three inputs:

- **Study hours per week**
- **Attendance percentage**
- **Previous exam score**

## How it works

1. **Dataset** (`data/generate_dataset.py`): Since no real dataset was
   provided, this generates a synthetic dataset of 1000 students. Each
   feature is sampled from a realistic distribution, and the target score is
   built from a weighted combination of the three features plus random noise
   — so the model has a genuine (not trivial) pattern to learn, similar to
   real academic data.

2. **Model** (`src/model.py`): Two models are trained on standardized
   features:
   - **Linear Regression** → predicts the continuous final score (0-100)
   - **Logistic Regression** → predicts pass/fail (pass = score ≥ 40)

3. **Training** (`src/train.py`): Trains both models, evaluates them on a
   held-out test set, prints metrics, saves the trained model to
   `models/performance_model.pkl`, and generates a diagnostic plot showing
   how each feature relates to final score.

4. **Prediction** (`src/predict.py`): Loads the trained model and predicts
   an outcome for a new student, either interactively or via command-line
   arguments.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset
```bash
python3 data/generate_dataset.py
```

### 2. Train the model
```bash
python3 src/train.py
```
This prints regression/classification metrics and saves:
- `models/performance_model.pkl` — the trained model
- `models/feature_relationships.png` — diagnostic scatter plots

### 3. Predict for a new student

**Option A — Web UI (recommended for presentations)**
```bash
python3 app.py
```
Then open **http://127.0.0.1:5000** in a browser. Enter study hours,
attendance, and previous score using the sliders and click **Predict
outcome** to see a report-card style result: predicted score, letter
grade, and pass probability.

**Option B — Command line**

Interactive mode:
```bash
python3 src/predict.py
```

Direct arguments:
```bash
python3 src/predict.py --study_hours 12 --attendance 85 --previous_score 70
```

Example output:
```
--- Prediction Result ---
  Predicted final score : 72.1 / 100
  Pass probability      : 100.0%
  Likely outcome        : PASS
```

## Model Performance (on synthetic test data)

| Task                        | Metric     | Value |
|------------------------------|-----------|-------|
| Final score (regression)     | R²        | ~0.79 |
| Final score (regression)     | MAE       | ~4.6 points |
| Pass/Fail (classification)   | Accuracy  | ~94.5% |
| Pass/Fail (classification)   | F1 Score  | ~0.97 |

## Project structure

```
student_performance_predictor/
├── app.py                      # Flask web app (UI)
├── templates/
│   └── index.html              # chalkboard/report-card UI
├── static/
│   ├── style.css
│   └── script.js
├── data/
│   ├── generate_dataset.py     # creates the synthetic dataset
│   └── student_data.csv        # generated dataset (1000 records)
├── src/
│   ├── model.py                # StudentPerformanceModel class
│   ├── train.py                # training + evaluation script
│   └── predict.py               # CLI prediction script
├── models/
│   ├── performance_model.pkl   # saved trained model
│   └── feature_relationships.png
├── requirements.txt
└── README.md
```

## Using your own real data

Replace `data/student_data.csv` with your own CSV as long as it has these
columns: `study_hours_per_week`, `attendance_percent`, `previous_score`,
`final_score`. The `passed` column is optional — `train.py` can be modified
to derive it automatically (`final_score >= 40`), or you can supply your own
pass/fail threshold.



