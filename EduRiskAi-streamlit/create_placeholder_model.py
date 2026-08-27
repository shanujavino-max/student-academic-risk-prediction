"""
create_placeholder_model.py

DEVELOPMENT / TESTING UTILITY ONLY -- NOT PART OF THE FINAL APPLICATION.

This is NOT your real model training pipeline (that already happened in
your Google Colab notebook on the real ~2,392-record Kaggle dataset). It
exists purely so you can run and click through the Streamlit app right now,
before you copy your real exported files into model/.

It fits a small RandomForestClassifier on SYNTHETIC data that follows the
same 12-feature schema and value ranges as the real dataset, and writes
three files into model/ in exactly the format the app expects:

    model/student_risk_model.pkl
    model/student_risk_features.pkl
    model/student_risk_metadata.pkl

>>> REPLACE these three files with your real Colab-exported files before <<<
>>> using the app for any real prediction, demo, or submission.          <<<

Usage (run once, from the project root):
    python create_placeholder_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Exact feature order the app expects -- this list, not the model object,
# is the single source of truth for feature order throughout the app.
FEATURES = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]

RNG = np.random.default_rng(42)
N_SAMPLES = 2392  # matches the real dataset size, for a realistic placeholder


def make_synthetic_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate synthetic student records using the published value ranges /
    category codes for this dataset (Age 15-18, Ethnicity 0-3, etc.).
    Values are random, so any pattern this placeholder model learns is NOT
    meaningful -- it exists purely so the app has something real to load
    and call while you finish exporting your actual files.
    """
    df = pd.DataFrame(
        {
            "Age": RNG.integers(15, 19, n),                     # 15-18 inclusive
            "Gender": RNG.integers(0, 2, n),                    # 0 Male, 1 Female
            "Ethnicity": RNG.integers(0, 4, n),                 # 0-3, see model/README.md
            "ParentalEducation": RNG.integers(0, 5, n),         # 0-4, see model/README.md
            "StudyTimeWeekly": RNG.uniform(0, 20, n).round(2),  # hours/week
            "Absences": RNG.integers(0, 30, n),                 # 0-29
            "Tutoring": RNG.integers(0, 2, n),                  # 0 No, 1 Yes
            "ParentalSupport": RNG.integers(0, 5, n),           # 0-4, see model/README.md
            "Extracurricular": RNG.integers(0, 2, n),
            "Sports": RNG.integers(0, 2, n),
            "Music": RNG.integers(0, 2, n),
            "Volunteering": RNG.integers(0, 2, n),
        }
    )

    # Build a plausible (not real) risk signal so predict_proba() returns a
    # believable spread instead of near-constant values: more absences,
    # less study time, less tutoring/support push risk up. This is only so
    # the placeholder "looks alive" during a demo -- it carries no real
    # evidence about your actual students.
    risk_score = (
        -0.12 * df["StudyTimeWeekly"]
        + 0.15 * df["Absences"]
        - 0.5 * df["Tutoring"]
        - 0.25 * df["ParentalSupport"]
        + RNG.normal(0, 2.5, n)
    )
    threshold = np.quantile(risk_score, 0.65)
    df["AcademicRisk"] = (risk_score > threshold).astype(int)
    return df


def main() -> None:
    data = make_synthetic_dataset()
    X, y = data[FEATURES], data["AcademicRisk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metadata = {
        "is_placeholder": True,
        "model_name": "Random Forest Classifier (PLACEHOLDER - synthetic data)",
        "dataset_description": (
            "Synthetic stand-in for the Kaggle Students Performance dataset "
            "(~2,392 records, rabieelkharoua). Replace with your real Colab "
            "metadata.pkl before any real use."
        ),
        "n_records": N_SAMPLES,
        "features": FEATURES,
        "target": "AcademicRisk",
        "target_mapping": {0: "Not At Risk", 1: "At Risk"},
        "algorithms_evaluated": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
        ],
        "selected_algorithm": "Random Forest",
        # These are REAL metrics computed on the SYNTHETIC test split above
        # -- not invented -- but they describe THIS placeholder model, not
        # your actual Colab-trained model. You separately reported approx.
        # 88.5% accuracy / ~94% at-risk recall / ~0.92 F1 / ~0.93 ROC-AUC for
        # the real Random Forest model; those figures belong in your real
        # metadata.pkl, not here, so the app never displays a number it
        # can't trace back to an actual evaluation run.
        "metrics_note": "Computed on synthetic placeholder data -- NOT the real model's performance.",
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "at_risk_recall": round(float(recall_score(y_test, y_pred)), 4),
            "at_risk_f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        },
    }

    joblib.dump(model, MODEL_DIR / "student_risk_model.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "student_risk_features.pkl")
    joblib.dump(metadata, MODEL_DIR / "student_risk_metadata.pkl")

    # Marker file: lets using_placeholder_model() in model_utils.py detect
    # placeholder status without depending on metadata.pkl's content, so a
    # real metadata.pkl can be dropped in on its own (e.g. while
    # model.pkl/features.pkl are still pending) without silently hiding the
    # warning that predictions still come from this synthetic model.
    (MODEL_DIR / ".placeholder").write_text(
        "model/student_risk_model.pkl and model/student_risk_features.pkl "
        "in this folder were generated by create_placeholder_model.py, not "
        "exported from your real Colab notebook.\n\n"
        "Delete this file once BOTH have been replaced with your real "
        "exports. student_risk_metadata.pkl can be swapped independently -- "
        "doing so does not affect this marker.\n"
    )

    print(f"Placeholder files written to {MODEL_DIR}/")
    print("  - student_risk_model.pkl")
    print("  - student_risk_features.pkl")
    print("  - student_risk_metadata.pkl")
    print("  - .placeholder (marker file)")
    print("\nReplace the three .pkl files with your real Colab exports when")
    print("ready, then delete model/.placeholder.")


if __name__ == "__main__":
    main()
