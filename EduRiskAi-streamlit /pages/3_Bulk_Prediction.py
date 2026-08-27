"""
pages/3_Bulk_Prediction.py

CSV upload -> validate required feature columns -> run model.predict()/
predict_proba() on every valid row -> interactive results table -> CSV
download -> persist to the predictions table.
"""

import pandas as pd
import streamlit as st

from utils import auth
from utils.database import batch_insert_predictions, init_db, log_activity, upsert_student
from utils.model_utils import (
    ModelLoadError,
    get_model_load_warning,
    load_feature_list,
    load_metadata,
    load_model,
    using_placeholder_model,
)
from utils.validation import validate_batch_dataframe

st.set_page_config(page_title="Bulk Prediction", page_icon="📂", layout="wide")
init_db()
user = auth.require_login()
auth.render_user_sidebar()

st.title("📂 Bulk Student Risk Assessment")
st.caption("Upload a CSV of student records to run predictions on many students at once.")

try:
    model = load_model()
    feature_list = load_feature_list()
    metadata = load_metadata()
except ModelLoadError as exc:
    st.error(f"Cannot run bulk predictions -- the model is not available.\n\n**Details:** {exc}")
    st.stop()

if using_placeholder_model():
    st.warning("⚠️ model.pkl / features.pkl are still placeholder files -- results below are for testing the pipeline only.")
version_warning = get_model_load_warning()
if version_warning:
    st.warning(f"⚠️ {version_warning}")

col_template, col_real = st.columns(2)
with col_template:
    template_df = pd.DataFrame([{**{f: 0 for f in feature_list}, "StudentID": "STU-001"}])
    template_df = template_df[["StudentID"] + feature_list]
    st.download_button(
        "Download CSV Template", template_df.to_csv(index=False),
        "student_risk_batch_template.csv", "text/csv", use_container_width=True,
    )
with col_real:
    st.caption(
        "Have `data/real_student_features.csv` from earlier in this project? "
        "That's real feature data (2,392 rows) and works directly here -- "
        "no need to fill in the template by hand to test this page."
    )

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file as CSV: {exc}")
        st.stop()

    st.write("Uploaded sample:")
    st.dataframe(raw_df.head(3), use_container_width=True)

    clean_df, warnings, fatal_error = validate_batch_dataframe(raw_df, feature_list)
    for w in warnings:
        st.warning(w)
    if fatal_error:
        st.error(fatal_error)
        st.stop()

    if st.button("Execute Batch Prediction", use_container_width=True):
        X_eval = clean_df[feature_list]
        predictions = model.predict(X_eval)
        probabilities = model.predict_proba(X_eval)
        class_list = list(model.classes_)
        at_risk_idx = class_list.index(1) if 1 in class_list else 1
        probabilities = probabilities[:, at_risk_idx]

        def classify(p: float) -> str:
            if p < 0.35:
                return "Low Risk"
            if p < 0.65:
                return "Moderate Risk"
            return "High Risk"

        results_df = clean_df.copy()
        results_df["Prediction"] = predictions
        results_df["Risk_Probability"] = probabilities
        results_df["Risk_Level"] = [classify(p) for p in probabilities]

        st.success(f"Batch prediction completed for {len(results_df)} records.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Processed", len(results_df))
        m2.metric("Identified At-Risk", int((results_df["Prediction"] == 1).sum()))
        m3.metric("High Risk Tier", int((results_df["Risk_Level"] == "High Risk").sum()))

        st.dataframe(results_df, use_container_width=True)

        # --- Persist to the database ---
        try:
            has_ids = "StudentID" in results_df.columns
            rows_to_insert = []
            for _, row in results_df.iterrows():
                student_pk = None
                if has_ids and pd.notna(row["StudentID"]):
                    student_pk = upsert_student(
                        student_code=str(row["StudentID"]), full_name="Bulk import",
                        course=None, academic_year=None, created_by=user["id"],
                    )
                rows_to_insert.append({
                    "student_id": student_pk,
                    **{f: row[f] for f in feature_list},
                    "prediction": int(row["Prediction"]),
                    "probability": float(row["Risk_Probability"]),
                    "risk_level": row["Risk_Level"],
                    "model_version": metadata.get("model_name", "unknown"),
                    "predicted_by": user["id"],
                })
            batch_insert_predictions(rows_to_insert)
            log_activity(user["id"], "bulk_prediction", f"Processed {len(results_df)} records.")
            st.caption(f"Saved {len(rows_to_insert)} prediction(s) to the database.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Predictions generated, but couldn't all be saved to the database: {exc}")

        st.download_button(
            "Download Full Prediction Results",
            results_df.to_csv(index=False).encode("utf-8"),
            "batch_prediction_results.csv", "text/csv", use_container_width=True,
        )
