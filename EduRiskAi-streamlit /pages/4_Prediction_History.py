"""
pages/4_Prediction_History.py

Every recorded prediction, filterable by student, risk level, and date.
"""

import streamlit as st

from utils import auth
from utils.database import get_all_predictions, init_db

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")
init_db()
auth.require_login()
auth.render_user_sidebar()

st.title("📜 Prediction History")
st.caption("Every recorded assessment, with filters for review and audit.")

df = get_all_predictions()
if df.empty:
    st.info("No predictions recorded yet.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    student_options = ["All"] + sorted(df["student_code"].dropna().unique().tolist())
    student_filter = st.selectbox("Student", student_options)
with col2:
    risk_filter = st.multiselect(
        "Risk Level", ["Low Risk", "Moderate Risk", "High Risk"],
        default=["Low Risk", "Moderate Risk", "High Risk"],
    )
with col3:
    date_range = st.date_input("Date range", value=())

filtered = df.copy()
if student_filter != "All":
    filtered = filtered[filtered["student_code"] == student_filter]
if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    dates = filtered["predicted_at"].str.slice(0, 10)
    filtered = filtered[(dates >= str(start)) & (dates <= str(end))]

st.caption(f"Showing {len(filtered)} of {len(df)} prediction(s).")
display_cols = [
    "predicted_at", "student_code", "student_name", "risk_level",
    "risk_probability", "prediction", "absences", "study_time_weekly", "model_version",
]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Export Filtered History (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "prediction_history.csv", "text/csv",
)
