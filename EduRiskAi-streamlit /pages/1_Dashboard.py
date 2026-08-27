"""
pages/1_Dashboard.py

Operational snapshot: headline KPIs, recent activity, and the required
charts (risk distribution, probability distribution, study time vs risk,
absences vs risk, prediction trends over time).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import auth
from utils.database import get_all_predictions, get_all_students, init_db

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
init_db()
auth.require_login()
auth.render_user_sidebar()

st.title("📊 Executive Risk Dashboard")
st.caption("Aggregated risk metrics across every recorded assessment.")

students = get_all_students()
preds = get_all_predictions()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
total_students = len(students)
total_predictions = len(preds)
at_risk = int((preds["prediction"] == 1).sum()) if not preds.empty else 0
not_at_risk = int((preds["prediction"] == 0).sum()) if not preds.empty else 0
high_risk = int((preds["risk_level"] == "High Risk").sum()) if not preds.empty else 0
avg_prob = float(preds["risk_probability"].mean() * 100) if not preds.empty else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Students", total_students)
c2.metric("Total Predictions", total_predictions)
c3.metric("At-Risk", at_risk)
c4.metric("Not At-Risk", not_at_risk)
c5.metric("High Risk", high_risk)
c6.metric("Avg Risk Probability", f"{avg_prob:.1f}%")

st.divider()

if preds.empty:
    st.info("No predictions recorded yet. Run some from Student Prediction or Bulk Prediction to populate this dashboard.")
    st.stop()

RISK_COLORS = {"Low Risk": "#2ECC71", "Moderate Risk": "#F39C12", "High Risk": "#E74C3C"}

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
row1_a, row1_b = st.columns(2)
with row1_a:
    st.subheader("Risk Distribution")
    fig = px.pie(preds, names="risk_level", color="risk_level", color_discrete_map=RISK_COLORS, hole=0.45)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with row1_b:
    st.subheader("Risk Probability Distribution")
    fig = px.histogram(preds, x="risk_probability", nbins=20, color="risk_level", color_discrete_map=RISK_COLORS)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Probability of Risk")
    st.plotly_chart(fig, use_container_width=True)

row2_a, row2_b = st.columns(2)
with row2_a:
    st.subheader("Study Time vs. Academic Risk")
    fig = px.scatter(
        preds, x="study_time_weekly", y="risk_probability", color="risk_level",
        color_discrete_map=RISK_COLORS,
        labels={"study_time_weekly": "Weekly Study Time (hrs)", "risk_probability": "Risk Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_b:
    st.subheader("Absences vs. Academic Risk")
    fig = px.scatter(
        preds, x="absences", y="risk_probability", color="risk_level",
        color_discrete_map=RISK_COLORS,
        labels={"absences": "Absences", "risk_probability": "Risk Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Trends Over Time")
trend = preds.copy()
trend["date"] = pd.to_datetime(trend["predicted_at"]).dt.date
daily = trend.groupby(["date", "risk_level"]).size().reset_index(name="count")
if len(daily) >= 1:
    fig = px.line(daily, x="date", y="count", color="risk_level", color_discrete_map=RISK_COLORS, markers=True)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Predictions")
recent_cols = ["student_code", "student_name", "risk_level", "risk_probability", "predicted_at"]
st.dataframe(preds[recent_cols].head(10), use_container_width=True, hide_index=True)
