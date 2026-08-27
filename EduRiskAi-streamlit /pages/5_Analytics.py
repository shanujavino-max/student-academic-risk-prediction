"""
pages/5_Analytics.py

Behavioral pattern analysis across recorded predictions, plus global
feature importance for explainability.
"""

import plotly.express as px
import streamlit as st

from utils import auth
from utils.database import get_all_predictions, init_db
from utils.model_utils import ModelLoadError, get_feature_importance, load_feature_list, load_model

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
init_db()
auth.require_login()
auth.render_user_sidebar()

st.title("📈 Analytics & Explainability")
st.caption("Behavioral patterns across recorded predictions, and what drives the model's decisions.")

RISK_COLORS = {"Low Risk": "#2ECC71", "Moderate Risk": "#F39C12", "High Risk": "#E74C3C"}

tab1, tab2 = st.tabs(["Cohort Analytics", "Model Explainability"])

with tab1:
    df = get_all_predictions()
    if df.empty:
        st.info("No predictions recorded yet -- run some to populate analytics.")
    else:
        pct_at_risk = (df["prediction"] == 1).mean() * 100
        st.metric("% of Assessed Students At Risk", f"{pct_at_risk:.1f}%")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average Absences by Risk Group")
            agg = df.groupby("risk_level")["absences"].mean().reset_index()
            fig = px.bar(agg, x="risk_level", y="absences", color="risk_level", color_discrete_map=RISK_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Average Weekly Study Time by Risk Group")
            agg = df.groupby("risk_level")["study_time_weekly"].mean().reset_index()
            fig = px.bar(agg, x="risk_level", y="study_time_weekly", color="risk_level", color_discrete_map=RISK_COLORS)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Tutoring Participation by Risk Group")
            agg = (df.groupby("risk_level")["tutoring"].mean() * 100).reset_index()
            fig = px.bar(agg, x="risk_level", y="tutoring", color="risk_level", color_discrete_map=RISK_COLORS,
                         labels={"tutoring": "% receiving tutoring"})
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.subheader("Parental Support Distribution")
            fig = px.histogram(df, x="parental_support", color="risk_level", color_discrete_map=RISK_COLORS, nbins=5)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Global Feature Importance")
    st.caption("From the trained Random Forest's `model.feature_importances_`.")
    try:
        model = load_model()
        feature_list = load_feature_list()
    except ModelLoadError as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    importance_df = get_feature_importance(model, feature_list)
    if importance_df.empty:
        st.warning("This model doesn't expose `feature_importances_`.")
    else:
        fig = px.bar(
            importance_df.sort_values("importance"), x="importance", y="feature", orientation="h",
            labels={"importance": "Relative Importance (Gini)", "feature": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "⚠️ Feature importance reflects how much weight the model's internal "
        "tree-splitting logic gave each variable -- it does **not** establish "
        "that any factor *causes* academic risk. Treat this as a description "
        "of the model's behaviour, not a causal claim about students."
    )
