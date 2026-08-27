"""
pages/2_Student_Prediction.py

Individual academic risk prediction: student registry (insert-or-update on
submit), the 12-feature form, real model inference, a risk gauge, rule-based
recommendations, and persistence to the predictions table.
"""

import plotly.graph_objects as go
import streamlit as st

from utils import auth
from utils.database import init_db, insert_prediction, log_activity, upsert_student
from utils.model_utils import (
    ModelLoadError,
    build_input_dataframe,
    get_model_load_warning,
    load_feature_list,
    load_metadata,
    load_model,
    predict_risk,
    using_placeholder_model,
)
from utils.recommendations import generate_recommendations
from utils.validation import (
    ETHNICITY_MAP,
    GENDER_MAP,
    PARENTAL_EDUCATION_MAP,
    PARENTAL_SUPPORT_MAP,
    YES_NO_MAP,
    ValidationError,
    validate_and_encode_form,
)

st.set_page_config(page_title="Student Prediction", page_icon="🎯", layout="wide")

init_db()
user = auth.require_login()
auth.render_user_sidebar()

st.title("🎯 Individual Student Risk Assessment")
st.caption("Enter demographic, study, participation, and support indicators to estimate academic risk.")

try:
    model = load_model()
    feature_list = load_feature_list()
    metadata = load_metadata()
except ModelLoadError as exc:
    st.error(f"Cannot make predictions -- the model is not available.\n\n**Details:** {exc}")
    st.stop()

if using_placeholder_model():
    st.warning(
        "⚠️ model.pkl / features.pkl are still placeholder files -- results "
        "below are for testing the pipeline only, not real predictions."
    )
version_warning = get_model_load_warning()
if version_warning:
    st.warning(f"⚠️ {version_warning}")

# ---------------------------------------------------------------------------
# Student registry -- insert-or-update on submit, matched by student_code.
# Kept inline with prediction (rather than a separate mandatory step) so a
# lecturer can run a one-off assessment without pre-registering a student.
# ---------------------------------------------------------------------------
with st.expander("Register or Select Existing Student Record", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        student_code = st.text_input("Student ID", placeholder="e.g. STU-2026-001")
        student_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
    with col_b:
        course = st.text_input("Course / Degree", placeholder="e.g. B.Sc. Computer Science")
        academic_year = st.selectbox("Academic Year", [1, 2, 3, 4], index=0)

st.subheader("Academic Risk Indicators")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age", min_value=15, max_value=18, value=16, step=1)
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        ethnicity = st.selectbox("Ethnicity", list(ETHNICITY_MAP.keys()))
        parental_education = st.selectbox("Parental Education", list(PARENTAL_EDUCATION_MAP.keys()))

    with col2:
        st.markdown("**Academic engagement**")
        study_time = st.number_input("Weekly Study Time (hours)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
        absences = st.number_input("Absences (this term)", min_value=0, max_value=30, value=5, step=1)
        tutoring = st.selectbox("Tutoring Classes", list(YES_NO_MAP.keys()))
        parental_support = st.selectbox("Parental Support Level", list(PARENTAL_SUPPORT_MAP.keys()))

    with col3:
        st.markdown("**Activities**")
        extracurricular = st.selectbox("Extracurricular Activities", list(YES_NO_MAP.keys()))
        sports = st.selectbox("Sports Participation", list(YES_NO_MAP.keys()))
        music = st.selectbox("Music Activities", list(YES_NO_MAP.keys()))
        volunteering = st.selectbox("Volunteering", list(YES_NO_MAP.keys()))

    submitted = st.form_submit_button("Run Risk Prediction", use_container_width=True)

RISK_COLORS = {"Low Risk": "#2ECC71", "Moderate Risk": "#F39C12", "High Risk": "#E74C3C"}


def classify_risk_level(probability: float) -> str:
    """
    Application-level presentation thresholds over the model's own
    predict_proba() output -- NOT separately validated against outcome
    data. The model itself only performs binary At Risk / Not At Risk
    classification; this 3-way split exists purely for triage.
    """
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Moderate Risk"
    return "High Risk"


if submitted:
    if not student_code.strip():
        st.error("Student ID is required before running a prediction.")
        st.stop()

    raw = {
        "Age": age, "Gender": gender, "Ethnicity": ethnicity,
        "ParentalEducation": parental_education, "StudyTimeWeekly": study_time,
        "Absences": absences, "Tutoring": tutoring, "ParentalSupport": parental_support,
        "Extracurricular": extracurricular, "Sports": sports, "Music": music,
        "Volunteering": volunteering,
    }

    try:
        encoded = validate_and_encode_form(raw)
        input_df = build_input_dataframe(encoded, feature_list)
        prediction, probability = predict_risk(model, input_df)
    except ValidationError as exc:
        st.error(f"Please check your input: {exc}")
        st.stop()
    except ModelLoadError as exc:
        st.error(f"Could not generate a prediction: {exc}")
        st.stop()

    risk_level = classify_risk_level(probability)
    # metadata["class_0"] / ["class_1"] is the verified real key structure
    # from this project's actual metadata.pkl -- falls back to a sensible
    # default only if that key happens to be missing.
    result_label = metadata.get(f"class_{prediction}") or ("At Risk" if prediction == 1 else "Not At Risk")

    try:
        student_pk = upsert_student(
            student_code=student_code.strip(),
            full_name=student_name.strip() or "Unassigned",
            course=course.strip() or None,
            academic_year=int(academic_year),
            created_by=user["id"],
        )
        insert_prediction(
            student_id=student_pk,
            encoded_input=encoded,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            model_version=metadata.get("model_name", "unknown"),
            predicted_by=user["id"],
        )
        log_activity(user["id"], "prediction_run", f"Assessed student {student_code.strip()}")
    except Exception as exc:  # noqa: BLE001
        # A DB failure should never hide a prediction that already
        # succeeded -- downgrade to a warning and keep showing the result.
        st.warning(f"Prediction generated, but couldn't be saved to the database: {exc}")

    st.divider()
    st.subheader("Academic Risk Assessment Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Probability", f"{probability:.1%}")
    m2.metric("Risk Level", risk_level)
    m3.metric("Classification", result_label)

    gauge_col, detail_col = st.columns([1.2, 1])

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"Risk Status: {risk_level}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[risk_level]},
                "steps": [
                    {"range": [0, 35], "color": "rgba(46, 204, 113, 0.20)"},
                    {"range": [35, 65], "color": "rgba(243, 156, 18, 0.20)"},
                    {"range": [65, 100], "color": "rgba(231, 76, 60, 0.20)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": probability * 100},
            },
        ))
        fig.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with detail_col:
        st.markdown("#### Assessment Outcome")
        if prediction == 1:
            st.error(f"Classified as **{result_label}**.")
        else:
            st.success(f"Classified as **{result_label}**.")
        st.caption(
            "Low / Moderate / High are application-level interpretation bands "
            "over the model's probability output, not separately validated "
            "categories in their own right."
        )

    st.subheader("Recommended Interventions")
    for i, rec in enumerate(generate_recommendations(encoded, probability, risk_level), start=1):
        st.markdown(f"**{i}. {rec['category']}** — {rec['action']}")
        st.caption(rec["reason"])

    with st.expander("Developer verification"):
        st.write("Expected feature order:")
        st.code(str(feature_list))
        st.write("Encoded input sent to the model:")
        st.json(encoded)

    st.divider()
    st.caption(
        "Decision-support output only. Review alongside academic evidence "
        "and professional judgement before any intervention -- this system "
        "must not be the sole basis for academic penalties or decisions."
    )
