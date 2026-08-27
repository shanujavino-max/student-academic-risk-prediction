"""
pages/6_About.py

Model information and the ethics/disclaimer section. The "Model Metadata"
card is read strictly from metadata.pkl and shows "Not available" for
anything missing -- it never fabricates a metric. The "Development Notes"
section below it is clearly separated static narrative (what the developer
has reported so far), never mixed into the metadata card itself.
"""

import streamlit as st

from utils import auth
from utils.database import init_db
from utils.model_utils import ModelLoadError, load_feature_list, load_metadata, load_model, using_placeholder_model

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
init_db()
auth.require_login()
auth.render_user_sidebar()

st.title("ℹ️ About & Model Information")

try:
    model = load_model()
    feature_list = load_feature_list()
    metadata = load_metadata()
except ModelLoadError as exc:
    st.error(f"Model unavailable: {exc}")
    metadata, feature_list, model = {}, [], None

if using_placeholder_model():
    st.warning("⚠️ The model currently loaded is a placeholder -- figures below describe that stand-in, not a finalised model.")

st.subheader("Model Metadata")
st.caption("Read directly from `model/student_risk_metadata.pkl`. Fields not present in that file show as 'Not available' rather than a guessed value.")

def field(label: str, key: str, default: str = "Not available") -> None:
    st.markdown(f"**{label}:** {metadata.get(key, default)}")

col1, col2 = st.columns(2)
with col1:
    field("Model name", "model_name")
    field("Target variable", "target")
    field("Class 0", "class_0")
    field("Class 1", "class_1")
with col2:
    st.markdown(f"**Input features ({len(feature_list)}):**")
    st.code(", ".join(feature_list) if feature_list else "Not available")
    field("GPA used as input", "gpa_used")

st.markdown("**Performance metrics:**")
metrics = metadata.get("metrics")
if metrics:
    m_cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        m_cols[i].metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))
else:
    st.info(
        "Not available in metadata.pkl yet. See `model/README.md` for the exact "
        "code to add `metrics` (accuracy, recall, F1, ROC-AUC) to your Colab export."
    )

st.divider()

st.subheader("Development Notes")
st.caption("Reported by the developer during this project's build -- not read from metadata.pkl, and not displayed as verified metrics above.")
st.markdown(
    """
- **Dataset**: Kaggle Students Performance dataset (rabieelkharoua) -- confirmed as 2,392 records from the uploaded raw data file.
- **Algorithms evaluated**: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine.
- **Selected algorithm**: Random Forest, for the strongest at-risk recall among the four.
    """
)

st.divider()

st.subheader("Ethical Use & Limitations")
st.markdown(
    """
1. **Decision support only.** This system produces an early-warning estimate for academic staff to review -- it is not authorised to make, or automatically trigger, any academic or disciplinary decision.
2. **Not a substitute for judgement.** A "High Risk" or "Low Risk" label is one input among many. Predictions must be reviewed by a qualified member of academic staff before any action is taken.
3. **False positives and negatives are possible.** No classifier is perfect. A student flagged as at-risk may not be, and a student not flagged may still need support.
4. **Presentation thresholds, not validated categories.** The Low/Moderate/High risk bands are an application-level split of the model's own probability output, chosen for readability -- they have not been separately validated against real outcomes.
5. **Feature importance is not causation.** Seeing that a factor is "important" to the model does not mean it *causes* risk -- see Analytics for the full caveat.
6. **Dataset and encoding limitations.** This model was trained on a specific dataset with a fixed set of categories; it will not generalise perfectly to students or contexts outside that data. One categorical encoding (Gender) in this build is still unverified against the original notebook -- see `model/README.md`.
7. **Confidentiality.** Student data entered here should be handled according to your institution's data protection policy. Only collect what's necessary for the assessment.
    """
)
