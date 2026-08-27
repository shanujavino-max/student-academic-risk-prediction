"""
utils/model_utils.py

Centralised utilities for loading and using the pre-trained Random Forest
academic risk model. This module NEVER trains a model -- it only loads the
three artefacts exported from the Google Colab notebook:

    model/student_risk_model.pkl       -> fitted scikit-learn estimator
    model/student_risk_features.pkl    -> ordered list of feature names
    model/student_risk_metadata.pkl    -> dict describing the model/dataset

All loading functions are defensive: if a file is missing or corrupt, a
ModelLoadError is raised with a clear, human-readable message instead of a
raw traceback reaching the Streamlit UI. Calling code (app.py, pages/*.py)
is expected to catch ModelLoadError and render it with st.error(...).

Feature-importance / SHAP explainability helpers are added later, in the
dedicated Explainability development stage -- kept out of this file until
then so each stage adds something new rather than pre-building ahead of it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths -- resolved relative to THIS file (utils/model_utils.py), so the app
# finds model/ correctly no matter what working directory `streamlit run` is
# launched from. This matters most on Streamlit Community Cloud.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "student_risk_model.pkl"
FEATURES_PATH = MODEL_DIR / "student_risk_features.pkl"
METADATA_PATH = MODEL_DIR / "student_risk_metadata.pkl"

# AcademicRisk encoding: 0 = Not At Risk, 1 = At Risk. Kept as a named
# constant (instead of a magic number scattered through the codebase) so
# every place that needs "the at-risk class" is easy to find and audit.
AT_RISK_CLASS = 1

# Set once, the one time load_model() actually runs (it's cached after
# that) -- exposed via get_model_load_warning() rather than changing
# load_model()'s return type, so existing call sites don't need updating.
_model_load_warning: str | None = None


class ModelLoadError(Exception):
    """Raised when the model, feature list, or metadata cannot be loaded or
    used correctly. Always caught by UI code -- never allowed to surface as
    a raw traceback to a lecturer or admin using the app."""


@st.cache_resource(show_spinner="Loading academic risk model...")
def load_model():
    """
    Load the trained scikit-learn model from model/student_risk_model.pkl.
    Cached with st.cache_resource so the model is unpickled once per app
    process, not on every page rerun.

    Returns
    -------
    A fitted scikit-learn estimator (Random Forest in the final project).

    Raises
    ------
    ModelLoadError if the file is missing or cannot be unpickled.
    """
    if not MODEL_PATH.exists():
        raise ModelLoadError(
            f"Model file not found at '{MODEL_PATH}'. Export it from your "
            "Colab notebook with joblib.dump(model, 'student_risk_model.pkl') "
            "and place it in the model/ folder -- or run "
            "`python create_placeholder_model.py` to generate a placeholder "
            "for local testing."
        )

    global _model_load_warning
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(MODEL_PATH)
    except Exception as exc:  # broad on purpose: re-raised as a clear, catchable error
        raise ModelLoadError(f"Could not load model file: {exc}") from exc

    # scikit-learn raises one InconsistentVersionWarning PER estimator inside
    # an ensemble -- for a 200-tree Random Forest that's 200+ near-identical
    # warnings. Collapse to the unique message text so the UI shows one
    # clear notice instead of that noise (and instead of silence, since a
    # bare warning printed to the console is easy to miss -- especially
    # once this is deployed and nobody is watching a terminal).
    version_messages = {
        str(w.message) for w in caught if "version" in str(w.message).lower()
    }
    _model_load_warning = (
        "This model was pickled with a different scikit-learn version than "
        "the one installed here, which scikit-learn warns can change "
        f"prediction behaviour. Detail: {next(iter(version_messages))}"
        if version_messages
        else None
    )

    return model


def get_model_load_warning() -> str | None:
    """
    Returns a human-readable warning if load_model() detected that
    model.pkl was pickled with a different scikit-learn version than the
    one currently installed (a real risk per scikit-learn's own docs, not
    just a cosmetic notice) -- or None if versions matched, or load_model()
    hasn't been called yet this session.
    """
    return _model_load_warning


@st.cache_data(show_spinner=False)
def load_feature_list() -> list[str]:
    """
    Load the exact, ordered list of feature names the model was trained on.

    Raises
    ------
    ModelLoadError if the file is missing, corrupt, or not a non-empty list.
    """
    if not FEATURES_PATH.exists():
        raise ModelLoadError(
            f"Feature list not found at '{FEATURES_PATH}'. Export it from "
            "Colab with joblib.dump(list(X_train.columns), "
            "'student_risk_features.pkl')."
        )
    try:
        features = joblib.load(FEATURES_PATH)
    except Exception as exc:
        raise ModelLoadError(f"Could not load feature list: {exc}") from exc

    if not isinstance(features, (list, tuple)) or not features:
        raise ModelLoadError(
            "student_risk_features.pkl did not contain a non-empty list of "
            "feature names."
        )
    return list(features)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    """
    Load optional model/dataset metadata (accuracy, algorithms compared,
    dataset description, etc.).

    If the file is missing, an empty dict is returned. Calling UI code must
    display "Not available" for any missing field -- this app never
    fabricates a metric that was not actually recorded in this file.
    """
    if not METADATA_PATH.exists():
        return {}
    try:
        metadata = joblib.load(METADATA_PATH)
    except Exception:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def using_placeholder_model() -> bool:
    """
    True if the currently-loaded model.pkl / features.pkl in model/ were
    generated by create_placeholder_model.py rather than being your real
    Colab exports.

    Deliberately checked via a separate marker file (model/.placeholder)
    instead of a flag inside metadata.pkl. Metadata can reasonably be
    swapped in on its own before the model/features files are ready (as
    happens when you upload one real export at a time) -- if "placeholder"
    status lived inside metadata.pkl, that swap would silently turn off
    this warning even though predictions are still coming from the fake
    model.
    """
    return (MODEL_DIR / ".placeholder").exists()


def build_input_dataframe(
    feature_values: dict[str, float], feature_list: list[str]
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for prediction with columns in EXACTLY the
    order stored in student_risk_features.pkl. This matters because the
    model was fit on a DataFrame with that column order/names -- mismatched
    order or names is a common, *silent* source of wrong predictions.

    Parameters
    ----------
    feature_values : dict mapping feature name -> already-encoded numeric
                      value. Dropdown-label-to-number conversion happens
                      BEFORE this call (that lives in the prediction page /
                      validation.py, added in later stages).
    feature_list   : the ordered list returned by load_feature_list().

    Raises
    ------
    ModelLoadError if a required feature is missing from feature_values.
    """
    missing = [f for f in feature_list if f not in feature_values]
    if missing:
        raise ModelLoadError(f"Missing required feature(s) for prediction: {missing}")

    ordered_row = {feature: feature_values[feature] for feature in feature_list}
    return pd.DataFrame([ordered_row], columns=feature_list)


def predict_risk(model, input_df: pd.DataFrame) -> tuple[int, float]:
    """
    Run model.predict() and model.predict_proba() on a single-row DataFrame
    that already matches the model's expected feature order.

    Returns
    -------
    (prediction, probability_at_risk) for a single-row input, where
    prediction is 0/1 and probability_at_risk is the model's confidence
    that the student belongs to the "At Risk" (1) class. The at-risk
    probability column is located defensively via model.classes_ rather
    than assumed to be column index 1.
    """
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    class_list = list(model.classes_)
    at_risk_index = class_list.index(AT_RISK_CLASS) if AT_RISK_CLASS in class_list else 1
    probability_at_risk = probabilities[:, at_risk_index]

    if len(input_df) == 1:
        return int(prediction[0]), float(probability_at_risk[0])
    return prediction, probability_at_risk


def get_feature_importance(model, feature_list: list[str]) -> pd.DataFrame:
    """
    Global feature importance via model.feature_importances_ (Random Forest /
    Decision Tree only). Returns an empty DataFrame if the model doesn't
    expose this attribute, rather than raising -- Analytics is expected to
    handle that gracefully, not crash.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame({"feature": feature_list, "importance": model.feature_importances_})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
