"""
utils/validation.py

Single source of truth for:
  1. The mapping between friendly dropdown labels and the numeric codes the
     trained model expects.
  2. Range/type validation for every input field.

If your Colab notebook encoded ParentalEducation or ParentalSupport
differently, this is the ONLY file you need to edit -- every page that
builds a prediction imports these dicts rather than hard-coding numbers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Encoding maps: {friendly label: numeric code}
# Verify these against your own Colab preprocessing cell -- see
# model/README.md for what was and wasn't independently confirmed.
# ---------------------------------------------------------------------------
GENDER_MAP = {"Male": 0, "Female": 1}

ETHNICITY_MAP = {
    "Caucasian": 0,
    "African American": 1,
    "Asian": 2,
    "Other": 3,
}

PARENTAL_EDUCATION_MAP = {
    "None": 0,
    "High School": 1,
    "Some College": 2,
    "Bachelor's": 3,
    "Higher": 4,
}

PARENTAL_SUPPORT_MAP = {
    "None": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4,
}

YES_NO_MAP = {"No": 0, "Yes": 1}  # Tutoring, Extracurricular, Sports, Music, Volunteering

# Numeric range limits
AGE_MIN, AGE_MAX = 15, 18
STUDY_TIME_MIN, STUDY_TIME_MAX = 0.0, 20.0
ABSENCES_MIN, ABSENCES_MAX = 0, 30


class ValidationError(Exception):
    """Raised when a form input fails validation. Always caught by the
    calling page and shown with st.error(), never as a raw traceback."""


def validate_age(age: int) -> None:
    if not (AGE_MIN <= age <= AGE_MAX):
        raise ValidationError(f"Age must be between {AGE_MIN} and {AGE_MAX}.")


def validate_study_time(hours: float) -> None:
    if hours < STUDY_TIME_MIN:
        raise ValidationError("Weekly study time cannot be negative.")
    if hours > STUDY_TIME_MAX:
        raise ValidationError(f"Weekly study time cannot exceed {STUDY_TIME_MAX} hours.")


def validate_absences(count: int) -> None:
    if count < ABSENCES_MIN:
        raise ValidationError("Absences cannot be negative.")
    if count > ABSENCES_MAX:
        raise ValidationError(f"Absences cannot exceed {ABSENCES_MAX}.")


def validate_and_encode_form(raw: dict) -> dict:
    """
    Validate every field in `raw` (friendly labels + raw numbers from the
    Streamlit form) and return a dict of {feature_name: numeric_value}
    ready for build_input_dataframe(). Raises ValidationError on the first
    problem found, with a message safe to show a lecturer directly.
    """
    validate_age(raw["Age"])
    validate_study_time(raw["StudyTimeWeekly"])
    validate_absences(raw["Absences"])

    for label, mapping in [
        (raw["Gender"], GENDER_MAP),
        (raw["Ethnicity"], ETHNICITY_MAP),
        (raw["ParentalEducation"], PARENTAL_EDUCATION_MAP),
        (raw["ParentalSupport"], PARENTAL_SUPPORT_MAP),
    ]:
        if label not in mapping:
            raise ValidationError(f"'{label}' is not a supported value.")

    return {
        "Age": int(raw["Age"]),
        "Gender": GENDER_MAP[raw["Gender"]],
        "Ethnicity": ETHNICITY_MAP[raw["Ethnicity"]],
        "ParentalEducation": PARENTAL_EDUCATION_MAP[raw["ParentalEducation"]],
        "StudyTimeWeekly": float(raw["StudyTimeWeekly"]),
        "Absences": int(raw["Absences"]),
        "Tutoring": YES_NO_MAP[raw["Tutoring"]],
        "ParentalSupport": PARENTAL_SUPPORT_MAP[raw["ParentalSupport"]],
        "Extracurricular": YES_NO_MAP[raw["Extracurricular"]],
        "Sports": YES_NO_MAP[raw["Sports"]],
        "Music": YES_NO_MAP[raw["Music"]],
        "Volunteering": YES_NO_MAP[raw["Volunteering"]],
    }


def validate_batch_dataframe(
    df, feature_list: list[str]
) -> tuple["pd.DataFrame", list[str], str | None]:
    """
    Validates an uploaded bulk-prediction CSV.

    Returns (clean_df, warnings, fatal_error):
      - fatal_error: set (str) if the file can't be used AT ALL (a required
        feature column is missing entirely). clean_df is empty in that case.
      - warnings: non-fatal issues -- rows dropped for being non-numeric or
        out of range. The caller should always show these, even when
        fatal_error is None, so nothing is silently discarded.
      - clean_df: the validated, numeric rows actually safe to predict on.
    """
    import pandas as pd

    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        return df.iloc[0:0], [], f"Missing required column(s): {missing}"

    clean_df = df.copy()
    warnings: list[str] = []

    for col in feature_list:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    bad_type = clean_df[feature_list].isnull().any(axis=1)
    if bad_type.any():
        warnings.append(f"{int(bad_type.sum())} row(s) had non-numeric/missing values and were excluded.")
        clean_df = clean_df[~bad_type].reset_index(drop=True)

    if clean_df.empty:
        return clean_df, warnings, "No valid rows remain after checking required columns."

    out_of_range = (
        (clean_df["Age"] < AGE_MIN) | (clean_df["Age"] > AGE_MAX)
        | (clean_df["Absences"] < ABSENCES_MIN) | (clean_df["Absences"] > ABSENCES_MAX)
        | (clean_df["StudyTimeWeekly"] < STUDY_TIME_MIN) | (clean_df["StudyTimeWeekly"] > STUDY_TIME_MAX)
    )
    if out_of_range.any():
        warnings.append(
            f"{int(out_of_range.sum())} row(s) fell outside expected ranges "
            f"(Age {AGE_MIN}-{AGE_MAX}, Absences {ABSENCES_MIN}-{ABSENCES_MAX}, "
            f"StudyTimeWeekly {STUDY_TIME_MIN}-{STUDY_TIME_MAX}) and were excluded."
        )
        clean_df = clean_df[~out_of_range].reset_index(drop=True)

    if clean_df.empty:
        return clean_df, warnings, "No valid rows remain after range checks."

    return clean_df, warnings, None
