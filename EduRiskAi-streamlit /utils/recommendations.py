"""
utils/recommendations.py

Rule-based intervention suggestions -- deliberately plain if/then logic on
the actual encoded inputs, rather than a second model. A lecturer reading
these needs to see WHY each one was suggested; a black-box recommender
wouldn't give them that.
"""

from __future__ import annotations

# Thresholds here are practical judgement calls, not statistically derived
# from the training data -- documented as such rather than presented as
# precise cutoffs.
HIGH_ABSENCES = 15
LOW_STUDY_HOURS = 5.0
LOW_PARENTAL_SUPPORT = 1  # encoded value: 0=None, 1=Low


def generate_recommendations(
    encoded_input: dict, probability: float, risk_level: str
) -> list[dict[str, str]]:
    """
    Returns an ordered list of {"category", "action", "reason"} dicts.
    Each recommendation is tied to a specific input value that's actually
    driving it -- never a generic list unrelated to this student's data.
    Always returns at least one item.
    """
    recs: list[dict[str, str]] = []

    absences = encoded_input.get("Absences", 0)
    if absences >= HIGH_ABSENCES:
        recs.append({
            "category": "Attendance Improvement",
            "action": "Refer to attendance support and agree a written attendance plan.",
            "reason": f"{absences} recorded absences is high for this cohort.",
        })

    study_hours = encoded_input.get("StudyTimeWeekly", 0)
    if study_hours < LOW_STUDY_HOURS:
        recs.append({
            "category": "Increased Study Planning",
            "action": "Provide a structured weekly study plan; review progress in two weeks.",
            "reason": f"Only {study_hours:.1f} reported study hours/week.",
        })

    if encoded_input.get("Tutoring", 0) == 0 and risk_level != "Low Risk":
        recs.append({
            "category": "Additional Tutoring",
            "action": "Offer enrolment in the peer or staff tutoring programme for this course.",
            "reason": "Not currently receiving tutoring.",
        })

    if encoded_input.get("ParentalSupport", 0) <= LOW_PARENTAL_SUPPORT:
        recs.append({
            "category": "Academic Counselling",
            "action": "Involve student counselling services given limited reported home support.",
            "reason": "Parental support recorded as None/Low.",
        })

    if risk_level == "High Risk":
        recs.append({
            "category": "Lecturer Review",
            "action": "Flag for direct lecturer/advisor review before the next assessment deadline.",
            "reason": f"Model-estimated risk probability of {probability:.0%} is in the High Risk band.",
        })

    # Always end with a monitoring item, framed to match the tier -- so the
    # list is never empty even when nothing else triggered.
    if risk_level == "Low Risk" and not recs:
        recs.append({
            "category": "Progress Monitoring",
            "action": "No immediate action needed; continue routine progress monitoring.",
            "reason": "Current indicators do not suggest elevated risk.",
        })
    else:
        recs.append({
            "category": "Progress Monitoring",
            "action": "Re-run this assessment after the next attendance/assessment cycle.",
            "reason": "Indicators should be re-checked as the term progresses.",
        })

    return recs
