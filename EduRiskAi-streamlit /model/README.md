# model/

Place the three files exported from your Google Colab notebook here.

| File | Contents | Export with |
|---|---|---|
| `student_risk_model.pkl` | Fitted scikit-learn estimator (Random Forest) | `joblib.dump(model, "student_risk_model.pkl")` |
| `student_risk_features.pkl` | Ordered list of the 12 input feature names, exactly as columns were passed to `model.fit()` | `joblib.dump(list(X_train.columns), "student_risk_features.pkl")` |
| `student_risk_metadata.pkl` | Optional dict describing the dataset/algorithms/metrics (schema below) | `joblib.dump(metadata_dict, "student_risk_metadata.pkl")` |

## Expected feature order

```
Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences,
Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering
```

`student_risk_features.pkl` is the single source of truth for this order —
the app always reindexes any input into that exact order before calling
`model.predict()`, so this list must match what your model was actually fit
on, in your notebook, not just this README.

## Confirmed: your real `metadata.pkl`

Your exported `student_risk_metadata.pkl` has been checked against this
pipeline (loads cleanly, feature order matches `features.pkl` exactly).
Its actual structure:

```python
{
    "model_name": "Random Forest Classifier",
    "target": "AcademicRisk",
    "class_0": "Not At Risk",
    "class_1": "At Risk",
    "features": [...],   # confirmed identical to the 12 names above
    "gpa_used": False,
}
```

The app reads this generically (`st.json(metadata)`, plus a couple of
`metadata.get(...)` lookups with fallbacks) so it displays correctly as-is
— no changes needed on your end.

**One gap worth knowing about:** this file has no `metrics` (accuracy,
recall, F1, ROC-AUC), no `dataset_description`, no `n_records`, and no
`algorithms_evaluated` list. The Model Information page (Stage 15) will
correctly show "Not available" for each of those rather than inventing
numbers — but if you'd rather it display your reported ~88.5% accuracy /
~94% at-risk recall / ~0.92 F1 / ~0.93 ROC-AUC, add them in Colab before
re-exporting:

```python
metadata["dataset_description"] = "Kaggle Students Performance dataset (rabieelkharoua)"
metadata["n_records"] = 2392
metadata["algorithms_evaluated"] = [
    "Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"
]
metadata["metrics"] = {
    "accuracy": 0.885,        # replace with your exact evaluation numbers
    "at_risk_recall": 0.94,
    "at_risk_f1": 0.92,
    "roc_auc": 0.93,
}
joblib.dump(metadata, "student_risk_metadata.pkl")
```
This is optional — the app works fully without it.

## Testing before your real `model.pkl` / `features.pkl` are ready

Run `python create_placeholder_model.py` from the project root to generate
placeholder `model.pkl` / `features.pkl` / `metadata.pkl`, plus a
`model/.placeholder` marker file. The app checks that marker file (not
anything inside `metadata.pkl`) to decide whether to show the "you're on a
placeholder" warning — so you can drop in your real `metadata.pkl` on its
own, as you already have, without it silently hiding the warning that
predictions still come from the synthetic model. Once your real
`student_risk_model.pkl` and `student_risk_features.pkl` are both in place,
delete `model/.placeholder` and the warning goes away.

## About the categorical encodings used here

Confirmed directly by inspecting your real 2,392-row dataset
(`data/real_student_features.csv`) — not just search this time:

- **Age**: 15–18 (exactly 4 values, confirmed) · **StudyTimeWeekly**: 0–~20 hrs · **Absences**: 0–29
- **Ethnicity**: exactly 4 categories (0–3). Category `0` accounts for precisely
  50.46% of rows, an exact match to a public analysis of this dataset reporting
  50.46% Caucasian — `0 = Caucasian` is about as confirmed as it gets without
  your notebook itself. The other three (African American / Asian / Other,
  assumed in that order) are *not* independently confirmed.
- **ParentalEducation**: exactly 5 categories (0–4) — confirmed by the real
  data. Order assumed ascending (None → Higher) as the standard convention
  for an ordinal field; not independently confirmed.
- **ParentalSupport**: exactly 5 categories (0–4) — confirmed by the real
  data. Order assumed ascending (None → Very High); not independently confirmed.
- **Gender**: exactly 2 categories (0–1), confirmed — but which is which is
  genuinely unresolved. I checked whether mean GPA differs meaningfully by
  group as a possible tiebreaker (1.919 vs. 1.894) and deliberately did **not**
  use it — the gap is small enough to be noise, this dataset is synthetic
  rather than necessarily reflecting real-world patterns, and guessing a
  demographic label from a stereotype-adjacent statistical pattern isn't a
  sound method regardless. This one needs your Colab notebook, not another guess.
- **Tutoring / Extracurricular / Sports / Music / Volunteering**: binary, order assumed (0=No, 1=Yes).

`GradeClass`'s relationship to `GPA` is noisier than a clean A–F cutoff
scheme in the real data (each `GradeClass` value spans a wide, overlapping
`GPA` range) — moot for this app since neither is ever sent to the model,
but worth knowing if you're also writing up the original dataset in your report.
