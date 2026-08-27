# Student Academic Risk Prediction & Early Intervention System
AI-Based Student Academic Risk Prediction System developed using Streamlit and Random Forest.

Final-year project (CSE6035) — a Streamlit web application around a
pre-trained Random Forest classifier that flags students who may be at
academic risk, for review by academic staff. **Decision support only** —
see `pages/6_About.py` for the full ethics statement.

## Current status

**Feature-complete, not data-complete.** Every page, the database, and
authentication are built and tested. The one thing not yet true: the model
making predictions is a synthetic placeholder, not your real Colab-trained
Random Forest — `model/student_risk_model.pkl` and
`model/student_risk_features.pkl` haven't been supplied yet (only
`student_risk_metadata.pkl` has). See "Using your real model" below —
it's a two-file swap, no code changes.

## Project structure

```
student-risk-streamlit/
├── app.py                      Entry point: bootstraps the DB, shows login, then the home screen
├── requirements.txt
├── create_placeholder_model.py Dev-only: generates a schema-matching placeholder model for testing
├── .streamlit/
│   ├── config.toml             Theme
│   └── secrets.toml.example    Copy to secrets.toml for real admin credentials
├── model/
│   ├── student_risk_model.pkl       <- your real Colab export goes here
│   ├── student_risk_features.pkl    <- your real Colab export goes here
│   ├── student_risk_metadata.pkl    <- already your real file
│   └── README.md               Exact expected schema + confirmed encoding notes
├── data/
│   └── real_student_features.csv    Real 2,392-row feature data, for testing Bulk Prediction
├── database/                   student_risk.db is created here at runtime (gitignored)
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Student_Prediction.py
│   ├── 3_Bulk_Prediction.py
│   ├── 4_Prediction_History.py
│   ├── 5_Analytics.py
│   ├── 6_About.py
│   └── 7_Admin.py              Not in the original spec's page list -- added because
│                                 "create/view/disable users" needed somewhere to live
└── utils/
    ├── database.py              SQLite schema + CRUD
    ├── model_utils.py           Model loading + prediction pipeline
    ├── validation.py            Input validation + encoding maps
    ├── recommendations.py       Rule-based intervention suggestions
    └── auth.py                  Authentication, hashing, session/role guards
```

## Setup

```bash
git clone <your-repo-url>
cd student-risk-streamlit
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
python create_placeholder_model.py   # only until your real .pkl files are ready
streamlit run app.py
```

Open `http://localhost:8501`. Sign in with **`admin`** / **`ChangeMe123!`**
(the default seeded on first run).

**Before showing this to anyone else**, copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml` and set a real `ADMIN_PASSWORD` — the default
is intentionally public (it's in this README).

## Using your real model

1. Export from Colab:
   ```python
   joblib.dump(model, "student_risk_model.pkl")
   joblib.dump(list(X_train.columns), "student_risk_features.pkl")
   ```
2. Copy both into `model/`, replacing the placeholder files.
3. Delete `model/.placeholder`.
4. Restart the app. The warning banners disappear automatically — no code changes.

## Known open item

One categorical encoding — **Gender** — could not be verified against an
accessible source or the raw dataset (see `model/README.md` for what *was*
confirmed). Check your Colab preprocessing cell and update
`GENDER_MAP` in `utils/validation.py` if it differs from `{"Male": 0, "Female": 1}`.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub, **including the real `model/*.pkl` files** —
   this app never retrains on deploy, so Cloud needs them committed (check
   they're comfortably under GitHub's 100MB/file limit; a Random Forest on
   12 features over ~2,400 rows normally is).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, branch `main`, main file `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   ADMIN_USERNAME = "admin"
   ADMIN_PASSWORD = "ChangMe123!"
   ```
4. Deploy. First load creates `database/student_risk.db` and seeds the
   admin account from those secrets automatically.

Note: Community Cloud's filesystem is ephemeral on redeploy — `database/student_risk.db`
resets when the app restarts/redeploys. Fine for a course demo; for anything
longer-lived, an external database would be needed (out of scope here).

## Testing

Manual verification performed throughout development is documented in the
conversation this project was built in — real DB writes read back and
asserted, a real version-mismatch bug reproduced and fixed, the full
2,392-row real dataset run through bulk prediction end to end. A formal
`pytest` suite has not yet been written.
