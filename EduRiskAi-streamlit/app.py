"""
app.py

Entry point for the AI-Based Student Academic Risk Prediction and Early
Intervention System.

This stage adds:
  1. Database initialisation (creates tables on first run, idempotent).
  2. A default Administrator account, seeded once into an empty database.
  3. A real login gate: nothing past the hero section renders until
     utils/auth.login() succeeds against a hashed password in SQLite.

Student management, the full prediction workflow with recommendations, the
dashboard, and analytics are added in the stages that follow -- this file
keeps growing, but nothing built here gets renamed later.
"""

import streamlit as st

from utils import auth
from utils import database as db
from utils.model_utils import (
    ModelLoadError,
    get_model_load_warning,
    load_feature_list,
    load_metadata,
    load_model,
    using_placeholder_model,
)

st.set_page_config(
    page_title="Student Academic Risk Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def bootstrap() -> None:
    """Create tables and seed a default admin if this is a brand-new database.
    Idempotent -- safe to call on every script rerun."""
    db.init_db()
    admin_username, admin_password, _ = auth.get_seed_admin_credentials()
    db.seed_default_admin(admin_username, auth.hash_password(admin_password))


def show_hero() -> None:
    st.markdown(
        "<div style='text-align:center; padding: 2rem 0 1rem 0;'>"
        "<div style='font-size:3rem;'>🎓</div>"
        "<h1 style='margin-bottom:0;'>Academic Risk Decision Support</h1>"
        "<p style='opacity:0.75; font-size:1.05rem;'>"
        "AI-Powered Student Early Warning &amp; Intervention Platform</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def show_login() -> None:
    _, col_form, _ = st.columns([1, 1.2, 1])
    with col_form:
        with st.container(border=True):
            st.markdown("#### System Authentication")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. jdoe")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                success, message = auth.login(username, password)
                if success:
                    st.rerun()
                else:
                    st.error(message)

            admin_username, _, using_default = auth.get_seed_admin_credentials()
            if using_default:
                st.caption(
                    f"First run: sign in as **{admin_username}** with the default "
                    "password in `utils/auth.py` (`DEFAULT_ADMIN_PASSWORD`). Set "
                    "`.streamlit/secrets.toml` from the included `.example` file "
                    "to use a real one instead."
                )


def show_model_status() -> None:
    """Compact model diagnostics, shown to signed-in users."""
    st.subheader("Model status")
    try:
        model = load_model()
        feature_list = load_feature_list()
        metadata = load_metadata()
    except ModelLoadError as exc:
        st.error(
            "The prediction model could not be loaded. The application "
            f"cannot make predictions until this is fixed.\n\n**Details:** {exc}"
        )
        return

    algorithm_name = metadata.get("selected_algorithm") or metadata.get("model_name", "Unknown")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model file", "Loaded")
    col2.metric("Input features", len(feature_list))
    col3.metric("Algorithm", algorithm_name)

    if using_placeholder_model():
        st.warning(
            "⚠️ `model.pkl` / `features.pkl` are still the **placeholder** "
            "files from `create_placeholder_model.py` -- predictions are "
            "not meaningful yet."
        )
    version_warning = get_model_load_warning()
    if version_warning:
        st.warning(f"⚠️ {version_warning}")

    with st.expander("Model metadata"):
        st.json(metadata) if metadata else st.caption("No metadata file found.")


def show_authenticated_home() -> None:
    auth.render_user_sidebar()
    user = auth.current_user()
    st.title("🎓 Student Academic Risk Prediction & Early Intervention System")
    st.success(f"Signed in as **{user['full_name'] or user['username']}** ({user['role']})")
    st.markdown(
        "Use the sidebar to navigate. Student management, the full "
        "prediction workflow, dashboard, and analytics are added in the "
        "stages that follow."
    )
    show_model_status()


def main() -> None:
    bootstrap()
    if not auth.is_authenticated():
        show_hero()
        show_login()
        return
    show_authenticated_home()


if __name__ == "__main__":
    main()
