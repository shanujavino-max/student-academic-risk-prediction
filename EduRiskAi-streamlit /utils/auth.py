"""
utils/auth.py

Authentication, password hashing, and session-state management.

Important architectural note: Streamlit's classic pages/ folder navigation
lists every page in the sidebar regardless of login state -- a login gate
in app.py alone does NOT stop someone from clicking straight to another
page. Every page (including app.py's authenticated section) must call
require_login() -- or require_role(...) for admin-only pages -- at the top
of its own script. st.session_state IS shared across all pages in the same
browser session, so the check itself is cheap and consistent everywhere.
"""

from __future__ import annotations

import bcrypt
import streamlit as st

from utils import database as db

SESSION_KEY = "auth_user"

# Only used when no real secret is configured -- see
# .streamlit/secrets.toml.example for how to set a real one.
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "ChangeMe123!"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed/legacy hash -- fail closed, never raise into the login page.
        return False


def get_seed_admin_credentials() -> tuple[str, str, bool]:
    """
    Returns (username, password, is_default) for the admin account to seed
    on first run. Reads from Streamlit secrets if configured, otherwise
    falls back to an obviously-fake default so local testing is never
    blocked on setup. is_default=True tells the caller to keep nagging
    until a real one is set -- this default must never be relied on
    anywhere the app is reachable by anyone but you.
    """
    try:
        username = st.secrets.get("ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)
        password = st.secrets.get("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
    except Exception:
        # st.secrets can raise if no secrets.toml exists at all yet.
        username, password = DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD
    return username, password, password == DEFAULT_ADMIN_PASSWORD


def current_user() -> dict | None:
    return st.session_state.get(SESSION_KEY)


def is_authenticated() -> bool:
    return current_user() is not None


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Attempt to authenticate. On success, stores the user in session_state.
    On failure, always returns the SAME generic message regardless of
    whether the username exists, the account is just disabled, or the
    password is wrong -- except for the disabled case, which is safe to
    disclose since it doesn't help guess a valid password.
    """
    if not username or not password:
        return False, "Enter both a username and password."

    user_row = db.get_user_by_username(username.strip())
    if user_row is None or not verify_password(password, user_row["password_hash"]):
        return False, "Invalid username or password."
    if not user_row["is_active"]:
        return False, "This account has been disabled. Contact an administrator."

    st.session_state[SESSION_KEY] = {
        "id": user_row["id"],
        "username": user_row["username"],
        "role": user_row["role"],
        "full_name": user_row["full_name"],
    }
    db.log_activity(user_row["id"], "login")
    return True, f"Welcome back, {user_row['full_name'] or user_row['username']}."


def logout() -> None:
    user = current_user()
    if user:
        db.log_activity(user["id"], "logout")
    st.session_state.pop(SESSION_KEY, None)


def require_login() -> dict:
    """
    Call at the top of every page that requires sign-in. Stops the script
    and shows a friendly message if nobody is logged in -- see the module
    docstring for why every page needs this individually.
    """
    user = current_user()
    if user is None:
        st.warning("Please sign in from the home page to continue.")
        st.stop()
    return user


def require_role(*allowed_roles: str) -> dict:
    """Call at the top of pages restricted to specific roles."""
    user = require_login()
    if user["role"] not in allowed_roles:
        st.error(f"This page is only available to: {', '.join(allowed_roles)}.")
        st.stop()
    return user


def render_user_sidebar() -> None:
    """Call from authenticated pages to show who's signed in + a logout button."""
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Signed in as **{user['full_name'] or user['username']}**")
        st.caption(f"Role: {user['role']}")
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
