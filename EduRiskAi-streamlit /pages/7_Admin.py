"""
pages/7_Admin.py

Administrator-only page for:

- Creating Lecturer / Administrator accounts
- Viewing all registered users
- Enabling and disabling user accounts
- Viewing system-wide prediction statistics

Access is restricted using require_role("Administrator").
"""

import streamlit as st

from utils import auth
from utils.database import (
    create_user,
    get_all_predictions,
    get_all_students,
    init_db,
    list_users,
    set_user_active,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Admin",
    page_icon="🛠️",
    layout="wide"
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_db()


# ============================================================
# AUTHENTICATION / AUTHORIZATION
# ============================================================

# Only Administrators can access this page
auth.require_role("Administrator")

# Display logged-in user information in sidebar
auth.render_user_sidebar()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🛠️ Administration")

st.caption(
    "Manage system users and review system-wide student "
    "academic risk prediction statistics."
)


# ============================================================
# TABS
# ============================================================

tab_users, tab_stats = st.tabs(
    [
        "User Management",
        "Prediction Statistics"
    ]
)


# ============================================================
# TAB 1 — USER MANAGEMENT
# ============================================================

with tab_users:

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    st.subheader("Create User Account")

    with st.form(
        "create_user_form",
        clear_on_submit=True
    ):

        c1, c2 = st.columns(2)

        with c1:

            new_username = st.text_input(
                "Username",
                placeholder="e.g. lecturer01"
            )

            new_full_name = st.text_input(
                "Full Name",
                placeholder="e.g. John Smith"
            )

        with c2:

            new_password = st.text_input(
                "Temporary Password",
                type="password",
                help="Password must contain at least 8 characters."
            )

            new_role = st.selectbox(
                "Role",
                [
                    "Lecturer",
                    "Administrator"
                ]
            )

        submitted = st.form_submit_button(
            "Create Account",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PROCESS USER CREATION
    # --------------------------------------------------------

    if submitted:

        username = new_username.strip()
        full_name = new_full_name.strip()

        if not username:

            st.error(
                "Username is required."
            )

        elif not new_password:

            st.error(
                "Password is required."
            )

        elif len(new_password) < 8:

            st.error(
                "Password must contain at least 8 characters."
            )

        else:

            try:

                password_hash = auth.hash_password(
                    new_password
                )

                ok, message = create_user(
                    username,
                    password_hash,
                    new_role,
                    full_name
                )

                if ok:

                    st.success(message)

                else:

                    st.error(message)

            except Exception as e:

                st.error(
                    f"Unable to create account: {str(e)}"
                )


    # ========================================================
    # DISPLAY ALL USERS
    # ========================================================

    st.divider()

    st.subheader("All Users")


    try:

        users_df = list_users()

    except Exception as e:

        st.error(
            f"Unable to load user accounts: {str(e)}"
        )

        users_df = None


    if users_df is not None and not users_df.empty:

        # Create a copy so the original dataframe is not modified
        display_users = users_df.copy()

        # Convert database integer values to readable status
        if "is_active" in display_users.columns:

            display_users["is_active"] = (
                display_users["is_active"]
                .map({
                    1: "Active",
                    0: "Disabled"
                })
                .fillna("Unknown")
            )

        st.dataframe(
            display_users,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No user accounts were found."
        )


    # ========================================================
    # ENABLE / DISABLE USERS
    # ========================================================

    st.subheader("Enable / Disable a User")


    if users_df is not None and not users_df.empty:

        try:

            current_user = auth.current_user()

            current_user_id = current_user["id"]

            # Administrator must not disable their own account
            selectable = users_df[
                users_df["id"] != current_user_id
            ].copy()


            if selectable.empty:

                st.caption(
                    "No other accounts are currently available "
                    "to manage."
                )

            else:

                target = st.selectbox(
                    "Select User",
                    selectable["id"].tolist(),
                    format_func=lambda uid: (
                        selectable.loc[
                            selectable["id"] == uid,
                            "username"
                        ].iloc[0]
                    )
                )


                target_row = selectable[
                    selectable["id"] == target
                ].iloc[0]


                # Handle SQLite integer / boolean representation
                target_is_active = bool(
                    target_row["is_active"]
                )


                col_a, col_b = st.columns(2)


                # ------------------------------------------------
                # DISABLE USER
                # ------------------------------------------------

                with col_a:

                    disable_clicked = st.button(
                        "Disable User",
                        use_container_width=True,
                        disabled=not target_is_active
                    )


                # ------------------------------------------------
                # ENABLE USER
                # ------------------------------------------------

                with col_b:

                    enable_clicked = st.button(
                        "Enable User",
                        use_container_width=True,
                        disabled=target_is_active
                    )


                # ------------------------------------------------
                # PROCESS DISABLE
                # ------------------------------------------------

                if disable_clicked:

                    try:

                        set_user_active(
                            int(target),
                            False
                        )

                        st.success(
                            "User account disabled successfully."
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Unable to disable user: {str(e)}"
                        )


                # ------------------------------------------------
                # PROCESS ENABLE
                # ------------------------------------------------

                if enable_clicked:

                    try:

                        set_user_active(
                            int(target),
                            True
                        )

                        st.success(
                            "User account enabled successfully."
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Unable to enable user: {str(e)}"
                        )


        except Exception as e:

            st.error(
                f"Unable to manage user accounts: {str(e)}"
            )

    else:

        st.caption(
            "No users are available to manage."
        )


# ============================================================
# TAB 2 — PREDICTION STATISTICS
# ============================================================

with tab_stats:

    st.subheader(
        "System-Wide Prediction Statistics"
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        preds = get_all_predictions()
        students = get_all_students()

    except Exception as e:

        st.error(
            f"Unable to load prediction statistics: {str(e)}"
        )

        st.stop()


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    # Total students
    c1.metric(
        "Total Students",
        len(students)
    )


    # Total predictions
    c2.metric(
        "Total Predictions",
        len(preds)
    )


    # --------------------------------------------------------
    # PREDICTION DATA AVAILABLE
    # --------------------------------------------------------

    if not preds.empty:

        # ====================================================
        # AT-RISK RATE
        # ====================================================

        if "prediction" in preds.columns:

            at_risk_rate = (
                (preds["prediction"] == 1).mean()
                * 100
            )

            c3.metric(
                "At-Risk Rate",
                f"{at_risk_rate:.1f}%"
            )

        else:

            c3.metric(
                "At-Risk Rate",
                "N/A"
            )


        # ====================================================
        # HIGH-RISK COUNT
        # ====================================================

        if "risk_level" in preds.columns:

            high_risk_count = int(
                (
                    preds["risk_level"]
                    == "High Risk"
                ).sum()
            )

        else:

            high_risk_count = 0


        c4.metric(
            "High Risk Count",
            high_risk_count
        )


        # ====================================================
        # RISK LEVEL DISTRIBUTION
        # ====================================================

        st.divider()

        st.subheader(
            "Risk Level Distribution"
        )


        if "risk_level" in preds.columns:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT use:
            #
            # reset_index(names=[...])
            #
            # because older Pandas versions do not support
            # that argument for Series.reset_index().
            #
            # This implementation is compatible with a much
            # wider range of Pandas versions.
            # ------------------------------------------------

            risk_counts = (
                preds["risk_level"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Risk Level")
                .reset_index(name="Count")
            )


            st.dataframe(
                risk_counts,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "The prediction data does not contain "
                "a 'risk_level' column."
            )


        # ====================================================
        # CLASSIFICATION DISTRIBUTION
        # ====================================================

        if "prediction" in preds.columns:

            st.subheader(
                "Prediction Classification Summary"
            )


            at_risk_count = int(
                (
                    preds["prediction"] == 1
                ).sum()
            )


            not_at_risk_count = int(
                (
                    preds["prediction"] == 0
                ).sum()
            )


            classification_data = {
                "Classification": [
                    "At Risk",
                    "Not At Risk"
                ],
                "Count": [
                    at_risk_count,
                    not_at_risk_count
                ]
            }


            st.dataframe(
                classification_data,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # NO PREDICTIONS
    # --------------------------------------------------------

    else:

        c3.metric(
            "At-Risk Rate",
            "N/A"
        )

        c4.metric(
            "High Risk Count",
            0
        )

        st.info(
            "No predictions have been recorded yet. "
            "Prediction statistics will appear after "
            "student assessments are completed."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Administrator access only • "
    "AI-Based Student Academic Risk Prediction System"
)