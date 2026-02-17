"""
Authentication module.
Handles user login and credential validation.
"""
from datetime import datetime

import streamlit as st

from app.auth.session_store import create_session, validate_session, delete_session, URL_PARAM
from app.config import config
from app.constants import (
    SESSION_AUTHENTICATED,
    SESSION_LOGIN_TIME,
    SESSION_USERNAME,
)


class Authenticator:
    """Handles user authentication."""

    @staticmethod
    def validate_credentials(username: str, password: str) -> bool:
        """
        Validate user credentials against environment variables.

        Args:
            username: Username to validate
            password: Password to validate

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        return username == config.auth.username and password == config.auth.password

    @staticmethod
    def login(username: str, password: str) -> bool:
        """
        Attempt to log in with provided credentials.
        On success, creates a server-side session and stores its token in
        the URL query parameter so it survives page reloads.

        Args:
            username: Username
            password: Password

        Returns:
            bool: True if login successful, False otherwise
        """
        if Authenticator.validate_credentials(username, password):
            st.session_state[SESSION_AUTHENTICATED] = True
            st.session_state[SESSION_LOGIN_TIME] = datetime.now()
            st.session_state[SESSION_USERNAME] = username
            token = create_session(username)
            st.query_params[URL_PARAM] = token
            return True
        return False

    @staticmethod
    def logout() -> None:
        """Log out the current user, invalidate the session token, and clear state."""
        token = st.query_params.get(URL_PARAM)
        if token:
            delete_session(token)
        st.query_params.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    @staticmethod
    def restore_from_url() -> bool:
        """
        Attempt to restore authentication from the URL query parameter.
        Called at the start of every run so that page reloads do not
        require the user to log in again.

        Returns:
            bool: True if session was restored, False otherwise
        """
        if st.session_state.get(SESSION_AUTHENTICATED):
            return True
        token = st.query_params.get(URL_PARAM)
        if not token:
            return False
        valid, username = validate_session(token)
        if valid:
            st.session_state[SESSION_AUTHENTICATED] = True
            st.session_state[SESSION_LOGIN_TIME] = datetime.now()
            st.session_state[SESSION_USERNAME] = username
            return True
        # Token invalid/expired — remove it from the URL
        st.query_params.clear()
        return False

    @staticmethod
    def is_authenticated() -> bool:
        """
        Check if user is authenticated.

        Returns:
            bool: True if user is authenticated, False otherwise
        """
        return st.session_state.get(SESSION_AUTHENTICATED, False)

    @staticmethod
    def get_current_user() -> str:
        """
        Get the current logged-in username.

        Returns:
            str: Current username or empty string if not authenticated
        """
        return st.session_state.get(SESSION_USERNAME, "")

    @staticmethod
    def get_login_time() -> datetime | None:
        """
        Get the login time of the current user.

        Returns:
            datetime | None: Login time or None if not authenticated
        """
        return st.session_state.get(SESSION_LOGIN_TIME)

    @staticmethod
    def check_session_timeout() -> bool:
        """
        Check if the session has timed out.

        Returns:
            bool: True if session has timed out, False otherwise
        """
        if not Authenticator.is_authenticated():
            return True

        login_time = Authenticator.get_login_time()
        if login_time is None:
            return True

        elapsed_minutes = (datetime.now() - login_time).total_seconds() / 60
        if elapsed_minutes > config.session.timeout_minutes:
            Authenticator.logout()
            return True

        return False
