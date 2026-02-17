"""
Authentication module.
Handles user login and credential validation.
"""
from datetime import datetime
from app.config import config
from app.constants import (
    SESSION_AUTHENTICATED,
    SESSION_LOGIN_TIME,
    SESSION_USERNAME,
)
import streamlit as st


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
        return (
            username == config.auth.username and password == config.auth.password
        )

    @staticmethod
    def login(username: str, password: str) -> bool:
        """
        Attempt to log in with provided credentials.

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
            return True
        return False

    @staticmethod
    def logout():
        """Log out the current user and clear session state."""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

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
