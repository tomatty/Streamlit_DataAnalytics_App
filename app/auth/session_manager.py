"""
Session state management module.
Centralizes session state initialization and management.
"""
import streamlit as st
from app.constants import (
    SESSION_AUTHENTICATED,
    SESSION_LOGIN_TIME,
    SESSION_USERNAME,
    SESSION_UPLOADED_FILE_NAME,
    SESSION_RAW_DATA,
    SESSION_PROCESSED_DATA,
    SESSION_DATA_METADATA,
    SESSION_PREPROCESSING_STEPS,
    SESSION_ANALYSIS_HISTORY,
    SESSION_CURRENT_ANALYSIS,
    SESSION_CURRENT_PAGE,
    SESSION_SETTINGS,
)
from app.config import config


class SessionManager:
    """Manages Streamlit session state."""

    @staticmethod
    def init_session_state():
        """Initialize session state with default values."""
        # Authentication state
        if SESSION_AUTHENTICATED not in st.session_state:
            st.session_state[SESSION_AUTHENTICATED] = False
        if SESSION_LOGIN_TIME not in st.session_state:
            st.session_state[SESSION_LOGIN_TIME] = None
        if SESSION_USERNAME not in st.session_state:
            st.session_state[SESSION_USERNAME] = ""

        # Data state
        if SESSION_UPLOADED_FILE_NAME not in st.session_state:
            st.session_state[SESSION_UPLOADED_FILE_NAME] = None
        if SESSION_RAW_DATA not in st.session_state:
            st.session_state[SESSION_RAW_DATA] = None
        if SESSION_PROCESSED_DATA not in st.session_state:
            st.session_state[SESSION_PROCESSED_DATA] = None
        if SESSION_DATA_METADATA not in st.session_state:
            st.session_state[SESSION_DATA_METADATA] = {}

        # Preprocessing state
        if SESSION_PREPROCESSING_STEPS not in st.session_state:
            st.session_state[SESSION_PREPROCESSING_STEPS] = []

        # Analysis state
        if SESSION_ANALYSIS_HISTORY not in st.session_state:
            st.session_state[SESSION_ANALYSIS_HISTORY] = []
        if SESSION_CURRENT_ANALYSIS not in st.session_state:
            st.session_state[SESSION_CURRENT_ANALYSIS] = {}

        # UI state
        if SESSION_CURRENT_PAGE not in st.session_state:
            st.session_state[SESSION_CURRENT_PAGE] = ""

        # Analysis settings
        if SESSION_SETTINGS not in st.session_state:
            st.session_state[SESSION_SETTINGS] = {
                "confidence_level": config.analysis.default_confidence_level,
                "significance_level": config.analysis.default_significance_level,
                "max_clustering_iterations": config.analysis.max_clustering_iterations,
                "max_features": config.analysis.default_max_features,
                "n_topics": config.analysis.default_n_topics,
            }

    @staticmethod
    def has_data() -> bool:
        """
        Check if data has been uploaded.

        Returns:
            bool: True if data exists, False otherwise
        """
        return st.session_state.get(SESSION_RAW_DATA) is not None

    @staticmethod
    def get_data():
        """
        Get the current data (processed if available, otherwise raw).

        Returns:
            pd.DataFrame | None: Current data or None if no data
        """
        if st.session_state.get(SESSION_PROCESSED_DATA) is not None:
            return st.session_state[SESSION_PROCESSED_DATA]
        return st.session_state.get(SESSION_RAW_DATA)

    @staticmethod
    def get_raw_data():
        """
        Get the raw (original) data.

        Returns:
            pd.DataFrame | None: Raw data or None if no data
        """
        return st.session_state.get(SESSION_RAW_DATA)

    @staticmethod
    def set_data(data, is_raw=True):
        """
        Set data in session state.

        Args:
            data: DataFrame to store
            is_raw: If True, store as raw data; otherwise as processed data
        """
        if is_raw:
            st.session_state[SESSION_RAW_DATA] = data
            st.session_state[SESSION_PROCESSED_DATA] = data.copy()
        else:
            st.session_state[SESSION_PROCESSED_DATA] = data

    @staticmethod
    def add_preprocessing_step(step: dict):
        """
        Add a preprocessing step to history.

        Args:
            step: Dictionary describing the preprocessing step
        """
        st.session_state[SESSION_PREPROCESSING_STEPS].append(step)

    @staticmethod
    def get_preprocessing_steps() -> list:
        """
        Get all preprocessing steps.

        Returns:
            list: List of preprocessing steps
        """
        return st.session_state.get(SESSION_PREPROCESSING_STEPS, [])

    @staticmethod
    def clear_preprocessing_history():
        """Clear preprocessing history."""
        st.session_state[SESSION_PREPROCESSING_STEPS] = []

    @staticmethod
    def add_analysis_result(result: dict):
        """
        Add an analysis result to history.

        Args:
            result: Dictionary containing analysis results
        """
        st.session_state[SESSION_ANALYSIS_HISTORY].append(result)
        st.session_state[SESSION_CURRENT_ANALYSIS] = result

    @staticmethod
    def get_analysis_history() -> list:
        """
        Get all analysis results.

        Returns:
            list: List of analysis results
        """
        return st.session_state.get(SESSION_ANALYSIS_HISTORY, [])

    @staticmethod
    def clear_data():
        """Clear all data from session state."""
        st.session_state[SESSION_UPLOADED_FILE_NAME] = None
        st.session_state[SESSION_RAW_DATA] = None
        st.session_state[SESSION_PROCESSED_DATA] = None
        st.session_state[SESSION_DATA_METADATA] = {}
        st.session_state[SESSION_PREPROCESSING_STEPS] = []
        st.session_state[SESSION_ANALYSIS_HISTORY] = []
        st.session_state[SESSION_CURRENT_ANALYSIS] = {}

    @staticmethod
    def get_setting(key: str, default=None):
        """
        Get a specific setting value.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        return st.session_state.get(SESSION_SETTINGS, {}).get(key, default)

    @staticmethod
    def update_setting(key: str, value):
        """
        Update a specific setting value.

        Args:
            key: Setting key
            value: New value
        """
        if SESSION_SETTINGS not in st.session_state:
            st.session_state[SESSION_SETTINGS] = {}
        st.session_state[SESSION_SETTINGS][key] = value

    @staticmethod
    def get_all_settings() -> dict:
        """
        Get all settings.

        Returns:
            dict: All settings
        """
        return st.session_state.get(SESSION_SETTINGS, {})

    @staticmethod
    def update_all_settings(settings: dict):
        """
        Update all settings.

        Args:
            settings: Dictionary of settings
        """
        st.session_state[SESSION_SETTINGS] = settings
