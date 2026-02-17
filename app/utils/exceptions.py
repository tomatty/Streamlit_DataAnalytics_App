"""
Custom exceptions module.
"""


class DataAnalyticsException(Exception):
    """Base exception for data analytics application."""
    pass


class DataLoadError(DataAnalyticsException):
    """Exception raised when data loading fails."""
    pass


class DataValidationError(DataAnalyticsException):
    """Exception raised when data validation fails."""
    pass


class AnalysisError(DataAnalyticsException):
    """Exception raised when analysis fails."""
    pass


class AuthenticationError(DataAnalyticsException):
    """Exception raised when authentication fails."""
    pass


class ConfigurationError(DataAnalyticsException):
    """Exception raised when configuration is invalid."""
    pass
