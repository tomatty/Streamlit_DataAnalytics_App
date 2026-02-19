"""
Application configuration module.
Loads settings from environment variables.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AuthConfig:
    """Authentication configuration."""

    users: dict

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load authentication config from environment variables.

        Reads APP_USERS in the format "user1:pass1,user2:pass2".
        Falls back to APP_USERNAME / APP_PASSWORD for backward compatibility.
        """
        users_str = os.getenv("APP_USERS", "")
        if users_str:
            users = {}
            for entry in users_str.split(","):
                entry = entry.strip()
                if ":" in entry:
                    name, password = entry.split(":", 1)
                    users[name.strip()] = password.strip()
        else:
            # Backward compatibility: single user from legacy env vars
            username = os.getenv("APP_USERNAME", "admin")
            password = os.getenv("APP_PASSWORD", "admin")
            users = {username: password}
        return cls(users=users)


@dataclass
class AppConfig:
    """Application configuration."""

    name: str
    max_upload_size_mb: int
    allowed_file_types: list[str]

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load application config from environment variables."""
        allowed_types = os.getenv("ALLOWED_FILE_TYPES", "csv,xlsx,xls,json")
        return cls(
            name=os.getenv("APP_NAME", "Data Analytics Platform"),
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "200")),
            allowed_file_types=allowed_types.split(","),
        )


@dataclass
class AnalysisConfig:
    """Analysis configuration."""

    default_confidence_level: float
    max_clustering_iterations: int
    default_significance_level: float
    default_max_features: int
    default_n_topics: int

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        """Load analysis config from environment variables."""
        return cls(
            default_confidence_level=float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", "0.95")),
            max_clustering_iterations=int(os.getenv("MAX_CLUSTERING_ITERATIONS", "300")),
            default_significance_level=float(
                os.getenv("DEFAULT_SIGNIFICANCE_LEVEL", "0.05")
            ),
            default_max_features=int(os.getenv("DEFAULT_MAX_FEATURES", "100")),
            default_n_topics=int(os.getenv("DEFAULT_N_TOPICS", "5")),
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str
    log_file: str

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load logging config from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "app.log"),
        )


@dataclass
class SessionConfig:
    """Session configuration."""

    timeout_minutes: int

    @classmethod
    def from_env(cls) -> "SessionConfig":
        """Load session config from environment variables."""
        return cls(timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")))


class Config:
    """Main configuration class."""

    def __init__(self):
        self.auth = AuthConfig.from_env()
        self.app = AppConfig.from_env()
        self.analysis = AnalysisConfig.from_env()
        self.logging = LoggingConfig.from_env()
        self.session = SessionConfig.from_env()


# Global configuration instance
config = Config()
