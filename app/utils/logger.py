"""
Logging configuration module.
"""
import logging
from app.config import config


def setup_logger(name: str) -> logging.Logger:
    """
    Set up logger with configured settings.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.logging.level)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(config.logging.level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger("streamlit_analytics")
