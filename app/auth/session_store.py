"""
Server-side session store for persistent login across page reloads.
Uses an in-memory dict keyed by random session tokens.
"""
import secrets
from datetime import datetime, timedelta

# Module-level dict: persists across Streamlit reruns while the server is running.
# Cleared only when the Docker container restarts (users will re-login in that case).
_store: dict[str, dict] = {}

SESSION_EXPIRY_DAYS = 30
URL_PARAM = "s"


def create_session(username: str) -> str:
    """Create a new session and return its token."""
    token = secrets.token_urlsafe(32)
    _store[token] = {
        "username": username,
        "expiry": datetime.now() + timedelta(days=SESSION_EXPIRY_DAYS),
    }
    return token


def validate_session(token: str) -> tuple[bool, str]:
    """Validate a session token. Returns (valid, username)."""
    data = _store.get(token)
    if not data:
        return False, ""
    if datetime.now() > data["expiry"]:
        del _store[token]
        return False, ""
    return True, data["username"]


def delete_session(token: str) -> None:
    """Invalidate a session token."""
    _store.pop(token, None)
