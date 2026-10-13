"""Credential detection and sensitivity inspection utilities for platform keys."""

_SENSITIVE_KEY_TERMS = (
    "token",
    "secret",
    "password",
    "api_key",
    "api-key",
    "apikey",
    "credential",
    "authorization",
    "auth",
    "bearer",
    "modal-key",
    "modal_key",
    "private_key",
    "access_key",
)
_REDACTED = "[REDACTED]"


def is_sensitive_key(key: str) -> bool:
    """Returns True if the key represents a sensitive platform credential or secret."""
    lower_k = key.lower()
    return any(term in lower_k for term in _SENSITIVE_KEY_TERMS)


def _is_sensitive_key(key: str) -> bool:
    """Legacy check matching sensitive terms against key names."""
    return is_sensitive_key(key)
