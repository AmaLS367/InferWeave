"""Credential detection and sensitivity inspection utilities for platform keys."""

import os
import re
from collections.abc import Iterable
from typing import Any

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

PLATFORM_CREDENTIAL_ENV = frozenset(
    {
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
        "LIGHTNING_USER_ID",
        "LIGHTNING_API_KEY",
        "LIGHTNING_AUTH_TOKEN",
        "RUNPOD_API_KEY",
        "VAST_API_KEY",
    }
)
"""Control-plane credential variables that must never reach a model container or persistence."""

_BENIGN_TOKEN_FLAGS = frozenset(
    {
        "eos-token-id",
        "bos-token-id",
        "pad-token-id",
        "unk-token-id",
        "sep-token-id",
        "cls-token-id",
        "mask-token-id",
        "stop-token-ids",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "stop_token_ids",
    }
)

_URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s@]+)@")
_FLAG_ASSIGN_RE = re.compile(
    r"((?:^|(?<=[\s;|&(]))--?[A-Za-z0-9][A-Za-z0-9_.-]*)=(\"[^\"]*\"|'[^']*'|[^\s;|&)]*)"
)
_FLAG_SPACE_RE = re.compile(
    r"((?:^|(?<=[\s;|&(]))--?[A-Za-z0-9][A-Za-z0-9_.-]*)\s+(\"[^\"]*\"|'[^']*'|[^\s;|&)\-][^\s;|&)]*)"
)
_ENV_ASSIGN_RE = re.compile(
    r"((?:^|(?<=[\s;|&(?]))(?:export\s+)?)([A-Za-z_][A-Za-z0-9_.-]*)=(\"[^\"]*\"|'[^']*'|[^\s;|&)]*)"
)
_SENSITIVE_HEADER_LINE_RE = re.compile(
    r"(?i)\b(authorization|proxy-authorization|x-api-key|api-key|modal-key|modal-secret|cookie|set-cookie)\s*:\s*(?!//)([^\r\n\"';|&)]+)"
)
_SENSITIVE_COLON_SPACE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]*)\s*:\s+([^\r\n\"';|&)]+)"
)
_AUTH_SCHEME_RE = re.compile(
    r"(?i)\b(bearer|basic)\s+(?!\[REDACTED\])([A-Za-z0-9._~+/=-]{4,})"
)
_KNOWN_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:"
    r"hf_[A-Za-z0-9]{6,}"
    r"|sk-[A-Za-z0-9_-]{6,}"
    r"|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{6,}"
    r"|github_pat_[A-Za-z0-9_]{6,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{12,}"
    r"|xox[baprs]-[A-Za-z0-9-]{6,}"
    r"|SENTINEL-[A-Za-z0-9_-]+"
    r")\b"
)


_BENIGN_KEY_TERMS = frozenset(
    {
        "sort_key",
        "sort-key",
        "sortkey",
        "sort_keys",
        "cache_key",
        "cache-key",
        "cachekey",
        "cache_keys",
        "primary_key",
        "primary-key",
        "primarykey",
        "primary_keys",
        "partition_key",
        "partition-key",
        "partitionkey",
        "partition_keys",
        "foreign_key",
        "foreign-key",
        "foreignkey",
        "foreign_keys",
        "routing_key",
        "routing-key",
        "routingkey",
        "routing_keys",
        "cluster_key",
        "cluster-key",
        "clusterkey",
        "cluster_keys",
        "shard_key",
        "shard-key",
        "shardkey",
        "shard_keys",
        "id_key",
        "id-key",
        "idkey",
        "id_keys",
        "search_key",
        "search_keys",
        "group_key",
        "group_keys",
        "filter_key",
        "filter_keys",
        "unique_key",
        "unique_keys",
        "composite_key",
        "surrogate_key",
        "natural_key",
        "dedup_key",
        "idempotency_key",
        "idempotency-key",
        "key_name",
        "key_type",
        "key_column",
        "key_field",
        "key_format",
        "key_algorithm",
        "key_algo",
        "key_id",
        "key_ids",
    }
)

_SENSITIVE_KEY_PREFIXES = frozenset(
    {
        "api",
        "modal",
        "private",
        "priv",
        "access",
        "secret",
        "sec",
        "auth",
        "client",
        "service",
        "signing",
        "encryption",
        "encrypt",
        "master",
        "license",
        "ssh",
        "token",
        "cert",
        "certificate",
        "credential",
    }
)


def is_sensitive_key(key: str) -> bool:
    """Returns True if the key represents a sensitive platform credential or secret."""
    cleaned = key.strip().lower()
    if not cleaned:
        return False
    if cleaned in _BENIGN_KEY_TERMS or cleaned in _BENIGN_TOKEN_FLAGS:
        return False
    if key.upper() in PLATFORM_CREDENTIAL_ENV:
        return True

    if cleaned in (
        "key",
        "keys",
        "secret",
        "secrets",
        "token",
        "tokens",
        "credential",
        "credentials",
        "password",
        "passwords",
        "passwd",
        "pwd",
    ):
        return True

    parts = [p for p in re.split(r"[-_.]", cleaned) if p]
    if parts:
        if parts[0] in (
            "sort",
            "cache",
            "partition",
            "primary",
            "foreign",
            "routing",
            "cluster",
            "shard",
            "id",
            "search",
            "group",
            "filter",
            "unique",
            "composite",
            "surrogate",
            "natural",
            "dedup",
            "idempotency",
        ) and parts[-1] in ("key", "keys"):
            return False
        if parts[-1] in (
            "name",
            "type",
            "column",
            "field",
            "format",
            "algorithm",
            "algo",
            "id",
            "ids",
        ) and parts[0] in ("key", "keys"):
            return False
        if ("tokenizer" in parts or "tokens" in parts) and not any(
            prefix in parts for prefix in _SENSITIVE_KEY_PREFIXES
        ):
            return False
        if any(k_part in ("key", "keys") for k_part in parts) and any(
            prefix in parts for prefix in _SENSITIVE_KEY_PREFIXES
        ):
            return True

    return any(term in cleaned for term in _SENSITIVE_KEY_TERMS)


def _is_sensitive_key(key: str) -> bool:
    """Legacy check matching sensitive terms against key names."""
    return is_sensitive_key(key)


def is_sensitive_header(key: str) -> bool:
    """Returns True if the HTTP header name represents a sensitive credential or cookie."""
    lower_k = key.lower()
    return (
        is_sensitive_key(key)
        or lower_k in ("cookie", "set-cookie", "proxy-authorization", "x-api-key")
    )


def is_sensitive_cli_flag(flag: str) -> bool:
    """Returns True if a CLI option flag name represents a credential or secret argument."""
    cleaned = flag.lstrip("-").lower()
    if not cleaned or cleaned in _BENIGN_TOKEN_FLAGS:
        return False
    if cleaned.upper() in PLATFORM_CREDENTIAL_ENV:
        return True
    parts = [p for p in re.split(r"[-_.]", cleaned) if p]
    if "tokenizer" in parts or "tokens" in parts:
        return False
    if any(
        term in parts
        for term in (
            "token",
            "secret",
            "secrets",
            "password",
            "passwd",
            "pwd",
            "apikey",
            "credential",
            "credentials",
            "authorization",
            "auth",
            "bearer",
        )
    ):
        return True
    if "key" in parts and any(
        prefix in parts
        for prefix in (
            "api",
            "modal",
            "private",
            "access",
            "secret",
            "auth",
            "client",
            "service",
        )
    ):
        return True
    return any(
        term in cleaned
        for term in (
            "api_key",
            "api-key",
            "apikey",
            "password",
            "secret",
            "credential",
            "authorization",
            "bearer",
            "modal-key",
            "modal_key",
            "private_key",
            "private-key",
            "access_key",
            "access-key",
        )
    )


def known_secret_values(secrets: Iterable[str] = ()) -> tuple[str, ...]:
    """Returns known secret strings (length >= 4) combined with ambient platform credentials."""
    explicit = tuple(
        s for s in secrets if isinstance(s, str) and len(s) >= 4 and s != _REDACTED
    )
    ambient = tuple(
        v
        for k in PLATFORM_CREDENTIAL_ENV
        if len(v := os.environ.get(k, "")) >= 4 and v != _REDACTED
    )
    combined = dict.fromkeys(sorted(explicit + ambient, key=len, reverse=True))
    return tuple(combined)


def sanitize_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Redacts known secrets, credential assignments, auth headers, and token patterns in text."""
    if not text or text == _REDACTED:
        return text

    result = text
    for sec in known_secret_values(secrets):
        if sec in result:
            result = result.replace(sec, _REDACTED)

    result = _URL_USERINFO_RE.sub(f"\\1{_REDACTED}@", result)

    def _replace_flag_assign(match: re.Match[str]) -> str:
        flag = match.group(1)
        if is_sensitive_cli_flag(flag.strip()):
            return f"{flag}={_REDACTED}"
        return match.group(0)

    result = _FLAG_ASSIGN_RE.sub(_replace_flag_assign, result)

    def _replace_flag_space(match: re.Match[str]) -> str:
        flag = match.group(1)
        if is_sensitive_cli_flag(flag.strip()):
            return f"{flag} {_REDACTED}"
        return match.group(0)

    result = _FLAG_SPACE_RE.sub(_replace_flag_space, result)

    def _replace_env_assign(match: re.Match[str]) -> str:
        prefix, key = match.group(1), match.group(2)
        if is_sensitive_key(key):
            return f"{prefix}{key}={_REDACTED}"
        return match.group(0)

    result = _ENV_ASSIGN_RE.sub(_replace_env_assign, result)
    result = _SENSITIVE_HEADER_LINE_RE.sub(f"\\1: {_REDACTED}", result)

    def _replace_colon_space(match: re.Match[str]) -> str:
        header_name = match.group(1)
        if is_sensitive_header(header_name):
            return f"{header_name}: {_REDACTED}"
        return match.group(0)

    result = _SENSITIVE_COLON_SPACE_RE.sub(_replace_colon_space, result)
    result = _AUTH_SCHEME_RE.sub(f"\\1 {_REDACTED}", result)
    result = _KNOWN_TOKEN_PREFIX_RE.sub(_REDACTED, result)
    return result


def sanitize_args(args: list[str], secrets: Iterable[str] = ()) -> list[str]:
    """Sanitizes a container argv token list, redacting secret flag values and inline secrets."""
    result: list[str] = []
    redact_next = False
    header_next = False
    for token in args:
        s_token = str(token)
        if redact_next:
            redact_next = False
            if not s_token.startswith("--"):
                result.append(_REDACTED)
                continue
        if header_next:
            header_next = False
            if ":" in s_token:
                h_name, _ = s_token.split(":", 1)
                if is_sensitive_header(h_name.strip()):
                    result.append(f"{h_name}: {_REDACTED}")
                    continue
        if s_token.startswith("-") and "=" not in s_token:
            stripped = s_token.lstrip("-").lower()
            if stripped in ("h", "header"):
                header_next = True
            elif is_sensitive_cli_flag(s_token):
                redact_next = True
        result.append(sanitize_text(s_token, secrets=secrets))
    return result


def redact_deep(value: Any) -> Any:
    """Recursively redacts all leaf values under a sensitive key while preserving container shapes."""
    if isinstance(value, dict):
        return {k: redact_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_REDACTED for _ in value]
    if isinstance(value, tuple):
        return [_REDACTED for _ in value]
    return _REDACTED


def sanitize_metadata(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively sanitizes arbitrary metadata, redacting sensitive keys and secret-bearing values."""
    if isinstance(value, dict):
        sanitized_dict: dict[Any, Any] = {}

        # Check if this dict represents an explicit key-value property pair, e.g. {"key": "env", "value": "prod"}
        is_kv_pair = (
            "key" in value
            and ("value" in value or "val" in value)
            and len(value) <= 3
        )
        if is_kv_pair:
            k_val = value.get("key")
            k_str = str(k_val) if k_val is not None else ""
            val_key = "value" if "value" in value else "val"
            if is_sensitive_key(k_str) or is_sensitive_header(k_str):
                for k, v in value.items():
                    if k == val_key:
                        sanitized_dict[k] = redact_deep(v)
                    elif k == "key":
                        sanitized_dict[k] = (
                            sanitize_text(str(v), secrets=secrets)
                            if isinstance(v, str)
                            else v
                        )
                    else:
                        sanitized_dict[k] = sanitize_metadata(v, secrets=secrets)
                return sanitized_dict
            else:
                for k, v in value.items():
                    sanitized_dict[k] = sanitize_metadata(v, secrets=secrets)
                return sanitized_dict

        for k, v in value.items():
            key_str = str(k)
            if (
                is_sensitive_key(key_str)
                or is_sensitive_header(key_str)
                or key_str.lower()
                in (
                    "secrets",
                    "credentials",
                    "tokens",
                    "passwords",
                    "keys",
                )
            ):
                sanitized_dict[k] = redact_deep(v)
            elif key_str.lower() == "headers" and isinstance(v, dict):
                sanitized_dict[k] = {
                    hk: (
                        _REDACTED
                        if is_sensitive_header(str(hk))
                        else sanitize_metadata(hv, secrets=secrets)
                    )
                    for hk, hv in v.items()
                }
            else:
                sanitized_dict[k] = sanitize_metadata(v, secrets=secrets)
        return sanitized_dict
    if isinstance(value, list):
        return [sanitize_metadata(item, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_metadata(item, secrets=secrets) for item in value)
    if isinstance(value, set):
        return {sanitize_metadata(item, secrets=secrets) for item in value}
    if isinstance(value, str):
        return sanitize_text(value, secrets=secrets)
    return value


def contains_redacted(value: Any) -> bool:
    """Returns True if the value or any nested element contains the redaction marker."""
    if isinstance(value, dict):
        return any(
            contains_redacted(k) or contains_redacted(v) for k, v in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(contains_redacted(item) for item in value)
    return isinstance(value, str) and _REDACTED in value
