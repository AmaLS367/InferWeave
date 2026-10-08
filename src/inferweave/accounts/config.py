"""Declarative, secret-free configuration of provider account pools.

Configuration names *where* secrets live (environment variable names, a CredWeave JSON
credentials file, or a custom CredWeave source); it never contains secret values itself, so it
is safe to commit or log. Example (YAML or JSON)::

    providers:
      modal:
        strategy: round_robin
        accounts:
          - id: modal-team-a
            env:
              token_id: MODAL_A_TOKEN_ID
              token_secret: MODAL_A_TOKEN_SECRET
              proxy_token_id: MODAL_A_PROXY_TOKEN_ID
              proxy_token_secret: MODAL_A_PROXY_TOKEN_SECRET
      runpod:
        strategy: weighted
        credentials_file: ~/.config/inferweave/runpod.json
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from credweave import CredentialSource, SelectionStrategy

from inferweave.accounts.schema import SCHEDULING_METADATA, schema_for
from inferweave.accounts.strategy import make_strategy
from inferweave.core.exceptions import AccountConfigurationError

ACCOUNTS_FILE_ENV = "INFERWEAVE_ACCOUNTS_FILE"
"""Environment variable naming the accounts configuration file used by default."""

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class AccountSpec:
    """One named account whose secrets are read from environment variables.

    Args:
        id: Stable, nonsecret account identifier. Deployments persist it to stay bound to the
            account that created them. Changed control keys cannot take over existing records.
        env: ``{secret field: environment variable name}`` for every secret field the provider
            requires (and optional fields you use).
        weight: Share for the ``weighted`` strategy.
        priority: Rank for the ``failover`` strategy (lower is preferred).
        max_concurrency: Maximum simultaneous provisioning operations on this account.
        metadata: Provider metadata, e.g. ``{"teamspace": "org/ts"}`` for Lightning or
            ``{"environment": "main"}`` for Modal.
    """

    id: str
    env: Mapping[str, str]
    weight: float | None = None
    priority: int | None = None
    max_concurrency: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def credweave_metadata(self) -> dict[str, Any]:
        merged = dict(self.metadata)
        for key, value in (
            ("weight", self.weight),
            ("priority", self.priority),
            ("max_concurrency", self.max_concurrency),
        ):
            if value is not None:
                merged[key] = value
        return merged


@dataclass(frozen=True)
class ProviderAccounts:
    """The account pool of one provider.

    Accounts may come from ``accounts`` (env var references), a CredWeave ``credentials_file``
    (JSON, hot-reloaded on change: add, rotate or remove accounts by atomically replacing the
    file) and custom CredWeave ``sources``; all of them are merged into one pool.

    Args:
        strategy: ``round_robin`` (default), ``weighted``, ``failover``, ``least_used``,
            ``least_recently_used``, ``random``, or any CredWeave ``SelectionStrategy``.
        cooldown_seconds: Cooldown after a transient failure (CredWeave ``default_cooldown``).
        max_consecutive_failures: Transient failures before an account is marked unhealthy.
        permission_cooldown_seconds: How long an account that was denied permission (HTTP 403)
            is parked before it is tried again. Its health is not degraded.
        quota_cooldown_seconds: Parking time after quota/billing exhaustion when the provider
            gives no reset hint.
    """

    accounts: Sequence[AccountSpec] = ()
    credentials_file: str | Path | None = None
    sources: Sequence[CredentialSource] = ()
    strategy: str | SelectionStrategy = "round_robin"
    cooldown_seconds: float = 60.0
    max_consecutive_failures: int = 3
    permission_cooldown_seconds: float = 900.0
    quota_cooldown_seconds: float = 3600.0


class AccountsConfig:
    """Account pools for every provider that should rotate between several accounts.

    Providers without a pool keep using their native default credentials (the ``ambient``
    account): Modal's ``~/.modal.toml``/``MODAL_TOKEN_*``, ``LIGHTNING_*`` variables, and the
    user's own SkyPilot setup for RunPod, Vast and other clouds.

    Args:
        providers: ``{provider name: ProviderAccounts}``.
        max_attempts: Upper bound on provisioning attempts per deploy across all accounts and
            providers (bounded failover; nothing retries forever).
        state_dir: Where per-account isolated provider homes are kept (SkyPilot state and
            Lightning sandboxes). Defaults to ``~/.inferweave/accounts``.
    """

    def __init__(
        self,
        providers: Mapping[str, ProviderAccounts] | None = None,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        state_dir: str | Path | None = None,
    ) -> None:
        if max_attempts < 1:
            raise AccountConfigurationError("max_attempts must be at least 1.")
        self.providers: dict[str, ProviderAccounts] = {}
        for name, pool in (providers or {}).items():
            provider = name.lower()
            schema_for(provider)
            if not isinstance(pool, ProviderAccounts):
                raise AccountConfigurationError(
                    f"providers['{name}'] must be a ProviderAccounts instance."
                )
            make_strategy(pool.strategy)  # validate the name early
            if not (pool.accounts or pool.credentials_file or pool.sources):
                raise AccountConfigurationError(
                    f"{provider}: configure at least one account, credentials_file or source."
                )
            seen: set[str] = set()
            for spec in pool.accounts:
                if spec.id in seen:
                    raise AccountConfigurationError(
                        f"{provider} account id '{spec.id}' is defined more than once."
                    )
                seen.add(spec.id)
                _validate_spec(provider, spec)
            self.providers[provider] = pool
        self.max_attempts = max_attempts
        self.state_dir = (
            Path(state_dir).expanduser()
            if state_dir is not None
            else Path.home() / ".inferweave" / "accounts"
        )

    def __repr__(self) -> str:
        return f"AccountsConfig(providers={sorted(self.providers)}, max_attempts={self.max_attempts})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path | None = None) -> "AccountsConfig":
        """Builds a configuration from the documented YAML/JSON structure."""
        if not isinstance(data, Mapping):
            raise AccountConfigurationError("Accounts configuration must be a mapping.")
        _reject_unknown("accounts configuration", data, {"providers", "max_attempts", "state_dir"})
        raw_providers = data.get("providers") or {}
        if not isinstance(raw_providers, Mapping):
            raise AccountConfigurationError("'providers' must be a mapping of provider names.")
        providers = {
            str(name): _provider_from_dict(str(name), body, base_dir)
            for name, body in raw_providers.items()
        }
        state_dir = data.get("state_dir")
        if state_dir is not None and base_dir is not None:
            state_dir = _resolve_path(str(state_dir), base_dir)
        return cls(
            providers,
            max_attempts=int(data.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
            state_dir=state_dir,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AccountsConfig":
        """Loads a YAML (``.yaml``/``.yml``) or JSON configuration file.

        Relative ``credentials_file`` paths are resolved against the file's directory.
        """
        file_path = Path(path).expanduser()
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as err:
            raise AccountConfigurationError(
                f"Cannot read accounts configuration '{file_path}': {err.strerror}."
            ) from None
        try:
            data = (
                json.loads(text)
                if file_path.suffix.lower() == ".json"
                else yaml.safe_load(text)
            )
        except (ValueError, yaml.YAMLError):
            # Parser messages can quote file content; keep them out of the error.
            raise AccountConfigurationError(
                f"Accounts configuration '{file_path}' is not valid YAML/JSON."
            ) from None
        return cls.from_dict(data or {}, base_dir=file_path.parent)

    @classmethod
    def from_env(cls) -> "AccountsConfig":
        """Loads ``$INFERWEAVE_ACCOUNTS_FILE`` when set, else an empty (all-ambient) config."""
        path = os.environ.get(ACCOUNTS_FILE_ENV)
        return cls.from_file(path) if path else cls()


def _validate_spec(provider: str, spec: AccountSpec) -> None:
    schema = schema_for(provider)
    if not isinstance(spec.id, str) or not spec.id.strip():
        raise AccountConfigurationError(f"{provider}: account id must be a non-empty string.")
    names = set(spec.env)
    schema.validate_secret_names(spec.id, names)
    for first, second in schema.paired:
        if (first in names) != (second in names):
            raise AccountConfigurationError(
                f"{provider} account '{spec.id}' must configure '{first}' and '{second}' together."
            )
    for secret_name, variable in spec.env.items():
        if not isinstance(variable, str) or not variable.strip() or "=" in variable:
            raise AccountConfigurationError(
                f"{provider} account '{spec.id}': env['{secret_name}'] must be an environment "
                "variable name (never the secret value itself)."
            )
    schema.validate_metadata(spec.id, spec.credweave_metadata())


def _reject_unknown(where: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise AccountConfigurationError(
            f"{where} has unsupported field(s): {', '.join(map(str, unknown))}."
        )


def _resolve_path(value: str, base_dir: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _provider_from_dict(name: str, body: Any, base_dir: Path | None) -> ProviderAccounts:
    where = f"providers.{name}"
    if not isinstance(body, Mapping):
        raise AccountConfigurationError(f"{where} must be a mapping.")
    _reject_unknown(
        where,
        body,
        {
            "accounts",
            "credentials_file",
            "strategy",
            "cooldown_seconds",
            "max_consecutive_failures",
            "permission_cooldown_seconds",
            "quota_cooldown_seconds",
        },
    )
    accounts = []
    for index, raw in enumerate(body.get("accounts") or []):
        item = f"{where}.accounts[{index}]"
        if not isinstance(raw, Mapping):
            raise AccountConfigurationError(f"{item} must be a mapping.")
        _reject_unknown(
            item,
            raw,
            {"id", "env", "weight", "priority", "max_concurrency", "metadata"}
            | set(schema_for(name).metadata),
        )
        env = raw.get("env")
        if not isinstance(env, Mapping):
            raise AccountConfigurationError(f"{item}.env must map secret fields to env var names.")
        metadata = dict(raw.get("metadata") or {})
        for key in schema_for(name).metadata:
            if key in raw:
                metadata[key] = raw[key]
        accounts.append(
            AccountSpec(
                id=str(raw.get("id") or ""),
                env={str(k): str(v) for k, v in env.items()},
                weight=raw.get("weight"),
                priority=raw.get("priority"),
                max_concurrency=raw.get("max_concurrency"),
                metadata=metadata,
            )
        )
    credentials_file = body.get("credentials_file")
    defaults = ProviderAccounts()
    return ProviderAccounts(
        accounts=accounts,
        credentials_file=(
            _resolve_path(str(credentials_file), base_dir) if credentials_file else None
        ),
        strategy=str(body.get("strategy", defaults.strategy)),
        cooldown_seconds=float(body.get("cooldown_seconds", defaults.cooldown_seconds)),
        max_consecutive_failures=int(
            body.get("max_consecutive_failures", defaults.max_consecutive_failures)
        ),
        permission_cooldown_seconds=float(
            body.get("permission_cooldown_seconds", defaults.permission_cooldown_seconds)
        ),
        quota_cooldown_seconds=float(
            body.get("quota_cooldown_seconds", defaults.quota_cooldown_seconds)
        ),
    )


def _scheduling(
    weight: float | None, priority: int | None, max_concurrency: int | None
) -> dict[str, Any]:
    return {"weight": weight, "priority": priority, "max_concurrency": max_concurrency}


def modal_account(
    id: str,
    *,
    token_id_env: str,
    token_secret_env: str,
    proxy_token_id_env: str | None = None,
    proxy_token_secret_env: str | None = None,
    environment: str | None = None,
    weight: float | None = None,
    priority: int | None = None,
    max_concurrency: int | None = None,
) -> AccountSpec:
    """A Modal workspace account: API token pair plus its own endpoint proxy-auth token pair.

    The API token (``token_id``/``token_secret``) manages apps; the proxy token pair is a
    separate credential sent as ``Modal-Key``/``Modal-Secret`` to the protected endpoint.
    """
    env = {"token_id": token_id_env, "token_secret": token_secret_env}
    if bool(proxy_token_id_env) != bool(proxy_token_secret_env):
        raise AccountConfigurationError(
            f"modal account '{id}' must configure proxy_token_id_env and "
            "proxy_token_secret_env together."
        )
    if proxy_token_id_env and proxy_token_secret_env:
        env["proxy_token_id"] = proxy_token_id_env
        env["proxy_token_secret"] = proxy_token_secret_env
    metadata = {"environment": environment} if environment else {}
    spec = AccountSpec(
        id=id, env=env, metadata=metadata, **_scheduling(weight, priority, max_concurrency)
    )
    _validate_spec("modal", spec)
    return spec


def lightning_account(
    id: str,
    *,
    user_id_env: str,
    api_key_env: str,
    teamspace: str | None = None,
    weight: float | None = None,
    priority: int | None = None,
    max_concurrency: int | None = None,
) -> AccountSpec:
    """A Lightning AI user API key (the same key authenticates the ApiKeyAuth endpoint).

    ``teamspace`` (``owner/teamspace``) is the account's default; a per-deploy
    ``custom_args={'lightning': {'teamspace': ...}}`` overrides it.
    """
    metadata = {"teamspace": teamspace} if teamspace else {}
    spec = AccountSpec(
        id=id,
        env={"user_id": user_id_env, "api_key": api_key_env},
        metadata=metadata,
        **_scheduling(weight, priority, max_concurrency),
    )
    _validate_spec("lightning", spec)
    return spec


def _api_key_account(
    provider: str,
    id: str,
    api_key_env: str,
    weight: float | None,
    priority: int | None,
    max_concurrency: int | None,
) -> AccountSpec:
    spec = AccountSpec(
        id=id, env={"api_key": api_key_env}, **_scheduling(weight, priority, max_concurrency)
    )
    _validate_spec(provider, spec)
    return spec


def runpod_account(
    id: str,
    *,
    api_key_env: str,
    weight: float | None = None,
    priority: int | None = None,
    max_concurrency: int | None = None,
) -> AccountSpec:
    """A RunPod account API key."""
    return _api_key_account("runpod", id, api_key_env, weight, priority, max_concurrency)


def vast_account(
    id: str,
    *,
    api_key_env: str,
    weight: float | None = None,
    priority: int | None = None,
    max_concurrency: int | None = None,
) -> AccountSpec:
    """A Vast.ai account API key."""
    return _api_key_account("vast", id, api_key_env, weight, priority, max_concurrency)


__all__ = [
    "ACCOUNTS_FILE_ENV",
    "SCHEDULING_METADATA",
    "AccountSpec",
    "AccountsConfig",
    "ProviderAccounts",
    "lightning_account",
    "modal_account",
    "runpod_account",
    "vast_account",
]
