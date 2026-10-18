"""Domain entity representing a deployment record and its persistent lifecycle state."""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from inferweave.core.credentials import (
    _REDACTED,
    _SENSITIVE_KEY_TERMS,
    _is_sensitive_key,
    contains_redacted,
    is_sensitive_header,
    is_sensitive_key,
    known_secret_values,
    redact_deep,
    sanitize_args,
    sanitize_metadata,
    sanitize_text,
)
from inferweave.core.exceptions import ModelNotFoundError
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import ModelProfile
from inferweave.models.usage import ComputePricing, DeploymentUsage

__all__ = [
    "_REDACTED",
    "_SENSITIVE_KEY_TERMS",
    "DeploymentRecord",
    "_is_sensitive_key",
    "_redact_value",
    "is_sensitive_key",
    "recover_profile_from_snapshot",
]


def _redact_value(value: object) -> object:
    """Redacts a value stored under a sensitive key, recursing into containers."""
    return redact_deep(value)


def _redact_secrets_entry(sec: Any) -> Any:
    """Redacts every value under a secrets container, including opaque strings without '='."""
    if isinstance(sec, dict):
        return {k: _redact_secrets_entry(v) for k, v in sec.items()}
    if isinstance(sec, list):
        redacted_list: list[Any] = []
        for item in sec:
            if isinstance(item, str):
                if "=" in item:
                    k, _ = item.split("=", 1)
                    redacted_list.append(f"{k}={_REDACTED}")
                else:
                    redacted_list.append(_REDACTED)
            elif isinstance(item, (dict, list, tuple, set)):
                redacted_list.append(_redact_secrets_entry(item))
            else:
                redacted_list.append(_REDACTED)
        return redacted_list
    if isinstance(sec, tuple):
        return tuple(_redact_secrets_entry(list(sec)))
    if isinstance(sec, set):
        return set(_redact_secrets_entry(list(sec)))
    if isinstance(sec, str):
        if "=" in sec:
            k, _ = sec.split("=", 1)
            return f"{k}={_REDACTED}"
        return _REDACTED
    return _REDACTED


def _validate_redacted_profile_dict(value: dict[str, Any]) -> ModelProfile | None:
    """Validates a serialized ModelProfile dict whose sensitive env keys were redacted."""
    cleaned = dict(value)
    redacted_default_env: dict[str, str] = {}
    if isinstance(cleaned.get("default_env"), dict):
        new_default_env: dict[str, Any] = {}
        for k, v in cleaned["default_env"].items():
            if isinstance(k, str) and is_sensitive_key(k) and v == _REDACTED:
                redacted_default_env[k] = _REDACTED
            else:
                new_default_env[k] = v
        cleaned["default_env"] = new_default_env

    redacted_container_env: dict[str, str] = {}
    if isinstance(cleaned.get("container"), dict):
        container_copy = dict(cleaned["container"])
        if isinstance(container_copy.get("env"), dict):
            new_c_env: dict[str, Any] = {}
            for k, v in container_copy["env"].items():
                if isinstance(k, str) and is_sensitive_key(k) and v == _REDACTED:
                    redacted_container_env[k] = _REDACTED
                else:
                    new_c_env[k] = v
            container_copy["env"] = new_c_env
        cleaned["container"] = container_copy

    if not redacted_default_env and not redacted_container_env:
        return None

    try:
        prof = ModelProfile.model_validate(cleaned)
    except Exception:  # noqa: BLE001
        return None

    prof.default_env.update(redacted_default_env)
    if prof.container is not None and redacted_container_env:
        prof.container.env.update(redacted_container_env)
    return prof


def _coerce_profile_snapshot(
    value: Any,
) -> ModelProfile | dict[str, Any] | None:
    """Coerces raw snapshot data into a ModelProfile or an invalid-snapshot marker."""
    if value is None:
        return None
    if isinstance(value, ModelProfile):
        return value
    if isinstance(value, dict):
        if value.get("__invalid_snapshot__") is True:
            return {"__invalid_snapshot__": True}
        try:
            return ModelProfile.model_validate(value)
        except (ValidationError, ValueError, TypeError):
            redacted_prof = _validate_redacted_profile_dict(value)
            if redacted_prof is not None:
                return redacted_prof
            return {"__invalid_snapshot__": True}
    return {"__invalid_snapshot__": True}


AMBIENT_ACCOUNT = "ambient"
"""Owner of deployments created with a provider's native default credentials (and of records
written before accounts existed). Mirrors ``inferweave.accounts.AMBIENT_ACCOUNT_ID``."""


class ResourceRef(BaseModel):
    """Nonsecret identity of the remote resource backing a deployment.

    ``name`` is chosen by InferWeave before the resource is created, so an operation that timed
    out can always be reconciled by looking the name up under the owning account.
    """

    name: str = Field(..., description="Remote resource name (Modal app, Lightning deployment, cluster)")
    scope: str | None = Field(
        default=None, description="Provider scope: Lightning teamspace or Modal environment"
    )
    resource_id: str | None = Field(default=None, description="Provider-assigned resource id")
    owned: bool = Field(
        default=True,
        description="False until InferWeave knows the name did not belong to a pre-existing resource",
    )


class DeploymentRecord(BaseModel):
    """Domain model tracking deployment identity, metadata, and operational lifecycle state."""

    id: str = Field(..., description="Unique deployment identifier")
    model: str = Field(..., description="Model identifier or HuggingFace ID")
    provider: str = Field(..., description="Compute infrastructure provider name")
    usage: DeploymentUsage | None = None  # None means historical usage was not tracked.
    pricing: ComputePricing | None = None
    last_stop_action: AutostopAction | None = None
    resources_confirmed_absent_at: datetime | None = None
    account: str | None = Field(
        default=AMBIENT_ACCOUNT,
        description=(
            "Nonsecret id of the provider account that owns the resource. Every later operation "
            "uses exactly this account. None for dry runs (no remote resource)."
        ),
    )
    resource: ResourceRef | None = Field(
        default=None, description="Remote resource identity used for recovery and cleanup"
    )
    owner_fingerprint: str | None = Field(
        default=None, description="Nonsecret digest binding the original control-plane credentials."
    )
    creation_may_continue: bool = Field(
        default=False,
        description="An unacknowledged create may still complete remotely; absence is not conclusive.",
    )
    needs_reconciliation: bool = Field(
        default=False,
        description=(
            "True while the remote state is unknown (a create or delete whose outcome was not "
            "confirmed). Cleared by stop(), reconcile() or a successful provisioning."
        ),
    )
    state: DeploymentState = Field(
        default=DeploymentState.PENDING,
        description="Current operational lifecycle state",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Active HTTP/HTTPS inference URL if provisioned",
    )
    error_message: str | None = Field(
        default=None,
        description="Diagnostic or error details if deployment degraded or failed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    ready_at: datetime | None = Field(
        default=None,
        description="Timestamp when readiness probe first succeeded",
    )
    stopped_at: datetime | None = Field(
        default=None,
        description="Timestamp when deployment was stopped or destroyed",
    )
    options: DeploymentOptions = Field(
        default_factory=DeploymentOptions,
        description="Consolidated runtime, provider, and lifecycle options",
    )
    is_dry_run: bool = Field(
        default=False,
        description="True if this deployment is a simulated dry-run",
    )
    workload_type: WorkloadType | None = Field(
        default=None,
        description="Workload category of the deployed model (None for records written by older versions)",
    )
    last_activity_at: datetime | None = Field(
        default=None,
        description=(
            "Most recent recorded user activity (inference calls); persisted so the idle "
            "destroy timer survives process restarts. None for records written by older versions."
        ),
    )
    profile_snapshot: ModelProfile | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Sanitized snapshot of the deployed ModelProfile so custom models can be "
            "recovered across process restarts. None for records written by older versions."
        ),
    )

    @field_validator("profile_snapshot", mode="before")
    @classmethod
    def _validate_profile_snapshot(
        cls, value: Any
    ) -> ModelProfile | dict[str, Any] | None:
        return _coerce_profile_snapshot(value)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_lightning(cls, data: Any) -> Any:
        """Preserves the remote identity of Lightning deployments written by 0.2."""
        if (
            isinstance(data, dict)
            and data.get("provider") == "lightning"
            and data.get("resource") is None
            and isinstance(legacy := data.get("lightning"), dict)
        ):
            data = {
                **data,
                "resource": ResourceRef(
                    name=legacy["name"],
                    scope=legacy["teamspace"],
                    resource_id=legacy.get("resource_id"),
                    owned=legacy.get("owned", True),
                ),
            }
        return data

    def mark_healthy(
        self,
        endpoint_url: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Transitions deployment state to HEALTHY and marks ready timestamp."""
        self.state = DeploymentState.HEALTHY
        if endpoint_url:
            self.endpoint_url = endpoint_url
        if not self.ready_at:
            self.ready_at = now or datetime.now(UTC)
        self.error_message = None

    def mark_stopped(self, now: datetime | None = None) -> None:
        """Transitions deployment state to STOPPED and records stopped timestamp."""
        self.state = DeploymentState.STOPPED
        self.stopped_at = now or datetime.now(UTC)

    def mark_failed(
        self,
        error_message: str,
        now: datetime | None = None,
    ) -> None:
        """Transitions deployment state to FAILED with descriptive error information."""
        self.state = DeploymentState.FAILED
        self.error_message = error_message
        if not self.stopped_at:
            self.stopped_at = now or datetime.now(UTC)

    def mark_degraded(self, error_message: str | None = None) -> None:
        """Transitions deployment state to DEGRADED."""
        self.state = DeploymentState.DEGRADED
        if error_message:
            self.error_message = error_message

    def is_active(self) -> bool:
        """Returns True if the deployment is actively running or provisioning."""
        return self.state in {
            DeploymentState.PENDING,
            DeploymentState.PROVISIONING,
            DeploymentState.STARTING,
            DeploymentState.HEALTHY,
            DeploymentState.DEGRADED,
        }

    def is_attachable(self) -> bool:
        """Returns True if a live handle can be rebuilt for this deployment.

        Stopped and failed deployments are terminal and dry runs have no real endpoint.
        UNHEALTHY deployments stay attachable: a Modal app scaled to zero fails probes
        until its container wakes up, yet the deployment is very much alive.
        """
        return not self.is_dry_run and self.state not in {
            DeploymentState.STOPPED,
            DeploymentState.FAILED,
        }

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        sanitized = self.to_sanitized_record()
        return [
            (
                k,
                (
                    getattr(sanitized, k)
                    if k is not None and hasattr(sanitized, k)
                    else v
                ),
            )
            for k, v in super().__repr_args__()
        ]

    def to_sanitized_record(
        self, secrets: Iterable[str] = ()
    ) -> "DeploymentRecord":
        """Returns a copy of the record with sensitive credentials and tokens redacted for disk storage."""
        secret_tuple = known_secret_values(secrets)
        sanitized = self.model_copy(deep=True)

        if sanitized.endpoint_url:
            sanitized.endpoint_url = sanitize_text(sanitized.endpoint_url, secret_tuple)
        if sanitized.error_message:
            sanitized.error_message = sanitize_text(sanitized.error_message, secret_tuple)
        if sanitized.model:
            sanitized.model = sanitize_text(sanitized.model, secret_tuple)
        if sanitized.provider:
            sanitized.provider = sanitize_text(sanitized.provider, secret_tuple)
        if sanitized.pricing:
            sanitized.pricing = sanitized.pricing.model_copy(update={
                key: sanitize_text(value, secret_tuple)
                for key, value in sanitized.pricing.model_dump().items()
                if key in {"provider", "gpu_type", "instance_type", "region"} and isinstance(value, str)
            })
        if sanitized.resource:
            if sanitized.resource.name:
                sanitized.resource.name = sanitize_text(
                    sanitized.resource.name, secret_tuple
                )
            if sanitized.resource.scope:
                sanitized.resource.scope = sanitize_text(
                    sanitized.resource.scope, secret_tuple
                )
            if sanitized.resource.resource_id:
                sanitized.resource.resource_id = sanitize_text(
                    sanitized.resource.resource_id, secret_tuple
                )

        if sanitized.profile_snapshot is not None:
            coerced = _coerce_profile_snapshot(sanitized.profile_snapshot)
            if isinstance(coerced, ModelProfile):
                try:
                    sanitized.profile_snapshot = coerced.to_sanitized(secret_tuple)
                except ValueError:
                    sanitized.profile_snapshot = {"__invalid_snapshot__": True}
            else:
                sanitized.profile_snapshot = {"__invalid_snapshot__": True}

        if not sanitized.options:
            return sanitized

        # Sanitize extra_provider_args
        if sanitized.options.provider and sanitized.options.provider.extra_provider_args:
            extra_args = dict(sanitized.options.provider.extra_provider_args)
            if "secrets" in extra_args:
                extra_args["secrets"] = _redact_secrets_entry(extra_args["secrets"])

            for key in list(extra_args.keys()):
                if key == "secrets":
                    continue  # already redacted above, keeping variable names
                if _is_sensitive_key(key):
                    extra_args[key] = _redact_value(extra_args[key])
                elif key.lower() == "headers" and isinstance(extra_args[key], dict):
                    extra_args[key] = {
                        hk: (
                            _REDACTED
                            if is_sensitive_header(str(hk))
                            else (
                                sanitize_text(hv, secret_tuple)
                                if isinstance(hv, str)
                                else sanitize_metadata(hv, secret_tuple)
                            )
                        )
                        for hk, hv in extra_args[key].items()
                    }
                else:
                    extra_args[key] = sanitize_metadata(
                        extra_args[key], secret_tuple
                    )

            sanitized.options.provider.extra_provider_args = extra_args

        # Sanitize runtime options
        if sanitized.options.runtime:
            if sanitized.options.runtime.extra_env:
                env_copy = dict(sanitized.options.runtime.extra_env)
                for k, v in list(env_copy.items()):
                    if is_sensitive_key(k):
                        env_copy[k] = _REDACTED
                    elif isinstance(v, str):
                        env_copy[k] = sanitize_text(v, secret_tuple)
                    else:
                        env_copy[k] = sanitize_metadata(v, secret_tuple)
                sanitized.options.runtime.extra_env = env_copy
            if sanitized.options.runtime.extra_cli_args:
                sanitized.options.runtime.extra_cli_args = sanitize_args(
                    sanitized.options.runtime.extra_cli_args, secret_tuple
                )
            if sanitized.options.runtime.engine_args:
                sanitized.options.runtime.engine_args = sanitize_metadata(
                    sanitized.options.runtime.engine_args, secret_tuple
                )

        return sanitized


def recover_profile_from_snapshot(
    record: DeploymentRecord, secrets: Iterable[str] = ()
) -> ModelProfile:
    """Validates and recovers a ModelProfile from a persisted DeploymentRecord snapshot.

    Raises ``ModelNotFoundError`` when the record has no snapshot, when the snapshot is
    structurally invalid or does not match the deployment record identity, or when
    sensitive values had to be redacted and cannot be safely recovered without secrets.
    """
    snapshot = record.profile_snapshot
    if snapshot is None:
        raise ModelNotFoundError(record.model)

    if isinstance(snapshot, dict):
        if snapshot.get("__invalid_snapshot__") is True:
            raise ModelNotFoundError(
                record.model, reason="persisted profile snapshot is invalid"
            )
        raw_data = snapshot
    elif isinstance(snapshot, ModelProfile):
        raw_data = snapshot.model_dump()
    else:
        raise ModelNotFoundError(
            record.model, reason="persisted profile snapshot is invalid"
        )

    snapshot_id = raw_data.get("id")
    if isinstance(snapshot_id, str) and snapshot_id != record.model:
        raise ModelNotFoundError(
            record.model,
            reason=(
                f"persisted profile snapshot id '{sanitize_text(snapshot_id)}' does not match "
                f"deployment model '{sanitize_text(record.model)}'"
            ),
        )

    if contains_redacted(raw_data):
        raise ModelNotFoundError(
            record.model,
            reason=(
                "persisted profile snapshot had sensitive values redacted and cannot "
                "be safely recovered without re-registering the model"
            ),
        )

    try:
        profile = ModelProfile.model_validate(raw_data)
    except Exception:  # noqa: BLE001
        raise ModelNotFoundError(
            record.model, reason="persisted profile snapshot is invalid"
        ) from None

    if profile.id != record.model:
        raise ModelNotFoundError(
            record.model,
            reason=(
                f"persisted profile snapshot id '{sanitize_text(profile.id)}' does not match "
                f"deployment model '{sanitize_text(record.model)}'"
            ),
        )

    if (
        record.workload_type is not None
        and profile.workload_type != record.workload_type
    ):
        raise ModelNotFoundError(
            record.model,
            reason="persisted profile snapshot workload_type does not match deployment record",
        )

    sanitized = profile.to_sanitized(secrets)
    if contains_redacted(sanitized.model_dump()) or (
        sanitized.model_dump() != profile.model_dump()
    ):
        raise ModelNotFoundError(
            record.model,
            reason=(
                "persisted profile snapshot contains sensitive values and cannot "
                "be safely recovered"
            ),
        )

    from inferweave.models.deployment import DeploymentRequest
    from inferweave.runtimes.templates import get_runtime_template

    try:
        template = get_runtime_template(profile.default_runtime)
        template.render(
            profile, DeploymentRequest(model=profile.id, provider=record.provider)
        )
    except Exception:  # noqa: BLE001
        raise ModelNotFoundError(
            record.model,
            reason="persisted profile snapshot failed runtime validation",
        ) from None

    return profile.model_copy(deep=True)
