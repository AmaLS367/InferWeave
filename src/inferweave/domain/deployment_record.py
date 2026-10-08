"""Domain entity representing a deployment record and its persistent lifecycle state."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState, WorkloadType

_SENSITIVE_KEY_TERMS = (
    "token",
    "secret",
    "password",
    "api_key",
    "api-key",
    "apikey",
    "credential",
    "authorization",
    "bearer",
    "modal-key",
    "modal_key",
)
_REDACTED = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    lower_k = key.lower()
    return any(term in lower_k for term in _SENSITIVE_KEY_TERMS)


def _redact_value(value: object) -> object:
    """Redacts a value stored under a sensitive key, recursing into containers."""
    if isinstance(value, dict):
        return {k: _REDACTED for k in value}
    if isinstance(value, list | tuple):
        return [_REDACTED for _ in value]
    if isinstance(value, str):
        return _REDACTED
    return value


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

    def to_sanitized_record(self) -> "DeploymentRecord":
        """Returns a copy of the record with sensitive credentials and tokens redacted for disk storage."""
        sanitized = self.model_copy(deep=True)
        if not sanitized.options:
            return sanitized

        # Sanitize extra_provider_args
        if sanitized.options.provider and sanitized.options.provider.extra_provider_args:
            extra_args = dict(sanitized.options.provider.extra_provider_args)
            if "secrets" in extra_args:
                sec = extra_args["secrets"]
                if isinstance(sec, dict):
                    extra_args["secrets"] = {k: "[REDACTED]" for k in sec}
                elif isinstance(sec, list):
                    redacted_list = []
                    for item in sec:
                        if isinstance(item, str) and "=" in item:
                            k, _ = item.split("=", 1)
                            redacted_list.append(f"{k}=[REDACTED]")
                        else:
                            redacted_list.append(item)
                    extra_args["secrets"] = redacted_list
                elif isinstance(sec, str):
                    extra_args["secrets"] = "[REDACTED]"

            for key in list(extra_args.keys()):
                if key == "secrets":
                    continue  # already redacted above, keeping variable names
                if _is_sensitive_key(key):
                    extra_args[key] = _redact_value(extra_args[key])
                elif key.lower() == "headers" and isinstance(extra_args[key], dict):
                    extra_args[key] = {
                        hk: (_REDACTED if _is_sensitive_key(str(hk)) else hv)
                        for hk, hv in extra_args[key].items()
                    }

            sanitized.options.provider.extra_provider_args = extra_args

        # Sanitize runtime extra_env
        if sanitized.options.runtime and sanitized.options.runtime.extra_env:
            env_copy = dict(sanitized.options.runtime.extra_env)
            for k in list(env_copy.keys()):
                if _is_sensitive_key(k) or "key" in k.lower() or "auth" in k.lower():
                    env_copy[k] = _REDACTED
            sanitized.options.runtime.extra_env = env_copy

        return sanitized
