"""Domain entity representing a deployment record and its persistent lifecycle state."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState


class DeploymentRecord(BaseModel):
    """Domain model tracking deployment identity, metadata, and operational lifecycle state."""

    id: str = Field(..., description="Unique deployment identifier")
    model: str = Field(..., description="Model identifier or HuggingFace ID")
    provider: str = Field(..., description="Compute infrastructure provider name")
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
                lower_k = key.lower()
                if any(term in lower_k for term in ("token", "secret", "password", "api_key")):
                    if isinstance(extra_args[key], str):
                        extra_args[key] = "[REDACTED]"
                    elif isinstance(extra_args[key], dict):
                        extra_args[key] = {k: "[REDACTED]" for k in extra_args[key]}

            sanitized.options.provider.extra_provider_args = extra_args

        # Sanitize runtime extra_env
        if sanitized.options.runtime and sanitized.options.runtime.extra_env:
            env_copy = dict(sanitized.options.runtime.extra_env)
            for k in list(env_copy.keys()):
                lower_k = k.lower()
                if any(term in lower_k for term in ("token", "secret", "password", "key")):
                    env_copy[k] = "[REDACTED]"
            sanitized.options.runtime.extra_env = env_copy

        return sanitized
