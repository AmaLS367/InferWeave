"""Domain models and rules for deployment lifecycle and autostop management."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from inferweave.models.enums import DeploymentState


class AutostopAction(str, Enum):
    """Action taken when deployment idle threshold is exceeded."""

    STOP = "stop"
    DOWN = "down"


class AutostopPolicy(BaseModel):
    """Inactivity-based shutdown policy configuration."""

    idle_minutes: int | None = Field(
        default=30,
        ge=1,
        description="Minutes of inactivity before triggering autostop. None disables timer.",
    )
    action: AutostopAction = Field(
        default=AutostopAction.STOP,
        description="Action to execute upon reaching idle threshold (stop or down)",
    )
    enabled: bool = Field(
        default=True,
        description="Master switch enabling or disabling idle autostop evaluations",
    )

    @field_validator("idle_minutes", mode="before")
    @classmethod
    def _validate_idle_minutes(cls, v: Any) -> Any:
        if v == 0 or v == "0":
            return None
        return v

    @model_validator(mode="after")
    def _sync_enabled(self) -> "AutostopPolicy":
        if self.idle_minutes is None:
            self.enabled = False
        return self


class LifecycleState(BaseModel):
    """Snapshot of a deployment's lifecycle and idle tracking state."""

    deployment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy: AutostopPolicy = Field(default_factory=AutostopPolicy)
    is_stopped: bool = Field(default=False)
    stopped_at: datetime | None = Field(default=None)


class DeploymentLifecycleEvaluator:
    """Pure domain service evaluating deployment idle state, autostop eligibility, and state reconciliation."""

    @staticmethod
    def calculate_idle_seconds(
        state: LifecycleState, now: datetime | None = None
    ) -> float:
        """Calculates elapsed seconds of inactivity since last recorded activity."""
        current_time = now or datetime.now(UTC)
        elapsed = (current_time - state.last_activity_at).total_seconds()
        return max(0.0, elapsed)

    @classmethod
    def is_idle(cls, state: LifecycleState, now: datetime | None = None) -> bool:
        """Determines if the deployment has been inactive longer than its policy threshold."""
        if not state.policy.enabled or state.policy.idle_minutes is None:
            return False
        idle_secs = cls.calculate_idle_seconds(state, now)
        threshold_secs = float(state.policy.idle_minutes * 60)
        return idle_secs >= threshold_secs

    @classmethod
    def should_autostop(
        cls, state: LifecycleState, now: datetime | None = None
    ) -> bool:
        """Returns True if the deployment is active and eligible for autostop."""
        if state.is_stopped or not state.policy.enabled:
            return False
        return cls.is_idle(state, now)

    @classmethod
    def reconcile_state(
        cls,
        infra_state: DeploymentState,
        probe_is_healthy: bool | None = None,
        is_stopped: bool = False,
        current_state: DeploymentState | None = None,
    ) -> DeploymentState:
        """Reconciles raw compute infrastructure state with application-level healthcheck probe results.

        Prevents false-positive 'HEALTHY' reports when cloud infrastructure is UP but the model server
        inside the container is still provisioning, crashed, or failing readiness checks.
        """
        if is_stopped or infra_state == DeploymentState.STOPPED:
            return DeploymentState.STOPPED

        if infra_state == DeploymentState.FAILED:
            return DeploymentState.FAILED

        if infra_state in {
            DeploymentState.PENDING,
            DeploymentState.PROVISIONING,
            DeploymentState.STARTING,
        }:
            return infra_state

        # Infrastructure is reported UP / HEALTHY by the cloud provider
        if probe_is_healthy is True:
            return DeploymentState.HEALTHY
        if probe_is_healthy is False:
            # If it was previously provisioning or starting, it may still be warming up
            if current_state in {
                DeploymentState.PROVISIONING,
                DeploymentState.STARTING,
            }:
                return DeploymentState.PROVISIONING
            # Otherwise, the endpoint failed its health check
            return DeploymentState.UNHEALTHY

        # If probe was not executed or not configured, fall back to infra state
        return infra_state
