"""Domain models and rules for deployment lifecycle and autostop management."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


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


class LifecycleState(BaseModel):
    """Snapshot of a deployment's lifecycle and idle tracking state."""

    deployment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy: AutostopPolicy = Field(default_factory=AutostopPolicy)
    is_stopped: bool = Field(default=False)
    stopped_at: datetime | None = Field(default=None)


class DeploymentLifecycleEvaluator:
    """Pure domain service evaluating deployment idle state and autostop eligibility."""

    @staticmethod
    def calculate_idle_seconds(
        state: LifecycleState, now: datetime | None = None
    ) -> float:
        """Calculates elapsed seconds of inactivity since last recorded activity."""
        current_time = now or datetime.now(UTC)
        elapsed = (current_time - state.last_activity_at).total_seconds()
        return max(0.0, elapsed)

    @classmethod
    def is_idle(
        cls, state: LifecycleState, now: datetime | None = None
    ) -> bool:
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
