"""Deployment request, status, and live deployment handle definitions."""

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState


class DeploymentRequest(BaseModel):
    """User request specification for deploying a model."""

    model: str = Field(..., description="Target model name or HuggingFace ID")
    provider: str = Field(
        default="auto",
        description="Compute provider name (e.g. 'runpod', 'aws', 'modal')",
    )
    strategy: str | None = Field(
        default="cheapest", description="Routing strategy when provider='auto'"
    )
    gpu_type: str | None = Field(
        default=None, description="Explicit GPU override, e.g. 'A100'"
    )
    num_gpus: int | None = Field(
        default=None, description="Explicit GPU count override"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Custom environment variable overrides"
    )
    autostop_mins: int | None = Field(
        default=30, description="Auto-terminate after idle minutes"
    )
    custom_args: dict[str, Any] = Field(
        default_factory=dict, description="Provider or runtime specific arguments"
    )
    options: DeploymentOptions | None = Field(
        default=None,
        description="Structured domain options for runtime, provider, and lifecycle autostop",
    )
    dry_run: bool = Field(
        default=False,
        description="When True, validates and builds deployment specs without provisioning remote cloud resources",
    )
    wait_for_ready: bool = Field(
        default=True,
        description="When True, blocks until endpoint healthcheck readiness probe passes",
    )

    @model_validator(mode="after")
    def _ensure_options(self) -> "DeploymentRequest":
        if self.options is None:
            self.options = DeploymentOptions.from_custom_args(
                custom_args=self.custom_args,
                autostop_mins=self.autostop_mins,
            )
        return self


class DeploymentStatus(BaseModel):
    """Snapshot of a deployment's current operational state."""

    id: str = Field(..., description="Unique deployment identifier")
    model: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider hosting the deployment")
    state: DeploymentState = Field(default=DeploymentState.PENDING)
    endpoint_url: str | None = Field(
        default=None, description="Live HTTP/HTTPS inference URL"
    )
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ready_at: datetime | None = Field(default=None)


class Deployment:
    """Active deployment handle providing interaction and lifecycle management."""

    def __init__(
        self,
        status: DeploymentStatus,
        stop_fn: Callable[..., Coroutine[Any, Any, None]] | None = None,
        refresh_fn: Callable[[], Coroutine[Any, Any, DeploymentStatus]] | None = None,
        healthcheck_fn: Callable[[], Coroutine[Any, Any, Any]] | None = None,
        wait_ready_fn: Callable[[int | None], Coroutine[Any, Any, DeploymentStatus]]
        | None = None,
        autostop_mins: int | None = 30,
        record_activity_fn: Callable[[], None] | None = None,
        is_idle_fn: Callable[[], bool] | None = None,
        last_activity_fn: Callable[[], datetime | None] | None = None,
    ) -> None:
        self._status = status
        self._stop_fn = stop_fn
        self._refresh_fn = refresh_fn
        self._healthcheck_fn = healthcheck_fn
        self._wait_ready_fn = wait_ready_fn
        self._autostop_mins = autostop_mins
        self._record_activity_fn = record_activity_fn
        self._is_idle_fn = is_idle_fn
        self._last_activity_fn = last_activity_fn

    @property
    def id(self) -> str:
        return self._status.id

    @property
    def model(self) -> str:
        return self._status.model

    @property
    def provider(self) -> str:
        return self._status.provider

    @property
    def state(self) -> DeploymentState:
        return self._status.state

    @property
    def endpoint_url(self) -> str | None:
        return self._status.endpoint_url

    @property
    def is_healthy(self) -> bool:
        return self._status.state == DeploymentState.HEALTHY

    @property
    def status(self) -> DeploymentStatus:
        return self._status

    async def stop(self, action: Any | None = None) -> None:
        """Terminates or shuts down this deployment."""
        if self._stop_fn:
            from inferweave.domain.lifecycle import AutostopAction

            act = action or AutostopAction.STOP
            if isinstance(act, str):
                act = AutostopAction(act.lower())
            await self._stop_fn(action=act)
            self._status.state = DeploymentState.STOPPED

    async def refresh(self) -> DeploymentStatus:
        """Refreshes and returns the latest deployment status."""
        if self._refresh_fn:
            current_model = self._status.model
            new_status = await self._refresh_fn()
            if new_status.model == "unknown" and current_model != "unknown":
                new_status.model = current_model
            self._status = new_status
        return self._status

    async def check_health(self) -> Any:
        """Executes an immediate health probe against this deployment endpoint."""
        if self._healthcheck_fn:
            return await self._healthcheck_fn()
        raise RuntimeError(
            "No healthcheck probe function configured for this deployment."
        )

    @property
    def autostop_mins(self) -> int | None:
        """Idle timeout limit in minutes before autostop is triggered."""
        return self._autostop_mins

    @property
    def last_activity_at(self) -> datetime | None:
        """Timestamp of the most recent recorded activity or probe."""
        if self._last_activity_fn:
            return self._last_activity_fn()
        return None

    def record_activity(self) -> None:
        """Signals activity or incoming requests on this deployment, resetting idle timers."""
        if self._record_activity_fn:
            self._record_activity_fn()

    def is_idle(self) -> bool:
        """Returns True if the deployment has been inactive longer than its configured autostop threshold."""
        if self._is_idle_fn:
            return self._is_idle_fn()
        return False

    async def wait_for_ready(
        self, timeout_seconds: int | None = None
    ) -> DeploymentStatus:
        """Blocks until the deployment passes its readiness healthcheck."""
        if self._wait_ready_fn:
            self._status = await self._wait_ready_fn(timeout_seconds)
            return self._status
        return self._status

    def __repr__(self) -> str:
        return (
            f"<Deployment id='{self.id}' model='{self.model}' "
            f"provider='{self.provider}' state='{self.state}' endpoint='{self.endpoint_url}'>"
        )
