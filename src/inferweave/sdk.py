"""Main InferWeave SDK client entrypoint."""

from datetime import UTC, datetime
from typing import Any

from inferweave.core.exceptions import HealthcheckError, HealthcheckTimeoutError
from inferweave.domain.healthcheck import ProbeResult
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState
from inferweave.models.profile import ModelProfile
from inferweave.providers.router import ProviderRouter
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template
from inferweave.services.healthcheck_service import HealthcheckService


class InferWeave:
    """Unified AI inference deployment orchestrator across GPU cloud providers."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        router: ProviderRouter | None = None,
        healthcheck_service: HealthcheckService | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.router = router or ProviderRouter()
        self.healthcheck_service = healthcheck_service or HealthcheckService()
        self._active_deployments: dict[str, Deployment] = {}

    async def deploy(
        self,
        model: str,
        provider: str = "auto",
        strategy: str = "cheapest",
        gpu_type: str | None = None,
        num_gpus: int | None = None,
        env: dict[str, str] | None = None,
        autostop_mins: int = 30,
        custom_args: dict[str, Any] | None = None,
        dry_run: bool = False,
        wait_for_ready: bool = True,
    ) -> Deployment:
        """Deploys a model to the requested compute provider.

        Args:
            model: Model identifier from registry (e.g. 'fish-s2-pro', 'meta-llama/Meta-Llama-3-8B-Instruct')
            provider: Target provider ('runpod', 'aws', 'modal', or 'auto')
            strategy: Routing strategy when provider='auto' ('cheapest', 'free_first')
            gpu_type: Optional GPU override ('A100', 'H100', 'L4')
            num_gpus: Optional GPU count override
            env: Custom environment variables
            autostop_mins: Idle shutdown timer in minutes
            custom_args: Provider-specific extra configurations
            dry_run: When True, constructs the configuration without launching live cloud resources
            wait_for_ready: When True, actively polls endpoint until readiness healthcheck passes

        Returns:
            Deployment: Live deployment handle with lifecycle controls and endpoint details.
        """
        request = DeploymentRequest(
            model=model,
            provider=provider,
            strategy=strategy,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            env=env or {},
            autostop_mins=autostop_mins,
            custom_args=custom_args or {},
            dry_run=dry_run,
            wait_for_ready=wait_for_ready,
        )

        # 1. Resolve model profile
        profile = self.registry.get(model)

        # 2. Pre-flight hardware validation if explicit GPU is requested
        if request.gpu_type:
            self.router.hardware_service.validate_deployment_hardware(profile, request)

        # 3. Resolve target compute provider (SkyPilot clouds or Modal) with VRAM-aware routing
        compute_provider = self.router.resolve(
            provider_name=request.provider,
            profile=profile,
            strategy=request.strategy,
            request=request,
        )

        # 4. Render runtime specification
        runtime_template = get_runtime_template(profile.default_runtime)
        runtime_spec = runtime_template.render(profile, request)

        # 5. Provision and launch deployment
        deployment = await compute_provider.deploy(
            request=request,
            profile=profile,
            runtime=runtime_spec,
        )

        # 6. Wire healthcheck and readiness polling closures to the Deployment
        async def _check_health() -> ProbeResult:
            if not deployment.endpoint_url:
                raise HealthcheckError(
                    f"Deployment '{deployment.id}' has no endpoint URL available.",
                    deployment_id=deployment.id,
                )
            return await self.healthcheck_service.check_health(
                endpoint_url=deployment.endpoint_url,
                config=profile.healthcheck,
            )

        async def _wait_for_ready(timeout_secs: int | None = None) -> DeploymentStatus:
            try:
                await self.healthcheck_service.wait_for_ready(
                    endpoint_url=deployment.endpoint_url,
                    config=profile.healthcheck,
                    deployment_id=deployment.id,
                    timeout_override=float(timeout_secs) if timeout_secs is not None else None,
                )
                deployment._status.state = DeploymentState.HEALTHY
                deployment._status.ready_at = datetime.now(UTC)
            except HealthcheckTimeoutError as err:
                deployment._status.state = DeploymentState.FAILED
                deployment._status.error_message = str(err)
                raise
            return deployment._status

        deployment._healthcheck_fn = _check_health
        deployment._wait_ready_fn = _wait_for_ready

        # 7. Track active deployment
        self._active_deployments[deployment.id] = deployment

        # 8. Actively poll for readiness if wait_for_ready is enabled
        if request.wait_for_ready and not request.dry_run and profile.healthcheck.enabled:
            await _wait_for_ready()

        return deployment

    def register_model(self, profile: ModelProfile) -> None:
        """Registers a custom model profile in the registry."""
        self.registry.register(profile)

    def list_deployments(self) -> list[Deployment]:
        """Returns all actively tracked deployments in the local session."""
        return list(self._active_deployments.values())
