"""Main InferWeave SDK client entrypoint."""

from typing import Any

from inferweave.models.deployment import Deployment, DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.providers.router import ProviderRouter
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template


class InferWeave:
    """Unified AI inference deployment orchestrator across GPU cloud providers."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        router: ProviderRouter | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.router = router or ProviderRouter()
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

        # 5. Track active deployment
        self._active_deployments[deployment.id] = deployment
        return deployment

    def register_model(self, profile: ModelProfile) -> None:
        """Registers a custom model profile in the registry."""
        self.registry.register(profile)

    def list_deployments(self) -> list[Deployment]:
        """Returns all actively tracked deployments in the local session."""
        return list(self._active_deployments.values())
