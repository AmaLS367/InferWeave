"""Provider router resolving provider names and selection strategies."""

from inferweave.core.exceptions import (
    NoFeasibleProviderError,
    ProviderNotFoundError,
)
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.models.routing import RoutingDecision
from inferweave.providers.base import ComputeProvider
from inferweave.providers.modal_provider import ModalProvider
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.services.hardware_service import HardwareValidationService
from inferweave.services.routing_service import SmartRoutingService


class ProviderRouter:
    """Manages available compute backends and handles provider resolution."""

    def __init__(
        self,
        routing_service: SmartRoutingService | None = None,
        hardware_service: HardwareValidationService | None = None,
    ) -> None:
        self._providers: dict[str, ComputeProvider] = {}
        self._routing_service = routing_service or SmartRoutingService()
        self._hardware_service = hardware_service or HardwareValidationService()
        self._last_decision: RoutingDecision | None = None
        self._register_default_providers()

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Returns the most recent RoutingDecision made during routing."""
        return self._last_decision

    @property
    def hardware_service(self) -> HardwareValidationService:
        """Returns the hardware validation service instance."""
        return self._hardware_service

    def register(self, provider: ComputeProvider) -> None:
        """Registers a compute provider."""
        self._providers[provider.name.lower()] = provider

    def get(self, provider_name: str) -> ComputeProvider:
        """Retrieves a provider by name. Raises ProviderNotFoundError if unknown."""
        name = provider_name.lower()
        if name not in self._providers:
            raise ProviderNotFoundError(provider_name)
        return self._providers[name]

    def resolve(
        self,
        provider_name: str,
        profile: ModelProfile,
        strategy: str | None = "cheapest",
        request: DeploymentRequest | None = None,
    ) -> ComputeProvider:
        """Resolves target provider, validates hardware VRAM, and selects optimal instance."""
        name = provider_name.lower()

        # Case 1: Auto routing across all providers
        if name == "auto":
            req = request or DeploymentRequest(
                model=profile.id,
                provider="auto",
                strategy=strategy or "cheapest",
                dry_run=True,
            )

            # Pre-flight check if user explicitly requested a GPU
            if req.gpu_type:
                self._hardware_service.validate_deployment_hardware(profile, req)

            decision = self._routing_service.resolve(profile=profile, request=req)
            self._last_decision = decision

            if request is not None and not request.gpu_type:
                request.gpu_type = decision.chosen_offer.gpu_spec.name

            return self.get(decision.chosen_provider)

        # Case 2: Specific provider requested (e.g. 'runpod', 'modal', 'aws')
        compute_provider = self.get(name)

        # If user explicitly specified a GPU, validate hardware VRAM compatibility
        if request is not None and request.gpu_type:
            self._hardware_service.validate_deployment_hardware(profile, request)
            return compute_provider

        # If user did NOT specify a GPU, run provider-scoped routing to pick best GPU for this provider
        req = request or DeploymentRequest(
            model=profile.id,
            provider=name,
            strategy=strategy or "cheapest",
            dry_run=True,
        )

        try:
            decision = self._routing_service.resolve(
                profile=profile,
                request=req,
                target_provider=name,
            )
            self._last_decision = decision
            if request is not None and not request.gpu_type:
                request.gpu_type = decision.chosen_offer.gpu_spec.name
        except NoFeasibleProviderError as err:
            # If the provider has catalog offers that were evaluated and rejected, re-raise
            if err.reasons:
                raise

            # If the provider has no catalog offers at all (e.g. uncataloged custom cloud),
            # check if recommended GPUs can provide a viable hardware configuration
            if profile.hardware.recommended_gpus:
                count = (
                    request.num_gpus or profile.hardware.gpu_count
                    if request
                    else profile.hardware.gpu_count
                )
                for rec_gpu in profile.hardware.recommended_gpus:
                    res = self._hardware_service.check_compatibility(
                        profile, rec_gpu, gpu_count=count
                    )
                    if res and res.is_compatible:
                        if request is not None and not request.gpu_type:
                            request.gpu_type = rec_gpu
                        return compute_provider
            raise

        return compute_provider

    def list_providers(self) -> list[str]:
        """Returns list of registered provider names."""
        return list(self._providers.keys())

    def _register_default_providers(self) -> None:
        """Registers default SkyPilot clouds and Modal backend."""
        # SkyPilot-managed IaaS clouds
        skypilot_clouds = [
            "runpod",
            "aws",
            "gcp",
            "azure",
            "lambda",
            "nebius",
            "vast",
            "oci",
            "kubernetes",
            "fluidstack",
        ]
        for cloud in skypilot_clouds:
            self.register(SkyPilotProvider(cloud_name=cloud))

        # Independent serverless provider
        self.register(ModalProvider())
