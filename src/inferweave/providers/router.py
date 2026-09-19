"""Provider router resolving provider names and selection strategies."""

from inferweave.core.exceptions import ProviderNotFoundError
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.models.routing import RoutingDecision
from inferweave.providers.base import ComputeProvider
from inferweave.providers.modal_provider import ModalProvider
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.services.routing_service import SmartRoutingService


class ProviderRouter:
    """Manages available compute backends and handles provider resolution."""

    def __init__(self, routing_service: SmartRoutingService | None = None) -> None:
        self._providers: dict[str, ComputeProvider] = {}
        self._routing_service = routing_service or SmartRoutingService()
        self._last_decision: RoutingDecision | None = None
        self._register_default_providers()

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Returns the most recent RoutingDecision made during 'auto' resolution."""
        return self._last_decision

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
        """Resolves target provider. If provider_name is 'auto', selects best fit according to strategy."""
        name = provider_name.lower()
        if name != "auto":
            return self.get(name)

        req = request or DeploymentRequest(
            model=profile.id,
            provider="auto",
            strategy=strategy or "cheapest",
            dry_run=True,
        )

        decision = self._routing_service.resolve(profile=profile, request=req)
        self._last_decision = decision

        # Populate recommended GPU into request if user did not specify one
        if request is not None and not request.gpu_type:
            request.gpu_type = decision.chosen_offer.gpu_spec.name

        return self.get(decision.chosen_provider)


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
