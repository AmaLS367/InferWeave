"""Provider router resolving provider names and selection strategies."""


from app.core.exceptions import ProviderNotFoundError
from app.models.profile import ModelProfile
from app.providers.base import ComputeProvider
from app.providers.modal_provider import ModalProvider
from app.providers.skypilot import SkyPilotProvider


class ProviderRouter:
    """Manages available compute backends and handles provider resolution."""

    def __init__(self) -> None:
        self._providers: dict[str, ComputeProvider] = {}
        self._register_default_providers()

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
    ) -> ComputeProvider:
        """Resolves target provider. If provider_name is 'auto', selects best fit according to strategy."""
        name = provider_name.lower()
        if name != "auto":
            return self.get(name)

        # Smart strategy selection for 'auto'
        # In early alpha: defaults to RunPod for IaaS or Modal for serverless workloads
        if strategy == "free_first" or strategy == "cheapest":
            return self.get("runpod")

        return self.get("runpod")

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
