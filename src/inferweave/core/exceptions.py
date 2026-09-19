"""InferWeave core exception hierarchy."""


class InferWeaveError(Exception):
    """Base exception for all InferWeave errors."""


class ModelNotFoundError(InferWeaveError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model '{model_id}' was not found in the InferWeave registry. "
            f"You can register it using weave.register_model(...)."
        )
        self.model_id = model_id


class ProviderNotFoundError(InferWeaveError):
    """Raised when an unrecognized or unregistered compute provider is requested."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(
            f"Compute provider '{provider_name}' is not supported or registered. "
            f"Available providers: runpod, aws, gcp, azure, lambda, nebius, modal, docker, auto."
        )
        self.provider_name = provider_name


class ProviderPlatformError(InferWeaveError):
    """Raised when a provider cannot run on the current host platform."""


class DeploymentError(InferWeaveError):
    """Raised when a deployment fails during provisioning, runtime startup, or healthcheck."""

    def __init__(self, message: str, deployment_id: str | None = None) -> None:
        super().__init__(message)
        self.deployment_id = deployment_id


class NoFeasibleProviderError(InferWeaveError):
    """Raised when no compute provider meets the model requirements or routing constraints."""

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []

