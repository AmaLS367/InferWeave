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


class InsufficientVramError(InferWeaveError):
    """Raised when requested or selected hardware does not have enough VRAM to run the model."""

    def __init__(
        self,
        model_id: str,
        required_vram_gb: float,
        provided_vram_gb: float,
        gpu_type: str,
        gpu_count: int = 1,
        suggested_gpus: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.required_vram_gb = required_vram_gb
        self.provided_vram_gb = provided_vram_gb
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.deficit_gb = max(0.0, required_vram_gb - provided_vram_gb)
        self.suggested_gpus = suggested_gpus or []

        alt_text = (
            f" Suggested alternative GPUs: {', '.join(self.suggested_gpus)}."
            if self.suggested_gpus
            else ""
        )
        count_text = (
            f" ({gpu_count}x {gpu_type})" if gpu_count > 1 else f" ({gpu_type})"
        )

        super().__init__(
            f"Model '{model_id}' requires at least {required_vram_gb:.1f}GB VRAM, but requested hardware{count_text} "
            f"provides only {provided_vram_gb:.1f}GB (deficit: {self.deficit_gb:.1f}GB).{alt_text}"
        )


class UnknownGpuError(InferWeaveError):
    """Raised when an unknown GPU specification is encountered and strict validation is required."""

    def __init__(self, gpu_name: str) -> None:
        super().__init__(
            f"Unknown GPU specification: '{gpu_name}'. Could not determine VRAM capacity."
        )
        self.gpu_name = gpu_name
