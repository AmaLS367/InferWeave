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


class DeploymentNotFoundError(DeploymentError):
    """Raised when a specified deployment identifier is not found in storage or runtime tracking."""

    def __init__(self, deployment_id: str) -> None:
        super().__init__(
            f"Deployment '{deployment_id}' was not found in active tracking or persistent storage.",
            deployment_id=deployment_id,
        )


class HealthcheckError(DeploymentError):
    """Base exception for all healthcheck and readiness probing errors."""


class HealthcheckTimeoutError(HealthcheckError):
    """Raised when an endpoint fails to pass readiness healthchecks within the specified timeout."""

    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float,
        total_probes: int = 0,
        last_error: str | None = None,
        deployment_id: str | None = None,
    ) -> None:
        last_err_msg = f" Last error: {last_error}." if last_error else ""
        dep_msg = f" for deployment '{deployment_id}'" if deployment_id else ""
        msg = (
            f"Readiness probe{dep_msg} at '{endpoint_url}' timed out after {timeout_seconds:.1f}s "
            f"({total_probes} probes executed).{last_err_msg}"
        )
        super().__init__(msg, deployment_id=deployment_id)
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds
        self.total_probes = total_probes
        self.last_error = last_error


class HealthcheckFailedError(HealthcheckError):
    """Raised when an endpoint probe returns an unrecoverable failure status."""

    def __init__(
        self,
        endpoint_url: str,
        message: str,
        deployment_id: str | None = None,
    ) -> None:
        super().__init__(
            f"Readiness probe failed at '{endpoint_url}': {message}",
            deployment_id=deployment_id,
        )
        self.endpoint_url = endpoint_url


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
