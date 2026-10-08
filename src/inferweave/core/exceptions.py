"""InferWeave core exception hierarchy.

Messages never contain credential values: provider SDK error text is withheld or redacted before
it reaches an exception, and exceptions raised from SDK errors suppress the chained traceback.
"""

from inferweave.core.failures import FailureKind


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

    def __init__(
        self,
        provider_name: str,
        available_providers: list[str] | None = None,
    ) -> None:
        avail_str = (
            ", ".join(available_providers)
            if available_providers
            else "runpod, aws, gcp, azure, lambda, nebius, vast, oci, kubernetes, fluidstack, modal, auto"
        )
        super().__init__(
            f"Compute provider '{provider_name}' is not supported or registered. "
            f"Available providers: {avail_str}."
        )
        self.provider_name = provider_name


class ProviderAuthError(InferWeaveError):
    """Raised when required provider/endpoint credentials are missing or unusable."""


class ProviderPlatformError(InferWeaveError):
    """Raised when a provider cannot run on the current host platform."""


class AccountConfigurationError(InferWeaveError):
    """Raised when provider account configuration is invalid (never includes secret values)."""


class AccountUnavailableError(ProviderAuthError):
    """Raised when the account that owns a deployment is no longer configured.

    InferWeave never falls back to another account for an existing deployment: restore the
    account (same id) in the configuration, or clean the resource up in the provider console.
    """

    def __init__(
        self, provider: str, account_id: str, deployment_id: str | None = None
    ) -> None:
        target = f" (deployment '{deployment_id}')" if deployment_id else ""
        super().__init__(
            f"Account '{account_id}' of provider '{provider}'{target} is not configured or no "
            "longer provides credentials. InferWeave will not operate on this deployment with "
            "another account. The account may be revoked or its control-plane keys may have "
            "changed identity; restore its original keys or recover in the provider console."
        )
        self.provider = provider
        self.account_id = account_id
        self.deployment_id = deployment_id


class ProviderOperationError(InferWeaveError):
    """A classified provider control-plane failure.

    ``kind`` says why the operation failed; ``resource_may_exist`` is True when the request may
    have taken effect remotely (e.g. a timeout after a create call was sent).
    """

    def __init__(
        self,
        message: str,
        kind: FailureKind,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        resource_may_exist: bool | None = None,
        deployment_id: str | None = None,
        operation_may_continue: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.resource_may_exist = (
            not kind.is_definitive_rejection if resource_may_exist is None else resource_may_exist
        )
        self.deployment_id = deployment_id
        self.operation_may_continue = operation_may_continue


class NoAccountAvailableError(InferWeaveError):
    """Raised when no account of the requested provider(s) could provision the deployment.

    ``attempts`` lists ``(provider, account_id, failure_kind)`` for every attempt made;
    ``retry_after`` is the shortest known cooldown, when one is known.
    """

    def __init__(
        self,
        message: str,
        attempts: list[tuple[str, str, str]] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])
        self.retry_after = retry_after


class DeploymentError(InferWeaveError):
    """Raised when a deployment fails during provisioning, runtime startup, or healthcheck."""

    def __init__(self, message: str, deployment_id: str | None = None) -> None:
        super().__init__(message)
        self.deployment_id = deployment_id


class ProvisioningUncertainError(DeploymentError):
    """Raised when InferWeave cannot tell whether a paid resource exists remotely.

    No other account or provider is tried, so a possibly running resource is never duplicated.
    The deployment record keeps ``needs_reconciliation=True`` and its owning account; run
    ``InferWeave.reconcile()`` (or ``stop(deployment_id)``) once the provider is reachable.
    """

    def __init__(self, deployment_id: str, provider: str, account_id: str) -> None:
        super().__init__(
            f"Provisioning of deployment '{deployment_id}' on {provider} account '{account_id}' "
            "ended in an unknown state; a billed resource may exist. No retry was made. Run "
            "InferWeave.reconcile() or stop() for this deployment ID to clean it up.",
            deployment_id=deployment_id,
        )
        self.provider = provider
        self.account_id = account_id


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


class DeploymentNotActiveError(DeploymentError):
    """Raised when a deployment cannot be attached because it is stopped, failed, or a dry run."""

    def __init__(self, deployment_id: str, reason: str) -> None:
        super().__init__(
            f"Deployment '{deployment_id}' cannot be attached: {reason}",
            deployment_id=deployment_id,
        )
        self.reason = reason


class AmbiguousDeploymentError(DeploymentError):
    """Raised when a ``find()`` query matches more than one active deployment."""

    def __init__(
        self, model: str | None, provider: str | None, candidate_ids: list[str]
    ) -> None:
        criteria = ", ".join(
            f"{k}='{v}'" for k, v in (("model", model), ("provider", provider)) if v
        )
        super().__init__(
            f"Found {len(candidate_ids)} active deployments matching {criteria or 'the query'}: "
            f"{', '.join(candidate_ids)}. Use attach(deployment_id) to pick one explicitly."
        )
        self.model = model
        self.provider = provider
        self.candidate_ids = list(candidate_ids)


class InferenceError(DeploymentError):
    """Base exception for failures while invoking a deployment's inference API.

    Messages and attributes never contain credentials: ``response_body`` is truncated and has
    known secret values redacted before it is stored.
    """

    def __init__(
        self,
        message: str,
        deployment_id: str | None = None,
        status_code: int | None = None,
        endpoint: str | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message, deployment_id=deployment_id)
        self.status_code = status_code
        self.endpoint = endpoint
        self.response_body = response_body


class EndpointNotReadyError(InferenceError):
    """Raised when the endpoint is missing, stopped, or still unavailable after bounded retries."""


class InferenceTimeoutError(InferenceError):
    """Raised when an inference request exceeds its configured timeout."""


class UnsupportedWorkloadError(InferenceError):
    """Raised when an inference method does not match the deployment's workload type."""

    def __init__(
        self,
        message: str,
        deployment_id: str | None = None,
        workload_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message, deployment_id=deployment_id)
        self.workload_type = workload_type
        self.operation = operation


class InvalidInferenceResponseError(InferenceError):
    """Raised when the endpoint returns a response that violates the expected API contract."""
