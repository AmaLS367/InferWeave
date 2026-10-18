"""Dedicated Modal serverless compute provider adapter.

Every Modal call runs under an explicit ``modal.Client.from_credentials(token_id, token_secret)``
built from the owning account, so concurrent deployments on different workspaces never share
credentials and the process environment is never touched. The ambient account passes no client,
letting Modal resolve ``MODAL_TOKEN_*`` / ``~/.modal.toml`` exactly as it does natively.
"""

import asyncio
import logging
import re
import threading
import uuid
from typing import Any

from inferweave.accounts.models import ProviderAccount
from inferweave.adapters.auth import default_endpoint_auth
from inferweave.core.credentials import is_sensitive_key
from inferweave.core.exceptions import (
    DeploymentError,
    ProviderAuthError,
    ProviderOperationError,
)
from inferweave.core.failures import FailureKind, retry_after_from_error
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.providers.base import ComputeProvider, ProvisionResult, ResourceStatus
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)

# Default Modal scale-to-zero window. Deliberately independent of the InferWeave destroy timer
# (``AutostopPolicy.idle_minutes``): scale-to-zero only frees GPU containers while the app stays
# deployed, whereas the destroy timer stops the whole app.
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 1800

_ERROR_KINDS = {
    "AuthError": FailureKind.AUTH,
    "PermissionDeniedError": FailureKind.PERMISSION,
    "ResourceExhaustedError": FailureKind.RATE_LIMIT,
    "InvalidError": FailureKind.INVALID_REQUEST,
    "NotFoundError": FailureKind.INVALID_REQUEST,
    "ImageBuildError": FailureKind.INVALID_REQUEST,
    "VersionError": FailureKind.INVALID_REQUEST,
}


class ModalConfigurationError(DeploymentError, ValueError):
    """Raised when a runtime or container specification is invalid for Modal."""


class ModalProvider(ComputeProvider):
    """Serverless GPU provider adapter using Modal.

    Since Modal is serverless and not covered by SkyPilot, this provider executes
    deployments directly via the Modal Python SDK with native container definition.
    """

    def __init__(self, endpoint_auth: EndpointAuthPort | None = None) -> None:
        self._endpoint_auth = endpoint_auth
        self._clients: dict[str, tuple[str | None, Any]] = {}
        self._clients_lock = threading.Lock()

    @property
    def endpoint_auth(self) -> EndpointAuthPort | None:
        """Credential resolver used for the deploy-time proxy-token check."""
        return self._endpoint_auth

    @endpoint_auth.setter
    def endpoint_auth(self, value: EndpointAuthPort | None) -> None:
        self._endpoint_auth = value

    @property
    def name(self) -> str:
        return "modal"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MODAL

    def _get_modal_module(self) -> Any:
        """Lazy loads the modal SDK module."""
        try:
            import modal

            return modal
        except ImportError as e:
            raise ImportError(
                "Modal SDK is not installed. Install it via: pip install 'inferweave[modal]'"
            ) from e

    @staticmethod
    def _validate_runtime(profile: ModelProfile, runtime: RuntimeSpec) -> list[str]:
        """Validates runtime image, port, Python injection, env vars, and structured argv for Modal."""
        if (
            not isinstance(runtime.docker_image, str)
            or not runtime.docker_image.strip()
            or any(c.isspace() for c in runtime.docker_image.strip())
        ):
            raise ModalConfigurationError(
                f"Modal deployments require a non-empty container image reference without whitespace, "
                f"got {runtime.docker_image!r}."
            )

        if runtime.add_python is not None:
            if not isinstance(runtime.add_python, str):
                raise ModalConfigurationError(
                    f"Modal runtime.add_python must be a string or None, got {type(runtime.add_python).__name__}."
                )
            cleaned_py = runtime.add_python.strip()
            if not cleaned_py or not re.fullmatch(r"^\d+(\.\d+)*$", cleaned_py):
                raise ModalConfigurationError(
                    f"Invalid Python version string {runtime.add_python!r} for Modal add_python; "
                    "expected format like '3.12'."
                )

        if (
            isinstance(runtime.port, bool)
            or not isinstance(runtime.port, int)
            or runtime.port < 1
            or runtime.port > 65535
        ):
            raise ModalConfigurationError(
                f"Modal runtime.port must be an integer from 1 to 65535, got {runtime.port!r}."
            )

        if runtime.healthcheck is not None and runtime.port != runtime.healthcheck.port:
            raise ModalConfigurationError(
                f"Inconsistent configuration: runtime.port ({runtime.port}) and "
                f"runtime.healthcheck.port ({runtime.healthcheck.port}) must match."
            )

        if (
            profile.container is not None
            and profile.healthcheck is not None
            and runtime.port != profile.healthcheck.port
        ):
            raise ModalConfigurationError(
                f"Inconsistent configuration: runtime.port ({runtime.port}) and "
                f"profile.healthcheck.port ({profile.healthcheck.port}) must match."
            )

        if (runtime.name == "container" or profile.container is not None) and runtime.env_vars:
            for k, v in runtime.env_vars.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ModalConfigurationError(
                        "Modal container environment variable keys and values must be strings."
                    )
                if is_sensitive_key(k):
                    raise ModalConfigurationError(
                        f"Platform credentials and sensitive keys are forbidden in Modal container env_vars: '{k}'"
                    )

        if not isinstance(runtime.run_args, (list, tuple)) or not runtime.run_args:
            raise ModalConfigurationError(
                "Modal deployments require non-empty command argv (runtime.run_args / container.args); "
                "Modal clears the container image entrypoint (ENTRYPOINT) for its web_server harness "
                "and cannot execute an empty argv."
            )

        for item in runtime.run_args:
            if not isinstance(item, str):
                raise ModalConfigurationError(
                    f"Each argument in runtime.run_args must be a string, got {type(item).__name__}."
                )

        if not runtime.run_args[0].strip():
            raise ModalConfigurationError(
                "Modal runtime.run_args executable (argv[0]) cannot be empty or whitespace."
            )

        return list(runtime.run_args)

    def new_deployment_id(self, profile: ModelProfile) -> str:
        return f"iw-modal-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"

    def resource_ref(
        self, deployment_id: str, request: DeploymentRequest, account: ProviderAccount
    ) -> ResourceRef:
        return ResourceRef(name=deployment_id, scope=account.get_metadata("environment"))

    def dry_run_endpoint(self, deployment_id: str, runtime: RuntimeSpec) -> str:
        return f"https://dryrun-{deployment_id}.modal.run"

    def preflight(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> None:
        self._get_modal_module()
        self._validate_runtime(profile, runtime)
        requires_proxy_auth = (
            request.options.provider.requires_proxy_auth if request.options else True
        )
        if not requires_proxy_auth or request.dry_run:
            return
        # Without proxy tokens the app would deploy but be unreachable for InferWeave's own
        # readiness probes and inference calls.
        auth = self._endpoint_auth or default_endpoint_auth()
        if not auth.is_configured_for(self.name, account):
            if account.is_ambient:
                hint = (
                    "export MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET (or pass "
                    "endpoint_auth= to InferWeave)"
                )
            else:
                hint = f"configure proxy_token_id/proxy_token_secret for account '{account.id}'"
            message = (
                "Modal endpoints are protected with proxy auth, but no proxy token was found. "
                f"Create a token in the Modal workspace settings and {hint}, or deliberately "
                "opt out with custom_args={'requires_proxy_auth': False}."
            )
            if account.is_ambient:
                raise ProviderAuthError(message)
            # Missing endpoint credentials prevent this account serving the request, but do
            # not prove its control-plane credentials invalid. Park it and allow failover.
            raise ProviderOperationError(
                message, FailureKind.PERMISSION, resource_may_exist=False,
            )

    # --- credentials -----------------------------------------------------------------------

    def _client(self, account: ProviderAccount) -> Any:
        """Explicit Modal client for ``account`` (``None`` = Modal's own default resolution).

        Clients are cached per account and rebuilt when the account's secret rotates. Blocking;
        call from a worker thread.
        """
        if account.is_ambient:
            return None
        modal = self._get_modal_module()
        fingerprint = account.secret_fingerprint
        with self._clients_lock:
            cached = self._clients.get(account.id)
            if cached is not None and cached[0] == fingerprint:
                return cached[1]
            client = modal.Client.from_credentials(
                account.secret("token_id"), account.secret("token_secret")
            )
            self._clients[account.id] = (fingerprint, client)
            return client

    def _failure(
        self, operation: str, err: Exception, account: ProviderAccount, deployment_id: str
    ) -> ProviderOperationError:
        kind = _ERROR_KINDS.get(type(err).__name__, FailureKind.TRANSIENT)
        if type(err).__name__ == "ResourceExhaustedError":
            detail = str(err).lower()
            if "quota" in detail or "balance" in detail:
                kind = FailureKind.QUOTA
            elif "capacity" in detail or "out of stock" in detail:
                kind = FailureKind.CAPACITY
        return ProviderOperationError(
            f"Modal {operation} failed for account '{account.id}' ({type(err).__name__}).",
            kind,
            retry_after=retry_after_from_error(err),
            deployment_id=deployment_id,
            resource_may_exist=operation == "deploy" and not kind.is_definitive_rejection,
            operation_may_continue=operation == "deploy" and not kind.is_definitive_rejection,
        )

    async def _call(
        self, operation: str, record: DeploymentRecord, account: ProviderAccount, fn: Any
    ) -> Any:
        task = asyncio.ensure_future(asyncio.to_thread(fn))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The SDK call keeps running in its thread; wait for it so the caller's cleanup
            # sees the app if the deploy went through, then let the cancellation win.
            try:
                await task
                if operation == "deploy":
                    record.creation_may_continue = False
            except Exception as err:  # noqa: BLE001 - cancellation wins, details withheld
                logger.debug("Cancelled Modal call ended with %s.", type(err).__name__)
            raise
        except ProviderOperationError:
            raise
        except Exception as err:  # noqa: BLE001 - classified and sanitized below
            raise self._failure(operation, err, account, record.id) from None

    # --- lifecycle ---------------------------------------------------------------------------

    async def prepare(self, record: DeploymentRecord, account: ProviderAccount) -> None:
        await self._call("authenticate", record, account, lambda: self._client(account))

    async def provision(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> ProvisionResult:
        """Configures and launches a serverless deployment on Modal."""
        try:
            modal = self._get_modal_module()
            run_args = self._validate_runtime(profile, runtime)
        except BaseException:
            record.creation_may_continue = False
            raise
        assert record.resource is not None
        app_name = record.resource.name
        environment = record.resource.scope
        requires_proxy_auth = (
            request.options.provider.requires_proxy_auth if request.options else True
        )

        # Build container image dynamically from runtime template spec
        from_registry_kwargs: dict[str, Any] = {
            "setup_dockerfile_commands": runtime.metadata.get(
                "modal_setup_dockerfile_commands", []
            ),
        }
        if runtime.add_python is not None:
            from_registry_kwargs["add_python"] = runtime.add_python.strip()
        image = modal.Image.from_registry(
            runtime.docker_image.strip(),
            **from_registry_kwargs,
        ).entrypoint([])
        if runtime.setup_commands:
            image = image.run_commands(*runtime.setup_commands)
        if runtime.env_vars:
            image = image.env(runtime.env_vars)
        # Local sources are mounted at container startup. Modal forbids subsequent
        # image build steps, so this must follow all RUN/ENV layers.
        if hasattr(image, "add_local_python_source"):
            try:
                image = image.add_local_python_source("inferweave")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not add_local_python_source('inferweave'): %s", type(exc).__name__)
        app = modal.App(name=app_name)

        # Map hardware to Modal GPU specification
        hardware = runtime.hardware or profile.hardware
        gpu_type = request.gpu_type or (
            hardware.recommended_gpus[0]
            if hardware.recommended_gpus
            else "A10G"
        )
        gpu_count = request.num_gpus or hardware.gpu_count
        gpu_spec = f"{gpu_type}:{gpu_count}" if gpu_count > 1 else gpu_type

        # Configure container scaling and timeout parameters from options
        provider_opts = request.options.provider if request.options else None
        scaledown_window = (
            provider_opts.scaledown_window_seconds
            if provider_opts and provider_opts.scaledown_window_seconds is not None
            else DEFAULT_SCALEDOWN_WINDOW_SECONDS
        )

        timeout_secs = (
            provider_opts.timeout_seconds
            if (provider_opts and provider_opts.timeout_seconds)
            else 86400
        )

        fn_kwargs: dict[str, Any] = {
            "image": image,
            "gpu": gpu_spec,
            "timeout": timeout_secs,
            "scaledown_window": scaledown_window,
            "serialized": True,
        }

        if provider_opts and provider_opts.extra_provider_args:
            if "cpu" in provider_opts.extra_provider_args:
                fn_kwargs["cpu"] = provider_opts.extra_provider_args["cpu"]
            if "memory" in provider_opts.extra_provider_args:
                fn_kwargs["memory"] = provider_opts.extra_provider_args["memory"]

        # Register containerized web server function listening on runtime port
        @app.function(**fn_kwargs)
        @modal.web_server(
            port=runtime.port,
            startup_timeout=300,
            requires_proxy_auth=requires_proxy_auth,
        )
        def serve() -> None:
            import subprocess

            subprocess.Popen(list(run_args), shell=False)

        def _deploy_sync() -> str | None:
            client = self._client(account)
            app.deploy(name=app_name, environment_name=environment, client=client)
            # From here on the app exists: even a definitive endpoint-lookup rejection
            # must preserve cleanup/reconciliation under the owning account.
            record.creation_may_continue = False
            try:
                return serve.get_web_url()
            except Exception as err:  # noqa: BLE001 - sanitized, resource already deployed
                failure = self._failure("endpoint lookup", err, account, record.id)
                failure.resource_may_exist = True
                raise failure from None

        endpoint_url = await self._call("deploy", record, account, _deploy_sync)
        record.creation_may_continue = False
        # A successful app.deploy() only proves the app was deployed, not that the model
        # runtime is serving. Report STARTING; HEALTHY is assigned only after a
        # successful readiness healthcheck.
        return ProvisionResult(
            state=DeploymentState.STARTING,
            endpoint_url=endpoint_url or f"https://{app_name}.modal.run",
        )

    async def _lookup_app(self, record: DeploymentRecord, account: ProviderAccount) -> bool:
        """Whether the Modal app named by the record exists in the owning workspace."""
        modal = self._get_modal_module()
        name = record.resource.name if record.resource else record.id
        environment = record.resource.scope if record.resource else None

        def _lookup() -> bool:
            try:
                modal.App.lookup(name, environment_name=environment, client=self._client(account))
            except modal.exception.NotFoundError:
                return False
            return True

        result: bool = await self._call("status lookup", record, account, _lookup)
        return result

    async def _stop_modal_app(self, record: DeploymentRecord, account: ProviderAccount) -> None:
        """Stops the deployed Modal app via the public SDK (idempotent).

        Uses ``modal.experimental.stop_app`` (public, non-underscore; verified on modal 1.6+).
        Stopping an app that is already gone is treated as success.
        """
        modal = self._get_modal_module()
        name = record.resource.name if record.resource else record.id
        environment = record.resource.scope if record.resource else None
        stop_fn = getattr(getattr(modal, "experimental", None), "stop_app", None)
        if not callable(stop_fn):
            raise ProviderOperationError(
                "Modal SDK does not expose a supported public app stop API "
                "(expected modal.experimental.stop_app).",
                FailureKind.INVALID_REQUEST,
                deployment_id=record.id,
            )

        def _stop() -> None:
            try:
                stop_fn(name, environment_name=environment, client=self._client(account))
            except modal.exception.NotFoundError:
                logger.info("Modal app '%s' not found; treating it as already stopped.", name)

        await self._call("stop", record, account, _stop)

    async def status(self, record: DeploymentRecord, account: ProviderAccount) -> ResourceStatus:
        """Maps Modal app presence to an infrastructure-level state.

        A deployed Modal app only proves the infrastructure exists, not that the model runtime
        is serving, so it maps to STARTING; HEALTHY requires an application readiness probe.
        """
        exists = await self._lookup_app(record, account)
        if not exists:
            return ResourceStatus(state=DeploymentState.STOPPED, endpoint_url=record.endpoint_url)
        return ResourceStatus(state=DeploymentState.STARTING, endpoint_url=record.endpoint_url)

    async def resource_exists(self, record: DeploymentRecord, account: ProviderAccount) -> bool:
        return await self._lookup_app(record, account)

    async def stop(
        self,
        record: DeploymentRecord,
        account: ProviderAccount,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Stops the Modal app and terminates its running containers (stop and down alike)."""
        await self._stop_modal_app(record, account)
