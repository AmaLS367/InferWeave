"""Dedicated Modal serverless compute provider adapter."""

import asyncio
import logging
import shlex
import uuid
from typing import Any

from inferweave.adapters.auth import default_endpoint_auth
from inferweave.core.exceptions import ProviderAuthError
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)

# Default Modal scale-to-zero window. Deliberately independent of the InferWeave destroy timer
# (``AutostopPolicy.idle_minutes``): scale-to-zero only frees GPU containers while the app stays
# deployed, whereas the destroy timer stops the whole app.
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 1800


class ModalProvider(ComputeProvider):
    """Serverless GPU provider adapter using Modal.

    Since Modal is serverless and not covered by SkyPilot, this provider executes
    deployments directly via the Modal Python SDK with native container definition.
    """

    def __init__(
        self,
        repository: DeploymentRepositoryPort | None = None,
        endpoint_auth: EndpointAuthPort | None = None,
    ) -> None:
        self._repository = repository
        self._endpoint_auth = endpoint_auth
        self._local_deployments: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "modal"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MODAL

    def _ensure_proxy_credentials(self) -> None:
        """Fails fast when proxy auth is on but no credentials exist to reach the endpoint.

        Without tokens the deployment would be created but unreachable by InferWeave's own
        readiness probes and inference calls.
        """
        auth = self._endpoint_auth or default_endpoint_auth()
        if not auth.is_configured_for(self.name):
            raise ProviderAuthError(
                "Modal endpoints are protected with proxy auth, but no proxy token was found. "
                "Create a token in the Modal workspace settings and export "
                "MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET (or pass endpoint_auth= "
                "to InferWeave), or deliberately opt out with "
                "custom_args={'requires_proxy_auth': False}."
            )

    def _get_modal_module(self):
        """Lazy loads the modal SDK module."""
        try:
            import modal  # type: ignore

            return modal
        except ImportError as e:
            raise ImportError(
                "Modal SDK is not installed. Install it via: pip install 'inferweave[modal]'"
            ) from e

    async def deploy(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
    ) -> Deployment:
        """Configures and launches a serverless deployment on Modal."""
        modal = self._get_modal_module()

        requires_proxy_auth = (
            request.options.provider.requires_proxy_auth if request.options else True
        )
        if requires_proxy_auth and not request.dry_run:
            self._ensure_proxy_credentials()

        deployment_id = (
            f"iw-modal-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"
        )

        record = DeploymentRecord(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.PROVISIONING,
            endpoint_url=f"https://dryrun-{deployment_id}.modal.run"
            if request.dry_run
            else None,
            options=request.options
            or DeploymentOptions.from_custom_args(
                request.custom_args, request.autostop_mins
            ),
            is_dry_run=request.dry_run,
            workload_type=profile.workload_type,
        )
        self._local_deployments[deployment_id] = {
            "model": profile.id,
            "dry_run": request.dry_run,
            "record": record,
        }
        if self._repository:
            await self._repository.save(record)

        # Build container image dynamically from runtime template spec
        image = modal.Image.from_registry(runtime.docker_image)
        if hasattr(image, "add_local_python_source"):
            try:
                image = image.add_local_python_source("inferweave")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not add_local_python_source('inferweave'): %s", exc)
        if runtime.setup_commands:
            image = image.run_commands(*runtime.setup_commands)
        if runtime.env_vars:
            image = image.env(runtime.env_vars)

        app = modal.App(name=deployment_id)

        # Map hardware to Modal GPU specification
        gpu_type = request.gpu_type or (
            profile.hardware.recommended_gpus[0]
            if profile.hardware.recommended_gpus
            else "A10G"
        )
        gpu_count = request.num_gpus or profile.hardware.gpu_count
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

        # Structured argv execution without shell interpretation
        run_args = (
            list(runtime.run_args)
            if runtime.run_args
            else shlex.split(runtime.run_command)
        )

        # Register containerized web server function listening on runtime port
        @app.function(**fn_kwargs)
        @modal.web_server(
            port=runtime.port,
            startup_timeout=300,
            requires_proxy_auth=requires_proxy_auth,
        )
        def serve():
            import subprocess

            subprocess.Popen(run_args, shell=False)

        # Handle dry-run mode without provisioning live Modal workers
        if request.dry_run:
            status = DeploymentStatus(
                id=deployment_id,
                model=profile.id,
                provider=self.name,
                state=DeploymentState.PROVISIONING,
                endpoint_url=f"https://dryrun-{deployment_id}.modal.run",
            )
            return Deployment(
                status=status,
                stop_fn=lambda action=None: self.stop(
                    deployment_id, action=action or AutostopAction.STOP
                ),
                refresh_fn=lambda: self.get_status(deployment_id),
            )

        # Deploy application to Modal platform asynchronously
        def _deploy_sync():
            return app.deploy(name=deployment_id)

        await asyncio.to_thread(_deploy_sync)

        # Retrieve live HTTPS web endpoint URL assigned by Modal
        endpoint_url = getattr(serve, "get_web_url", lambda: None)()
        if not endpoint_url:
            endpoint_url = f"https://{deployment_id}.modal.run"

        # A successful app.deploy() only proves the app was deployed, not that the model
        # runtime is serving. Report STARTING; HEALTHY is assigned only after a
        # successful readiness healthcheck (see InferWeave.deploy / wait_for_ready).
        status = DeploymentStatus(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.STARTING,
            endpoint_url=endpoint_url,
        )
        record.state = DeploymentState.STARTING
        record.endpoint_url = endpoint_url
        if self._repository:
            await self._repository.save(record)

        async def _stop(action: AutostopAction | str | None = None) -> None:
            await self.stop(deployment_id, action=action)

        async def _refresh() -> DeploymentStatus:
            return await self.get_status(deployment_id)

        return Deployment(status=status, stop_fn=_stop, refresh_fn=_refresh)

    async def _stop_modal_app(self, deployment_id: str) -> None:
        """Stops the deployed Modal app named ``deployment_id`` via the public SDK.

        Uses ``modal.experimental.stop_app`` (public, non-underscore; verified on
        modal 1.6+). Supports forward capability detection for public stop APIs.
        Stopping an app that is already gone is treated as success (idempotent).
        """
        modal = self._get_modal_module()
        stop_fn = (
            getattr(getattr(modal, "experimental", None), "stop_app", None)
            or getattr(modal, "stop_app", None)
            or getattr(getattr(modal, "App", None), "stop", None)
        )
        try:
            if callable(stop_fn):
                await asyncio.to_thread(stop_fn, deployment_id)
            else:
                raise RuntimeError(  # noqa: TRY004
                    "Modal SDK does not expose a supported public app stop API "
                    "(expected modal.experimental.stop_app)."
                )
        except modal.exception.NotFoundError:
            logger.info(
                "Modal app '%s' not found; treating it as already stopped.",
                deployment_id,
            )

    async def _fetch_modal_state(self, deployment_id: str) -> DeploymentState:
        """Maps Modal app presence to an infrastructure-level DeploymentState.

        A deployed Modal app only proves the infrastructure exists, not that the model
        runtime is serving, so it maps to STARTING; HEALTHY requires an application
        readiness probe (see LifecycleService.refresh_status).
        """
        modal = self._get_modal_module()
        try:
            await asyncio.to_thread(modal.App.lookup, deployment_id)
        except modal.exception.NotFoundError:
            return DeploymentState.STOPPED
        return DeploymentState.STARTING

    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction | str | None = None,
    ) -> None:
        """Stops the Modal deployment app and terminates its running containers."""
        is_dry_run = False
        if deployment_id in self._local_deployments:
            is_dry_run = self._local_deployments[deployment_id].get("dry_run", False)
        elif self._repository:
            rec = await self._repository.get(deployment_id)
            if rec and rec.is_dry_run:
                is_dry_run = True

        if is_dry_run:
            logger.info("Dry-run Modal deployment '%s' marked stopped.", deployment_id)
            if deployment_id in self._local_deployments:
                self._local_deployments[deployment_id]["record"].mark_stopped()
            if self._repository:
                rec = await self._repository.get(deployment_id)
                if rec:
                    rec.mark_stopped()
                    await self._repository.save(rec)
            return

        self._get_modal_module()
        try:
            await self._stop_modal_app(deployment_id)
        except Exception as err:
            logger.error("Failed to stop Modal app '%s': %s", deployment_id, err)
            raise

        # Only on confirmed success update persistent state to STOPPED
        if deployment_id in self._local_deployments:
            self._local_deployments[deployment_id]["record"].mark_stopped()
        if self._repository:
            rec = await self._repository.get(deployment_id)
            if rec:
                rec.mark_stopped()
                await self._repository.save(rec)

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves deployment state from Modal."""
        model_name = "unknown"
        is_dry_run = False
        cached_record = None

        if self._repository:
            cached_record = await self._repository.get(deployment_id)
            if cached_record:
                model_name = cached_record.model
                is_dry_run = cached_record.is_dry_run

        if model_name == "unknown" and deployment_id in self._local_deployments:
            meta = self._local_deployments[deployment_id]
            model_name = meta.get("model", "unknown")
            is_dry_run = meta.get("dry_run", False)
            cached_record = meta.get("record")

        if is_dry_run:
            state = (
                cached_record.state if cached_record else DeploymentState.PROVISIONING
            )
            endpoint = cached_record.endpoint_url if cached_record else None
            return DeploymentStatus(
                id=deployment_id,
                model=model_name,
                provider=self.name,
                state=state,
                endpoint_url=endpoint,
            )

        self._get_modal_module()
        endpoint = cached_record.endpoint_url if cached_record else None
        try:
            state = await self._fetch_modal_state(deployment_id)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Failed to get Modal status for '%s': %s", deployment_id, err
            )
            state = cached_record.state if cached_record else DeploymentState.PENDING
        return DeploymentStatus(
            id=deployment_id,
            model=model_name,
            provider=self.name,
            state=state,
            endpoint_url=endpoint,
        )
