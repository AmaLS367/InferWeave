"""Dedicated Modal serverless compute provider adapter."""

import asyncio
import logging
import uuid
from typing import Any

from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)


class ModalProvider(ComputeProvider):
    """Serverless GPU provider adapter using Modal.

    Since Modal is serverless and not covered by SkyPilot, this provider executes
    deployments directly via the Modal Python SDK with native container definition.
    """

    def __init__(self, repository: DeploymentRepositoryPort | None = None) -> None:
        self._repository = repository
        self._local_deployments: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "modal"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MODAL

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

        deployment_id = (
            f"iw-modal-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"
        )

        record = DeploymentRecord(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.PROVISIONING
            if request.dry_run
            else DeploymentState.HEALTHY,
            endpoint_url=f"https://dryrun-{deployment_id}.modal.run"
            if request.dry_run
            else None,
            options=request.options
            or DeploymentOptions.from_custom_args(
                request.custom_args, request.autostop_mins
            ),
            is_dry_run=request.dry_run,
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
            (request.autostop_mins * 60)
            if request.autostop_mins
            else (provider_opts.scaledown_window_seconds if provider_opts else 1800)
        )
        timeout_secs = (
            provider_opts.timeout_seconds
            if (provider_opts and provider_opts.timeout_seconds)
            else 86400
        )
        run_command = runtime.run_command

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
        @modal.web_server(port=runtime.port, startup_timeout=300)
        def serve():
            import subprocess

            subprocess.Popen(run_command, shell=True)

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
                stop_fn=lambda: self.stop(deployment_id),
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

        status = DeploymentStatus(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.HEALTHY,
            endpoint_url=endpoint_url,
        )
        record.state = DeploymentState.HEALTHY
        record.endpoint_url = endpoint_url
        if self._repository:
            await self._repository.save(record)

        async def _stop(action: AutostopAction = AutostopAction.STOP) -> None:
            await self.stop(deployment_id, action=action)

        async def _refresh() -> DeploymentStatus:
            return await self.get_status(deployment_id)

        return Deployment(status=status, stop_fn=_stop, refresh_fn=_refresh)

    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction = AutostopAction.STOP,
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
            from modal.cli.app import resolve_app_identifier
            from modal.client import _Client
            from modal_proto import api_pb2

            client = await _Client.from_env()
            app_id, _, lifecycle = await resolve_app_identifier(
                deployment_id, None, client
            )
            if lifecycle.app_state != api_pb2.APP_STATE_STOPPED:
                req = api_pb2.AppStopRequest(
                    app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI
                )
                await client.stub.AppStop(req)
        except (RuntimeError, ValueError, OSError) as err:
            logger.warning("Failed to stop Modal app '%s': %s", deployment_id, err)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Unexpected error stopping Modal app '%s': %s", deployment_id, err
            )
        finally:
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
        try:
            from modal.cli.app import resolve_app_identifier
            from modal.client import _Client
            from modal_proto import api_pb2

            client = await _Client.from_env()
            _, _, lifecycle = await resolve_app_identifier(deployment_id, None, client)
            state_map = {
                api_pb2.APP_STATE_DEPLOYED: DeploymentState.HEALTHY,
                api_pb2.APP_STATE_EPHEMERAL: DeploymentState.HEALTHY,
                api_pb2.APP_STATE_INITIALIZING: DeploymentState.PROVISIONING,
                api_pb2.APP_STATE_STOPPED: DeploymentState.STOPPED,
                api_pb2.APP_STATE_STOPPING: DeploymentState.STOPPED,
                api_pb2.APP_STATE_DISABLED: DeploymentState.FAILED,
            }
            state = state_map.get(lifecycle.app_state, DeploymentState.PENDING)
            endpoint = cached_record.endpoint_url if cached_record else None
            return DeploymentStatus(
                id=deployment_id,
                model=model_name,
                provider=self.name,
                state=state,
                endpoint_url=endpoint,
            )
        except (RuntimeError, ValueError, OSError) as err:
            logger.warning(
                "Failed to get Modal status for '%s': %s", deployment_id, err
            )
            fallback_state = (
                cached_record.state if cached_record else DeploymentState.PENDING
            )
            endpoint = cached_record.endpoint_url if cached_record else None
            return DeploymentStatus(
                id=deployment_id,
                model=model_name,
                provider=self.name,
                state=fallback_state,
                endpoint_url=endpoint,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Unexpected error getting Modal status for '%s': %s", deployment_id, err
            )
            fallback_state = (
                cached_record.state if cached_record else DeploymentState.PENDING
            )
            endpoint = cached_record.endpoint_url if cached_record else None
            return DeploymentStatus(
                id=deployment_id,
                model=model_name,
                provider=self.name,
                state=fallback_state,
                endpoint_url=endpoint,
            )
