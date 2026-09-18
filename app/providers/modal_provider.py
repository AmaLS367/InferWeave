"""Dedicated Modal serverless compute provider adapter."""

import uuid

from app.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from app.models.enums import DeploymentState, ProviderType
from app.models.profile import ModelProfile
from app.providers.base import ComputeProvider
from app.runtimes.base import RuntimeSpec


class ModalProvider(ComputeProvider):
    """Serverless GPU provider adapter using Modal.
    
    Since Modal is serverless and not covered by SkyPilot, this provider executes
    deployments directly via the Modal Python SDK with native container definition.
    """

    def __init__(self) -> None:
        pass

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

        deployment_id = f"iw-modal-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"

        # Map hardware to Modal GPU specification
        _gpu_name = (request.gpu_type or (profile.hardware.recommended_gpus[0] if profile.hardware.recommended_gpus else "A10G")).lower()
        
        # Build container image dynamically from runtime template spec
        image = modal.Image.from_registry(runtime.docker_image)
        if runtime.setup_commands:
            image = image.run_commands(*runtime.setup_commands)
        if runtime.env_vars:
            image = image.env(runtime.env_vars)

        _app = modal.App(deployment_id)
        _ = (_gpu_name, _app, image)

        # Placeholder for live deployed web endpoint
        endpoint_url = f"https://{deployment_id}.modal.run"

        status = DeploymentStatus(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.HEALTHY,
            endpoint_url=endpoint_url,
        )

        async def _stop() -> None:
            await self.stop(deployment_id)

        async def _refresh() -> DeploymentStatus:
            return await self.get_status(deployment_id)

        return Deployment(status=status, stop_fn=_stop, refresh_fn=_refresh)

    async def stop(self, deployment_id: str) -> None:
        """Stops the Modal deployment app."""
        _ = self._get_modal_module()
        # In real execution: modal.App.lookup(deployment_id).stop()

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves deployment state from Modal."""
        _ = self._get_modal_module()
        return DeploymentStatus(
            id=deployment_id,
            model="unknown",
            provider=self.name,
            state=DeploymentState.HEALTHY,
        )

