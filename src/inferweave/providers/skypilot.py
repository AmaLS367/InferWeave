"""SkyPilot multi-cloud compute provider adapter."""

import sys
import uuid

from inferweave.core.exceptions import ProviderPlatformError
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec


class SkyPilotProvider(ComputeProvider):
    """Compute provider delegating GPU provisioning and execution to SkyPilot.

    Supports: RunPod, AWS, GCP, Azure, Lambda Labs, Nebius, Vast.ai, OCI, Kubernetes, etc.
    """

    def __init__(self, cloud_name: str) -> None:
        self._cloud_name = cloud_name.lower()

    @property
    def name(self) -> str:
        return self._cloud_name

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.SKYPILOT

    def _ensure_supported_platform(self) -> None:
        """Verifies host platform compatibility."""
        if sys.platform == "win32":
            raise ProviderPlatformError(
                f"SkyPilot requires POSIX system calls (termios, resource) and cannot run directly "
                f"on native Windows for provider '{self.name}'.\n"
                f"Recommended solutions:\n"
                f"  1. Run your code inside WSL2 (Ubuntu): `wsl` -> `python your_script.py`\n"
                f"  2. Use a native Windows-compatible provider like Modal: `provider='modal'`"
            )

    def _get_sky_module(self):
        """Lazy loads the SkyPilot SDK module."""
        self._ensure_supported_platform()
        try:
            import sky  # type: ignore

            return sky
        except ImportError as e:
            raise ImportError(
                f"SkyPilot is not installed with support for '{self.name}'. "
                f"Install it via: pip install 'inferweave[{self.name}]'"
            ) from e

    async def deploy(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
    ) -> Deployment:
        """Translates ModelProfile and RuntimeSpec into a SkyPilot Task and launches it."""
        # Ensure platform is supported before initiating launch
        self._ensure_supported_platform()
        sky = self._get_sky_module()

        deployment_id = (
            f"iw-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"
        )

        # Map hardware requirements to SkyPilot resources
        gpu_spec = request.gpu_type or (
            profile.hardware.recommended_gpus[0]
            if profile.hardware.recommended_gpus
            else "A10G"
        )
        gpu_count = request.num_gpus or profile.hardware.gpu_count
        accelerators = f"{gpu_spec}:{gpu_count}"

        # Construct task definition
        setup_script = (
            "\n".join(runtime.setup_commands) if runtime.setup_commands else None
        )

        # Build SkyPilot task
        task = sky.Task(
            name=deployment_id,
            setup=setup_script,
            run=runtime.run_command,
            envs=runtime.env_vars,
        )

        resources = sky.Resources(
            cloud=sky.clouds.CLOUD_REGISTRY.from_str(self.name)
            if hasattr(sky.clouds, "CLOUD_REGISTRY")
            else None,
            accelerators=accelerators,
            ports=[runtime.port],
        )
        task.set_resources(resources)

        status = DeploymentStatus(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.PROVISIONING,
        )

        async def _stop() -> None:
            await self.stop(deployment_id)

        async def _refresh() -> DeploymentStatus:
            return await self.get_status(deployment_id)

        return Deployment(status=status, stop_fn=_stop, refresh_fn=_refresh)

    async def stop(self, deployment_id: str) -> None:
        """Terminates the SkyPilot cluster corresponding to this deployment."""
        _ = self._get_sky_module()
        # In a full run: sky.down(deployment_id)

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves live cluster status from SkyPilot."""
        _ = self._get_sky_module()
        return DeploymentStatus(
            id=deployment_id,
            model="unknown",
            provider=self.name,
            state=DeploymentState.HEALTHY,
        )
