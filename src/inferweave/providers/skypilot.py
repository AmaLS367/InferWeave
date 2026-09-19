"""SkyPilot multi-cloud compute provider adapter."""

import asyncio
import logging
import sys
import uuid
from typing import Any

from inferweave.core.exceptions import ProviderPlatformError
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)


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

    def _get_sky_module(self) -> Any:
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
        deployment_id = (
            f"iw-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"
        )

        # In dry-run mode, simulate provisioning without launching live cloud resources
        if request.dry_run:
            status = DeploymentStatus(
                id=deployment_id,
                model=profile.id,
                provider=self.name,
                state=DeploymentState.PROVISIONING,
                endpoint_url=f"http://dryrun-{deployment_id}.cloud:{runtime.port}",
            )
            return Deployment(
                status=status,
                stop_fn=lambda: self.stop(deployment_id),
                refresh_fn=lambda: self.get_status(deployment_id),
            )

        # Ensure platform is supported before initiating launch
        self._ensure_supported_platform()
        sky = self._get_sky_module()

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

        # Launch the task asynchronously
        def _launch_sync() -> None:
            sky.launch(
                task,
                cluster_name=deployment_id,
                idle_minutes_to_autostop=request.autostop_mins,
                stream_logs=False,
            )

        await asyncio.to_thread(_launch_sync)

        # Query live cluster endpoint if ready
        endpoint_url = await self._resolve_endpoint(sky, deployment_id, runtime.port)
        state = (
            DeploymentState.HEALTHY
            if endpoint_url is not None
            else DeploymentState.PROVISIONING
        )

        status = DeploymentStatus(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=state,
            endpoint_url=endpoint_url,
        )

        async def _stop() -> None:
            await self.stop(deployment_id)

        async def _refresh() -> DeploymentStatus:
            return await self.get_status(deployment_id)

        return Deployment(status=status, stop_fn=_stop, refresh_fn=_refresh)

    async def _resolve_endpoint(
        self, sky: Any, cluster_name: str, port: int
    ) -> str | None:
        """Safely queries SkyPilot endpoints for a cluster."""

        def _fetch() -> str | None:
            try:
                eps = sky.endpoints(cluster_name, port=port)
                if not eps:
                    return None
                ep = eps.get(port) or next(iter(eps.values()), None)
                if ep:
                    return ep if str(ep).startswith("http") else f"http://{ep}"
                return None
            except (RuntimeError, ValueError, OSError) as err:
                logger.debug(
                    "Endpoints for cluster '%s' not yet ready: %s", cluster_name, err
                )
                return None
            except Exception as err:  # noqa: BLE001
                logger.debug(
                    "Unexpected error fetching endpoints for '%s': %s",
                    cluster_name,
                    err,
                )
                return None

        return await asyncio.to_thread(_fetch)

    async def stop(self, deployment_id: str) -> None:
        """Terminates the SkyPilot cluster corresponding to this deployment."""
        self._ensure_supported_platform()
        sky = self._get_sky_module()
        await asyncio.to_thread(sky.down, cluster_name=deployment_id)

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves live cluster status from SkyPilot."""
        self._ensure_supported_platform()
        sky = self._get_sky_module()

        def _fetch_status() -> tuple[DeploymentState, str | None]:
            try:
                clusters = sky.status(cluster_names=[deployment_id])
                if not clusters:
                    return DeploymentState.STOPPED, None

                rec = clusters[0]
                status_raw = getattr(rec, "status", None)
                if status_raw is None and isinstance(rec, dict):
                    status_raw = rec.get("status")

                status_str = str(getattr(status_raw, "value", status_raw)).upper()

                if "UP" in status_str:
                    state = DeploymentState.HEALTHY
                elif "INIT" in status_str:
                    state = DeploymentState.PROVISIONING
                elif "STOPPED" in status_str:
                    state = DeploymentState.STOPPED
                else:
                    state = DeploymentState.PENDING

                # Try fetching endpoint if UP
                endpoint = None
                if state == DeploymentState.HEALTHY:
                    try:
                        eps = sky.endpoints(deployment_id)
                        if eps:
                            ep = next(iter(eps.values()), None)
                            if ep:
                                endpoint = (
                                    ep if str(ep).startswith("http") else f"http://{ep}"
                                )
                    except (RuntimeError, ValueError, OSError) as err:
                        logger.debug(
                            "Endpoint query failed for healthy cluster: %s", err
                        )
                    except Exception as err:  # noqa: BLE001
                        logger.debug("Unexpected endpoint query error: %s", err)

                return state, endpoint
            except (RuntimeError, ValueError, OSError) as err:
                logger.warning(
                    "Failed to fetch SkyPilot status for '%s': %s", deployment_id, err
                )
                return DeploymentState.PENDING, None
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Unexpected error fetching SkyPilot status for '%s': %s",
                    deployment_id,
                    err,
                )
                return DeploymentState.PENDING, None

        state, endpoint = await asyncio.to_thread(_fetch_status)
        return DeploymentStatus(
            id=deployment_id,
            model="unknown",
            provider=self.name,
            state=state,
            endpoint_url=endpoint,
        )
