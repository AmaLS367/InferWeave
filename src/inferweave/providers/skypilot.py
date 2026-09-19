"""SkyPilot multi-cloud compute provider adapter."""

import asyncio
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from inferweave.core.exceptions import ProviderPlatformError
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


def resolve_worker_workdir(
    pkg_path: Path | None = None,
    staging_base: Path | None = None,
) -> str | None:
    """Resolves a lightweight workdir containing only the inferweave package for remote workers.

    Guarantees that an entire site-packages or dist-packages directory is never synced to remote cluster.
    - If running from a source checkout (parent directory is 'src'), returns 'src/'.
    - If running from an installed wheel (site-packages / dist-packages), stages only the 'inferweave'
      package directory into '~/.inferweave/staging/worker_pkg' and returns that path.
    """
    pkg_dir = pkg_path or Path(__file__).resolve().parent.parent
    if not pkg_dir.is_dir() or pkg_dir.name != "inferweave":
        return None

    parent_dir = pkg_dir.parent

    # 1. Source checkout: parent is named 'src'
    if parent_dir.name == "src" and (parent_dir / "inferweave").is_dir():
        return str(parent_dir)

    # 2. Installed wheel / site-packages: create lightweight staging directory
    if staging_base is None:
        env_staging = os.environ.get("INFERWEAVE_STAGING_DIR")
        staging_base = (
            Path(env_staging)
            if env_staging
            else Path.home() / ".inferweave" / "staging"
        )

    worker_stage = staging_base / "worker_pkg"
    target_inferweave = worker_stage / "inferweave"

    try:
        worker_stage.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            pkg_dir,
            target_inferweave,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        return str(worker_stage)
    except OSError as err:
        logger.warning("Could not stage inferweave package for worker: %s", err)
        return None


class SkyPilotProvider(ComputeProvider):

    """Compute provider delegating GPU provisioning and execution to SkyPilot.

    Supports: RunPod, AWS, GCP, Azure, Lambda Labs, Nebius, Vast.ai, OCI, Kubernetes, etc.
    """

    def __init__(
        self,
        cloud_name: str,
        repository: DeploymentRepositoryPort | None = None,
    ) -> None:
        self._cloud_name = cloud_name.lower()
        self._repository = repository
        self._local_deployments: dict[str, dict[str, Any]] = {}

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

        record = DeploymentRecord(
            id=deployment_id,
            model=profile.id,
            provider=self.name,
            state=DeploymentState.PROVISIONING
            if request.dry_run
            else DeploymentState.HEALTHY,
            endpoint_url=f"http://dryrun-{deployment_id}.cloud:{runtime.port}"
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

        provider_opts = request.options.provider if request.options else None
        workdir = None
        if provider_opts and provider_opts.extra_provider_args:
            workdir = provider_opts.extra_provider_args.get("workdir")

        if workdir is None and runtime.run_command and "inferweave" in runtime.run_command:
            workdir = resolve_worker_workdir()


        task_envs = dict(runtime.env_vars) if runtime.env_vars else {}
        if workdir and "PYTHONPATH" not in task_envs:
            task_envs["PYTHONPATH"] = ".:$PYTHONPATH"

        task = sky.Task(
            name=deployment_id,
            setup=setup_script,
            run=runtime.run_command,
            envs=task_envs,
            workdir=workdir,
        )


        use_spot = provider_opts.allow_spot if provider_opts else True
        region = (
            provider_opts.preferred_regions[0]
            if (provider_opts and provider_opts.preferred_regions)
            else None
        )
        disk_size = provider_opts.disk_size_gb if provider_opts else None
        autodown = provider_opts.autodown if provider_opts else False

        resources_kwargs: dict[str, Any] = {
            "cloud": (
                sky.clouds.CLOUD_REGISTRY.from_str(self.name)
                if hasattr(sky.clouds, "CLOUD_REGISTRY")
                else None
            ),
            "accelerators": accelerators,
            "ports": [runtime.port],
            "use_spot": use_spot,
        }
        if runtime.docker_image:
            resources_kwargs["image_id"] = f"docker:{runtime.docker_image}"
        if region:
            resources_kwargs["region"] = region
        if disk_size:
            resources_kwargs["disk_size"] = disk_size

        if provider_opts and provider_opts.extra_provider_args:
            for k in ("zone", "image_id"):
                if k in provider_opts.extra_provider_args:
                    resources_kwargs[k] = provider_opts.extra_provider_args[k]

        resources = sky.Resources(**resources_kwargs)
        task.set_resources(resources)

        # Launch the task asynchronously
        def _launch_sync() -> None:
            sky.launch(
                task,
                cluster_name=deployment_id,
                idle_minutes_to_autostop=request.autostop_mins,
                down=autodown,
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
        record.state = state
        record.endpoint_url = endpoint_url
        if self._repository:
            await self._repository.save(record)

        async def _stop(action: AutostopAction = AutostopAction.STOP) -> None:
            await self.stop(deployment_id, action=action)

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

    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction | str = AutostopAction.STOP,
    ) -> None:
        """Terminates or pauses the SkyPilot cluster corresponding to this deployment."""
        target_action = (
            AutostopAction(action.lower()) if isinstance(action, str) else action
        )
        is_dry_run = False
        if deployment_id in self._local_deployments:
            is_dry_run = self._local_deployments[deployment_id].get("dry_run", False)
        elif self._repository:
            rec = await self._repository.get(deployment_id)
            if rec and rec.is_dry_run:
                is_dry_run = True

        if is_dry_run:
            logger.info("Dry-run deployment '%s' marked stopped.", deployment_id)
            if deployment_id in self._local_deployments:
                self._local_deployments[deployment_id]["record"].mark_stopped()
            if self._repository:
                rec = await self._repository.get(deployment_id)
                if rec:
                    rec.mark_stopped()
                    await self._repository.save(rec)
            return

        self._ensure_supported_platform()
        sky = self._get_sky_module()
        if target_action == AutostopAction.DOWN:
            logger.info("Tearing down SkyPilot cluster '%s'...", deployment_id)
            await asyncio.to_thread(sky.down, cluster_name=deployment_id)
        else:
            logger.info("Stopping SkyPilot cluster '%s'...", deployment_id)
            await asyncio.to_thread(sky.stop, cluster_name=deployment_id)

        if deployment_id in self._local_deployments:
            self._local_deployments[deployment_id]["record"].mark_stopped()
        if self._repository:
            rec = await self._repository.get(deployment_id)
            if rec:
                rec.mark_stopped()
                await self._repository.save(rec)

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves live cluster status from SkyPilot."""
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
                fallback_state = (
                    cached_record.state if cached_record else DeploymentState.PENDING
                )
                fallback_endpoint = (
                    cached_record.endpoint_url if cached_record else None
                )
                return fallback_state, fallback_endpoint
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Unexpected error fetching SkyPilot status for '%s': %s",
                    deployment_id,
                    err,
                )
                fallback_state = (
                    cached_record.state if cached_record else DeploymentState.PENDING
                )
                fallback_endpoint = (
                    cached_record.endpoint_url if cached_record else None
                )
                return fallback_state, fallback_endpoint

        state, endpoint = await asyncio.to_thread(_fetch_status)
        return DeploymentStatus(
            id=deployment_id,
            model=model_name,
            provider=self.name,
            state=state,
            endpoint_url=endpoint,
        )
