"""SkyPilot multi-cloud compute provider adapter.

SkyPilot calls run in ``inferweave/isolation/skypilot_worker.py`` worker processes. RunPod and
Vast.ai account pools get one private SkyPilot home and API server per account (see the worker
docstring), so clusters, credentials and cached SDK state never cross accounts. Clouds without a
pool, and RunPod/Vast without one, use the user's own SkyPilot setup (the ambient account).
The worker may run under a separate interpreter (``INFERWEAVE_SKYPILOT_PYTHON``), keeping
SkyPilot's dependencies apart from Lightning SDK's.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from inferweave.accounts.models import ProviderAccount
from inferweave.core.credentials import is_sensitive_key
from inferweave.core.exceptions import (
    DeploymentError,
    ProviderOperationError,
    ProviderPlatformError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.domain.lifecycle import AutostopAction
from inferweave.isolation import (
    WORKER_DIR,
    WorkerRunner,
    account_environment,
    ambient_environment,
)
from inferweave.isolation.file_lock import file_lock
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.providers.base import ComputeProvider, ProvisionResult, ResourceStatus
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)

PYTHON_ENV = "INFERWEAVE_SKYPILOT_PYTHON"
POOLED_CLOUDS = frozenset({"runpod", "vast"})
_PORT_BASE = 47000
_PORT_SLOTS = 1000
_ports_lock = threading.Lock()


class SkyPilotConfigurationError(DeploymentError, ValueError):
    """Raised when a runtime or container specification is invalid for SkyPilot."""


def _uses_inferweave_worker(run_args: list[str]) -> bool:
    """Whether the launch argv invokes an in-tree inferweave Python module."""
    return any(
        arg == "inferweave" or arg.startswith("inferweave.")
        for arg in run_args
    )


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
        python: str | None = None,
        state_dir: Path | None = None,
        runner: WorkerRunner | None = None,
    ) -> None:
        self._cloud_name = cloud_name.lower()
        self.state_dir = state_dir or Path.home() / ".inferweave" / "accounts"
        self.runner = runner or WorkerRunner(
            self._cloud_name,
            WORKER_DIR / "skypilot_worker.py",
            python=python or os.environ.get(PYTHON_ENV),
        )

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

    @staticmethod
    def _validate_runtime(profile: ModelProfile, runtime: RuntimeSpec) -> list[str]:
        """Validates runtime image, port, Python metadata, setup, env vars, and structured argv for SkyPilot."""
        is_container = runtime.name == "container" or profile.container is not None

        if not isinstance(runtime.docker_image, str):
            raise SkyPilotConfigurationError(
                f"SkyPilot runtime.docker_image must be a string, got {type(runtime.docker_image).__name__}."
            )
        stripped_image = runtime.docker_image.strip()
        if is_container and not stripped_image:
            raise SkyPilotConfigurationError(
                f"SkyPilot container deployments require a non-empty container image reference, "
                f"got {runtime.docker_image!r}."
            )
        if runtime.docker_image and (
            not stripped_image
            or stripped_image == "docker:"
            or any(c.isspace() for c in stripped_image)
        ):
            raise SkyPilotConfigurationError(
                f"SkyPilot container image reference cannot be whitespace or contain whitespace, "
                f"got {runtime.docker_image!r}."
            )

        if runtime.add_python is not None:
            if not isinstance(runtime.add_python, str):
                raise SkyPilotConfigurationError(
                    f"SkyPilot runtime.add_python must be a string or None, got {type(runtime.add_python).__name__}."
                )
            cleaned_py = runtime.add_python.strip()
            if not cleaned_py or not re.fullmatch(r"^\d+(\.\d+)*$", cleaned_py):
                raise SkyPilotConfigurationError(
                    f"Invalid Python version string {runtime.add_python!r} in runtime.add_python; "
                    "expected format like '3.12'."
                )

        if (
            isinstance(runtime.port, bool)
            or not isinstance(runtime.port, int)
            or runtime.port < 1
            or runtime.port > 65535
        ):
            raise SkyPilotConfigurationError(
                f"SkyPilot runtime.port must be an integer from 1 to 65535, got {runtime.port!r}."
            )

        if runtime.healthcheck is not None and runtime.port != runtime.healthcheck.port:
            raise SkyPilotConfigurationError(
                f"Inconsistent configuration: runtime.port ({runtime.port}) and "
                f"runtime.healthcheck.port ({runtime.healthcheck.port}) must match."
            )

        if (
            profile.container is not None
            and profile.healthcheck is not None
            and runtime.port != profile.healthcheck.port
        ):
            raise SkyPilotConfigurationError(
                f"Inconsistent configuration: runtime.port ({runtime.port}) and "
                f"profile.healthcheck.port ({profile.healthcheck.port}) must match."
            )

        if not isinstance(runtime.setup_commands, (list, tuple)) or any(
            not isinstance(cmd, str) for cmd in runtime.setup_commands
        ):
            raise SkyPilotConfigurationError(
                "SkyPilot runtime.setup_commands must be a list of strings."
            )

        if not isinstance(runtime.env_vars, dict):
            raise SkyPilotConfigurationError(
                f"SkyPilot runtime.env_vars must be a dict, got {type(runtime.env_vars).__name__}."
            )
        for k, v in runtime.env_vars.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise SkyPilotConfigurationError(
                    "SkyPilot environment variable keys and values must be strings."
                )
            if is_container and is_sensitive_key(k):
                raise SkyPilotConfigurationError(
                    f"Platform credentials and sensitive keys are forbidden in SkyPilot container env_vars: '{k}'"
                )

        if not isinstance(runtime.run_args, (list, tuple)) or not runtime.run_args:
            raise SkyPilotConfigurationError(
                "SkyPilot deployments require non-empty command argv (runtime.run_args / container.args); "
                "SkyPilot Task.run requires an explicit launch command and cannot execute an empty argv."
            )

        for item in runtime.run_args:
            if not isinstance(item, str):
                raise SkyPilotConfigurationError(
                    f"Each argument in runtime.run_args must be a string, got {type(item).__name__}."
                )

        if not runtime.run_args[0].strip():
            raise SkyPilotConfigurationError(
                "SkyPilot runtime.run_args executable (argv[0]) cannot be empty or whitespace."
            )

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", runtime.run_args[0].strip()):
            raise SkyPilotConfigurationError(
                f"SkyPilot runtime.run_args executable (argv[0]) must be a command or path, "
                f"not a shell variable assignment: {runtime.run_args[0]!r}."
            )

        return list(runtime.run_args)

    def new_deployment_id(self, profile: ModelProfile) -> str:
        return f"iw-{profile.id.replace('/', '-').lower()}-{uuid.uuid4().hex[:6]}"

    def resource_ref(
        self, deployment_id: str, request: DeploymentRequest, account: ProviderAccount
    ) -> ResourceRef:
        return ResourceRef(name=deployment_id)

    def dry_run_endpoint(self, deployment_id: str, runtime: RuntimeSpec) -> str:
        return f"http://dryrun-{deployment_id}.cloud:{runtime.port}"

    def preflight(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> None:
        self._validate_runtime(profile, runtime)
        if request.dry_run:
            return
        self._ensure_supported_platform()
        if not account.is_ambient:
            account.secret("api_key")

    # --- account isolation ---------------------------------------------------------------------

    def _account_dir(self, account: ProviderAccount) -> Path:
        digest = hashlib.sha256(f"{account.id}/{account.owner_fingerprint}".encode()).hexdigest()[:16]
        return self.state_dir / "skypilot" / self.name / digest

    def account_port(self, account: ProviderAccount) -> int:
        """Stable localhost port of the account's private SkyPilot API server."""
        registry = self.state_dir / "skypilot" / "ports.json"
        key = f"{self.name}/{account.id}/{account.owner_fingerprint}"
        with _ports_lock, file_lock(registry.with_suffix(".lock"), blocking=True):
            try:
                ports: dict[str, int] = json.loads(registry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ports = {}
            if key in ports:
                return int(ports[key])
            used = set(ports.values())
            slot = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % _PORT_SLOTS
            for offset in range(_PORT_SLOTS):
                # API, metrics and the account-private multiprocessing request queue.
                port = _PORT_BASE + ((slot + offset) % _PORT_SLOTS) * 3
                if port not in used:
                    break
            else:  # pragma: no cover - a thousand accounts on one machine
                raise ProviderPlatformError("No free port for another SkyPilot account server.")
            ports[key] = port
            registry.parent.mkdir(parents=True, exist_ok=True)
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(ports, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, registry)
            return port

    async def _run(
        self,
        operation: str,
        payload: dict[str, Any],
        account: ProviderAccount,
        *,
        timeout: float,
        deployment_id: str | None = None,
    ) -> Any:
        self._ensure_supported_platform()
        payload = {"cloud": self.name, **payload}
        secrets: dict[str, str] = {}
        if account.is_ambient:
            env = ambient_environment()
            payload["mode"] = "ambient"
        else:
            if self.name not in POOLED_CLOUDS:
                raise ProviderOperationError(
                    f"Account pools are not supported for SkyPilot cloud '{self.name}'.",
                    FailureKind.INVALID_REQUEST,
                    resource_may_exist=False,
                )
            env = account_environment(self._account_dir(account) / "home")
            payload.update(
                mode="account",
                server_port=self.account_port(account),
                fingerprint=account.owner_fingerprint,
            )
            secrets["api_key"] = account.secret("api_key")
        return await self.runner.run(
            operation,
            payload,
            env=env,
            secrets=secrets,
            timeout=timeout,
            deployment_id=deployment_id,
            account_id=account.id,
        )

    # --- lifecycle ---------------------------------------------------------------------------

    def launch_payload(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
    ) -> dict[str, Any]:
        """Translates ModelProfile and RuntimeSpec into the worker's SkyPilot task spec."""
        assert record.resource is not None
        run_args = self._validate_runtime(profile, runtime)
        hardware = runtime.hardware or profile.hardware
        gpu_spec = request.gpu_type or (
            hardware.recommended_gpus[0]
            if hardware.recommended_gpus
            else "A10G"
        )
        gpu_count = request.num_gpus or hardware.gpu_count
        provider_opts = request.options.provider if request.options else None
        workdir = None
        if provider_opts and provider_opts.extra_provider_args:
            workdir = provider_opts.extra_provider_args.get("workdir")

        # SkyPilot's Task.run is a shell string. It is built exclusively from the structured
        # argv (every token shell-quoted), never from the free-form run_command, so model IDs,
        # engine args and extra_cli_args stay literal arguments instead of shell fragments.
        run_script = shlex.join(run_args)
        if workdir is None and _uses_inferweave_worker(run_args):
            workdir = resolve_worker_workdir()

        task_envs = dict(runtime.env_vars) if runtime.env_vars else {}
        if workdir and "PYTHONPATH" not in task_envs:
            task_envs["PYTHONPATH"] = ".:$PYTHONPATH"

        # Determine provider-native autostop policy (action & idle minutes)
        autodown = False
        idle_mins = request.autostop_mins
        if request.options and request.options.autostop:
            if request.options.autostop.enabled:
                autodown = request.options.autostop.action == AutostopAction.DOWN
                if request.options.autostop.idle_minutes is not None:
                    idle_mins = request.options.autostop.idle_minutes
            else:
                idle_mins = None
        elif provider_opts and provider_opts.autodown:
            autodown = True

        resources: dict[str, Any] = {
            "cloud": self.name,
            "accelerators": f"{gpu_spec}:{gpu_count}",
            "ports": [runtime.port],
            "use_spot": provider_opts.allow_spot if provider_opts else True,
        }
        if runtime.docker_image:
            image_ref = runtime.docker_image.strip()
            resources["image_id"] = (
                image_ref if image_ref.startswith("docker:") else f"docker:{image_ref}"
            )
        if provider_opts and provider_opts.preferred_regions:
            resources["region"] = provider_opts.preferred_regions[0]
        if provider_opts and provider_opts.disk_size_gb:
            resources["disk_size"] = provider_opts.disk_size_gb
        if provider_opts and provider_opts.extra_provider_args:
            for k in ("zone", "image_id"):
                if k in provider_opts.extra_provider_args:
                    resources[k] = provider_opts.extra_provider_args[k]

        return {
            "cluster": record.resource.name,
            "setup": "\n".join(runtime.setup_commands) if runtime.setup_commands else None,
            "run": run_script,
            "envs": task_envs,
            "workdir": workdir,
            "resources": resources,
            "idle_minutes": idle_mins,
            "down": autodown,
            "port": runtime.port,
        }

    async def provision(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> ProvisionResult:
        try:
            payload = self.launch_payload(record, request, profile, runtime)
        except BaseException:
            record.creation_may_continue = False
            raise
        result = await self._run(
            "launch",
            payload,
            account,
            timeout=3600,
            deployment_id=record.id,
        )
        endpoint_url = result.get("endpoint") if isinstance(result, dict) else None
        # A launched cluster with an endpoint only proves the infrastructure is up, not that
        # the model runtime is serving. HEALTHY is assigned only after a readiness healthcheck.
        return ProvisionResult(
            state=DeploymentState.STARTING if endpoint_url else DeploymentState.PROVISIONING,
            endpoint_url=endpoint_url,
        )

    def _cluster(self, record: DeploymentRecord) -> dict[str, Any]:
        return {"cluster": record.resource.name if record.resource else record.id}

    async def status(self, record: DeploymentRecord, account: ProviderAccount) -> ResourceStatus:
        data = await self._run(
            "status", self._cluster(record), account, timeout=180, deployment_id=record.id
        )
        if not isinstance(data, dict) or not isinstance(data.get("exists"), bool):
            raise ProviderOperationError("SkyPilot returned an invalid status.", FailureKind.TRANSIENT)
        if not data["exists"]:
            return ResourceStatus(state=DeploymentState.STOPPED, endpoint_url=record.endpoint_url)
        status = str(data.get("status") or "")
        # Cluster UP is infrastructure state only; readiness is established by a probe.
        if "UP" in status:
            state = DeploymentState.STARTING
        elif "INIT" in status:
            state = DeploymentState.PROVISIONING
        elif "STOPPED" in status:
            state = DeploymentState.STOPPED
        else:
            state = DeploymentState.PENDING
        return ResourceStatus(state=state, endpoint_url=data.get("endpoint") or record.endpoint_url)

    async def resource_exists(self, record: DeploymentRecord, account: ProviderAccount) -> bool:
        data = await self._run(
            "status", self._cluster(record), account, timeout=180, deployment_id=record.id
        )
        if not isinstance(data, dict) or not isinstance(data.get("exists"), bool):
            raise ProviderOperationError("SkyPilot returned an invalid status.", FailureKind.TRANSIENT)
        return bool(data["exists"])

    async def stop(
        self,
        record: DeploymentRecord,
        account: ProviderAccount,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Stops (disk kept, still billed) or tears down (DOWN) the cluster."""
        target = AutostopAction(action.lower()) if isinstance(action, str) else action
        operation = "down" if target == AutostopAction.DOWN else "stop"
        logger.info("%s SkyPilot cluster '%s'...", operation.title(), record.id)
        await self._run(operation, self._cluster(record), account, timeout=900, deployment_id=record.id)
