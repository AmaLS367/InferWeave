"""Lightning container Deployments using the public SDK and its supported teardown CLI."""

import asyncio
import base64
import importlib
import io
import json
import logging
import os
import shlex
import subprocess
import sys
import uuid
import zipfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

from inferweave.adapters.auth import default_endpoint_auth
from inferweave.core.exceptions import (
    DeploymentError,
    DeploymentNotFoundError,
    ProviderAuthError,
    ProviderPlatformError,
)
from inferweave.domain.deployment_record import (
    DeploymentRecord,
    LightningDeploymentMetadata,
)
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions, LightningOptions
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec
from inferweave.services.hardware_service import HardwareValidationService

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Deliberately bounded to machines verified in lightning-sdk 2026.10.1. These are
# implementation capabilities, not a promise of account entitlement or availability.
MACHINES = {
    "T4": "T4",
    "L4": "L4",
    "L40S": "L40S",
    "A100-40GB": "A100_40GB",
    "A100-80GB": "A100_80GB",
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",
}
ENDPOINT_DISCOVERY_TIMEOUT_SECONDS = 60.0


async def _blocking(call: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Drain a synchronous operation on cancellation before cleaning its resources."""
    task = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # The outer cancellation stays authoritative; the provisioning caller owns cleanup.
        with suppress(Exception):
            await task
        raise


def runtime_command(runtime: RuntimeSpec) -> str:
    """Translate the shared runtime recipe into a foreground container command.

    Setup commands are trusted runtime-template code; launch argv stays literal. Worker
    sources come from the installed package, so unreleased InferWeave needs no PyPI upload.
    """
    commands = ["set -eu"]
    if any(arg.startswith("inferweave.workers.") for arg in runtime.run_args):
        package = Path(__file__).resolve().parents[1]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(package / "__init__.py", "inferweave/__init__.py")
            for path in (package / "workers").glob("*.py"):
                archive.write(path, f"inferweave/workers/{path.name}")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        bootstrap = (
            "import base64,io,zipfile;"
            f"zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r})))"
            ".extractall('/tmp/iw-runtime')"
        )
        commands.append(shlex.join(["python3", "-c", bootstrap]))
        commands.append('export PYTHONPATH="/tmp/iw-runtime:${PYTHONPATH:-}"')
    commands.extend(runtime.setup_commands)
    commands.append("exec " + shlex.join(runtime.run_args))
    return shlex.join(["-c", " && ".join(commands)])


class LightningProvider(ComputeProvider):
    """Owns one Lightning Deployment per InferWeave deployment; creates no Studios."""

    cleanup_failed_deployment = True

    def __init__(
        self,
        repository: DeploymentRepositoryPort | None = None,
        endpoint_auth: EndpointAuthPort | None = None,
        resource_prefix: str = "iw-lightning",
    ) -> None:
        self._repository = repository
        self.endpoint_auth = endpoint_auth
        if not resource_prefix.startswith("iw-lightning") or not all(
            c.isalnum() or c == "-" for c in resource_prefix
        ):
            raise ValueError("Lightning resource prefix must start with iw-lightning and use letters/digits/hyphens.")
        self.resource_prefix = resource_prefix
        self._records: dict[str, DeploymentRecord] = {}
        self._deleted_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "lightning"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LIGHTNING

    def bind_repository(self, repository: DeploymentRepositoryPort) -> None:
        self._repository = repository

    def _sdk(self) -> tuple[Any, Any]:
        try:
            return (
                importlib.import_module("lightning_sdk"),
                importlib.import_module("lightning_sdk.deployment"),
            )
        except ImportError:
            raise ImportError(
                "Lightning SDK is not installed. Install via: pip install 'inferweave[lightning]'"
            ) from None

    @staticmethod
    def _credentials() -> None:
        if not os.getenv("LIGHTNING_USER_ID") or not os.getenv("LIGHTNING_API_KEY"):
            raise ProviderAuthError(
                "Lightning requires LIGHTNING_USER_ID and LIGHTNING_API_KEY from a user "
                "API key. Set them locally; do not put credentials in provider options."
            )

    @staticmethod
    def _teamspace(options: LightningOptions) -> str:
        value = options.teamspace or os.getenv("LIGHTNING_TEAMSPACE") or ""
        if "/" not in value and os.getenv("LIGHTNING_ORG") and value:
            value = f"{os.environ['LIGHTNING_ORG']}/{value}"
        # Validate without propagating Pydantic input values in a public exception.
        parts = value.split("/")
        if len(parts) != 2 or not all(parts) or any(c.isspace() for c in value):
            raise ProviderAuthError(
                "Select a Lightning teamspace with LIGHTNING_TEAMSPACE=owner/teamspace "
                "or custom_args={'lightning': {'teamspace': 'owner/teamspace'}}."
            )
        return value

    @staticmethod
    def machine_name(request: DeploymentRequest, profile: ModelProfile) -> str:
        service = HardwareValidationService()
        count = request.num_gpus or profile.hardware.gpu_count
        candidates = [request.gpu_type] if request.gpu_type else profile.hardware.recommended_gpus
        for candidate in candidates:
            spec = service.get_gpu_spec(candidate or "")
            if spec is None or spec.name not in MACHINES or count not in (1, 2, 4, 8):
                continue
            if spec.name == "B200" and count not in (1, 8):
                continue
            service.validate_deployment_hardware(
                profile, request.model_copy(update={"gpu_type": spec.name}), strict=True
            )
            return MACHINES[spec.name] + (f"_X_{count}" if count > 1 else "")
        raise ProviderPlatformError(
            "Requested GPU/count cannot be represented on Lightning. Supported GPUs: "
            + ", ".join(MACHINES)
            + "; supported counts: 1, 2, 4, 8 (B200: 1 or 8). Account access is separate."
        )

    @staticmethod
    def _translate_error(operation: str, deployment_id: str | None, err: Exception) -> Exception:
        # SDK exceptions can contain request bodies/credentials. Never expose their text,
        # chained traceback or repr. Only a numeric HTTP status is safe to retain.
        status = getattr(err, "status", None)
        if status in (401, 403):
            return ProviderAuthError(
                f"Lightning {operation} was denied (HTTP {status}); check user key, "
                "teamspace permissions and machine entitlement."
            )
        return DeploymentError(
            f"Lightning {operation} failed; check account access, machine availability "
            "and the Lightning console. SDK error details are withheld to protect credentials.",
            deployment_id=deployment_id,
        )

    @staticmethod
    def _cli_sync(*args: str, missing_ok: bool = False) -> str | None:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "lightning_sdk.cli.entrypoint", *args],
                env=env, capture_output=True, text=True, encoding="utf-8",
                timeout=180, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise DeploymentError("Lightning CLI operation failed or timed out.") from None
        if result.returncode:
            # This exact message comes from the installed CLI's public resource resolver.
            missing_message = f"Deployment {args[2]!r} was not found." if len(args) > 2 else ""
            if missing_ok and missing_message and missing_message in result.stderr + result.stdout:
                return None
            raise DeploymentError(
                "Lightning CLI operation failed; check credentials, permissions and "
                "the Lightning console. Output withheld to protect credentials."
            ) from None
        return result.stdout

    async def verify_credentials(self) -> None:
        """Cheap read-only identity check; refuse scoped keys for ApiKeyAuth endpoints."""
        self._credentials()
        await _blocking(self._sdk)
        output = await _blocking(self._cli_sync, "auth", "whoami", "--json")
        try:
            identity = json.loads(output or "{}")
        except ValueError:
            raise ProviderAuthError("Lightning returned an invalid identity response.") from None
        if identity.get("auth_type") != "user":
            raise ProviderAuthError(
                "Lightning ApiKeyAuth requires a user API key; scoped API keys are not "
                "supported by this provider."
            )

    async def _save(self, record: DeploymentRecord) -> None:
        self._records[record.id] = record
        if self._repository:
            await self._repository.save(record)

    async def _record(self, deployment_id: str) -> DeploymentRecord:
        record = await self._repository.get(deployment_id) if self._repository else None
        record = record or self._records.get(deployment_id)
        if not record or record.provider != self.name or record.lightning is None:
            raise DeploymentNotFoundError(deployment_id)
        return record

    async def deploy(
        self, request: DeploymentRequest, profile: ModelProfile, runtime: RuntimeSpec,
    ) -> Deployment:
        machine_name = self.machine_name(request, profile)
        options = request.options or DeploymentOptions()
        teamspace = self._teamspace(options.provider.lightning) if not request.dry_run else (
            options.provider.lightning.teamspace or "dryrun/teamspace"
        )
        # Platform credentials must never become model environment variables or SDK
        # command-history arguments. Models do not need them.
        platform_keys = {"LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_AUTH_TOKEN"}
        credentials = {os.getenv(key) for key in platform_keys} - {None, ""}

        def contains_credentials(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    str(key).upper() in platform_keys or contains_credentials(item)
                    for key, item in value.items()
                )
            if isinstance(value, (list, tuple)):
                return any(contains_credentials(item) for item in value)
            return isinstance(value, str) and value in credentials

        if contains_credentials(options.model_dump()) or contains_credentials(runtime.env_vars):
            raise ProviderAuthError(
                "Lightning platform credentials must not be injected into runtimes or persisted options."
            )
        if options.provider.extra_provider_args:
            raise ProviderPlatformError(
                "Lightning accepts typed custom_args['lightning'] settings; provider_args "
                "pass-through is unsupported."
            )
        deployment_id = f"{self.resource_prefix}-{uuid.uuid4().hex}"
        record = DeploymentRecord(
            id=deployment_id, model=profile.id, provider=self.name,
            state=DeploymentState.PROVISIONING, options=options.model_copy(deep=True),
            workload_type=profile.workload_type, is_dry_run=request.dry_run,
            lightning=LightningDeploymentMetadata(name=deployment_id, teamspace=teamspace, owned=False),
        )
        record.options.cleanup_on_failure = True
        # The SDK persists these same effective options during lifecycle registration.
        options.cleanup_on_failure = True
        await self._save(record)
        if request.dry_run:
            record.endpoint_url = f"https://8080-{deployment_id}.cloudspaces.litng.ai"
            await self._save(record)
            return self._handle(record)

        remote = None
        try:
            self._credentials()
            await self.verify_credentials()
            sdk, config = await _blocking(self._sdk)
            machine = getattr(sdk.Machine, machine_name)
            remote = await _blocking(sdk.Deployment, deployment_id, teamspace=teamspace)
            if remote.is_started:
                raise DeploymentError("Lightning resource name collision; existing resource was preserved.")
            assert record.lightning is not None
            record.lightning.owned = True
            await self._save(record)
            scaling = options.provider.lightning
            await _blocking(
                remote.start, image=runtime.docker_image, machine=machine, ports=[runtime.port],
                entrypoint="/bin/sh", command=runtime_command(runtime), env=runtime.env_vars,
                include_credentials=False, spot=False, replicas=1,
                auth=config.ApiKeyAuth(),
                health_check=config.HttpHealthCheck(path=runtime.healthcheck_path, port=runtime.port),
                autoscale=config.AutoScaleConfig(
                    min_replicas=scaling.min_replicas, max_replicas=scaling.max_replicas,
                    metric="GPU", threshold=90,
                    idle_threshold_seconds=str(scaling.idle_threshold_seconds),
                ),
            )
            record.lightning.resource_id = remote.id
            deadline = asyncio.get_running_loop().time() + ENDPOINT_DISCOVERY_TIMEOUT_SECONDS
            while True:
                urls = await _blocking(lambda: remote.urls)
                if urls or asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(1)
            if not urls or len(urls) != 1:
                raise DeploymentError("Lightning did not return a single runtime endpoint.")
            record.endpoint_url = urls[0]
            parsed = urlsplit(record.endpoint_url)
            auth = self.endpoint_auth or default_endpoint_auth()
            if parsed.scheme != "https" or not auth.headers_for(self.name, record.endpoint_url):
                raise ProviderAuthError("Lightning endpoint has no safely configured HTTPS auth resolver.")
            record.state = DeploymentState.STARTING
            await self._save(record)
            return self._handle(record)
        except BaseException as err:
            if record.lightning and record.lightning.owned:
                try:
                    await self._destroy(record)
                except Exception:  # noqa: BLE001 - preserve the provisioning failure
                    logger.error("Lightning cleanup failed for %s; retry stop using this deployment ID.", record.id)
            record.mark_failed("Lightning provisioning failed.")
            try:
                await self._save(record)
            except Exception:  # noqa: BLE001 - preserve the provisioning failure
                logger.error("Could not persist Lightning failure for %s.", record.id)
            if not isinstance(err, Exception):
                raise
            if isinstance(err, (ProviderAuthError, ProviderPlatformError, DeploymentError, ImportError)):
                raise err from None
            raise self._translate_error("deploy", record.id, err) from None

    def _handle(self, record: DeploymentRecord) -> Deployment:
        return Deployment(
            status=self._status(record),
            stop_fn=lambda action=None: self.stop(record.id, action or AutostopAction.STOP),
            refresh_fn=lambda: self.get_status(record.id),
        )

    @staticmethod
    def _status(record: DeploymentRecord) -> DeploymentStatus:
        return DeploymentStatus(
            id=record.id, model=record.model, provider=record.provider, state=record.state,
            endpoint_url=record.endpoint_url, created_at=record.created_at,
            ready_at=record.ready_at, error_message=record.error_message,
        )

    async def _destroy(self, record: DeploymentRecord) -> None:
        metadata = record.lightning
        if not metadata or not metadata.owned:
            raise DeploymentError("Refusing to delete a Lightning resource not owned by InferWeave.")
        await _blocking(
            self._cli_sync, "deployment", "delete", metadata.resource_id or metadata.name,
            "--teamspace", metadata.teamspace, "--yes", missing_ok=True,
        )
        # Delete is asynchronous and can remain visible for minutes. Confirm absence
        # rather than equating an accepted request with stopped GPU replicas.
        deadline = asyncio.get_running_loop().time() + 300
        while await self.resource_snapshot(record) is not None:
            if asyncio.get_running_loop().time() >= deadline:
                raise DeploymentError(
                    "Lightning deletion was requested but removal is not confirmed; retry stop.",
                    deployment_id=record.id,
                )
            await asyncio.sleep(5)
        self._deleted_ids.add(record.id)

    async def resource_snapshot(self, record: DeploymentRecord) -> dict[str, Any] | None:
        """Read the live resource independently of cached STOPPED state (cleanup audit)."""
        metadata = record.lightning
        if not metadata:
            raise DeploymentNotFoundError(record.id)
        output = await _blocking(
            self._cli_sync, "deployment", "inspect", metadata.resource_id or metadata.name,
            "--teamspace", metadata.teamspace, "--json", missing_ok=True,
        )
        if output is None:
            return None
        try:
            data = json.loads(output)
            if not isinstance(data, dict):
                raise TypeError
            return data
        except (ValueError, TypeError):
            raise DeploymentError("Lightning returned an invalid deployment status.") from None

    async def stop(self, deployment_id: str, action: AutostopAction = AutostopAction.STOP) -> None:
        """Both stop/down fully delete the owned Deployment; scale-to-zero is separate."""
        record = await self._record(deployment_id)
        if record.id in self._deleted_ids or record.is_dry_run and record.state == DeploymentState.STOPPED:
            return
        if not record.is_dry_run:
            self._credentials()
            await self._destroy(record)
        record.mark_stopped()
        await self._save(record)

    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        record = await self._record(deployment_id)
        if record.is_dry_run or record.state == DeploymentState.STOPPED:
            return self._status(record)
        self._credentials()
        data = await self.resource_snapshot(record)
        if data is None:
            record.mark_stopped()
        else:
            try:
                status = data.get("status") or {}
                desired = str(data.get("desired_state", "")).upper()
                if data.get("deleted_at") or desired in ("STOPPED", "DELETED"):
                    record.mark_stopped()
                elif int(status.get("failing_replicas") or 0) > 0:
                    record.state = DeploymentState.FAILED
                elif int(status.get("pending_replicas") or 0) > 0:
                    record.state = DeploymentState.PROVISIONING
                else:
                    # Zero idle replicas remain a deployed, attachable endpoint.
                    record.state = DeploymentState.STARTING
            except (ValueError, TypeError, AttributeError):
                raise DeploymentError("Lightning returned an invalid deployment status.") from None
        await self._save(record)
        return self._status(record)
