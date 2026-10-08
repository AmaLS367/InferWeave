"""Lightning container Deployments, one isolated worker process per account operation.

The Lightning SDK authenticates once per process from ``LIGHTNING_USER_ID``/``LIGHTNING_API_KEY``
and offers no per-client credentials, so every operation runs in
``inferweave/isolation/lightning_worker.py`` with only the owning account's credentials (sent on
stdin) and an account-private home. The worker may use a separate interpreter
(``INFERWEAVE_LIGHTNING_PYTHON`` or ``LightningProvider(python=...)``), which lets Lightning SDK
live in its own virtual environment next to SkyPilot.
"""

import asyncio
import base64
import hashlib
import io
import os
import shlex
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from inferweave.accounts.models import ProviderAccount
from inferweave.adapters.auth import default_endpoint_auth
from inferweave.core.exceptions import (
    ProviderAuthError,
    ProviderOperationError,
    ProviderPlatformError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions
from inferweave.isolation import (
    WORKER_DIR,
    WorkerRunner,
    account_environment,
    ambient_environment,
)
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.providers.base import ComputeProvider, ProvisionResult, ResourceStatus
from inferweave.runtimes.base import RuntimeSpec
from inferweave.services.hardware_service import HardwareValidationService

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
DELETE_CONFIRM_TIMEOUT_SECONDS = 300.0
PYTHON_ENV = "INFERWEAVE_LIGHTNING_PYTHON"
PLATFORM_KEYS = frozenset({"LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_AUTH_TOKEN"})


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


def _valid_teamspace(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not any(c.isspace() for c in value)


class LightningProvider(ComputeProvider):
    """Owns one Lightning Deployment per InferWeave deployment; creates no Studios."""

    cleanup_failed_deployment = True

    def __init__(
        self,
        endpoint_auth: EndpointAuthPort | None = None,
        resource_prefix: str = "iw-lightning",
        python: str | None = None,
        state_dir: Path | None = None,
        runner: WorkerRunner | None = None,
    ) -> None:
        self.endpoint_auth = endpoint_auth
        if not resource_prefix.startswith("iw-lightning") or not all(
            c.isalnum() or c == "-" for c in resource_prefix
        ):
            raise ValueError(
                "Lightning resource prefix must start with iw-lightning and use letters/digits/hyphens."
            )
        self.resource_prefix = resource_prefix
        self.state_dir = state_dir or Path.home() / ".inferweave" / "accounts"
        self.runner = runner or WorkerRunner(
            "lightning",
            WORKER_DIR / "lightning_worker.py",
            python=python or os.environ.get(PYTHON_ENV),
        )

    @property
    def name(self) -> str:
        return "lightning"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LIGHTNING

    # --- configuration -----------------------------------------------------------------------

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

    def teamspace(self, request: DeploymentRequest, account: ProviderAccount) -> str:
        """Per-deploy teamspace, else the account's teamspace, else ``LIGHTNING_TEAMSPACE``."""
        options = (request.options or DeploymentOptions()).provider.lightning
        value = options.teamspace or account.get_metadata("teamspace") or ""
        if not value and account.is_ambient:
            value = os.getenv("LIGHTNING_TEAMSPACE") or ""
            if value and "/" not in value and os.getenv("LIGHTNING_ORG"):
                value = f"{os.environ['LIGHTNING_ORG']}/{value}"
        if request.dry_run and not value:
            return "dryrun/teamspace"
        if not _valid_teamspace(value):
            raise ProviderAuthError(
                "Select a Lightning teamspace (owner/teamspace) with the account's 'teamspace' "
                "metadata, LIGHTNING_TEAMSPACE for ambient credentials, or "
                "custom_args={'lightning': {'teamspace': 'owner/teamspace'}}."
            )
        return value

    def new_deployment_id(self, profile: ModelProfile) -> str:
        return f"{self.resource_prefix}-{uuid.uuid4().hex}"

    def resource_ref(
        self, deployment_id: str, request: DeploymentRequest, account: ProviderAccount
    ) -> ResourceRef:
        # Not owned until the worker confirmed the name was free (no pre-existing resource).
        return ResourceRef(name=deployment_id, scope=self.teamspace(request, account), owned=False)

    def dry_run_endpoint(self, deployment_id: str, runtime: RuntimeSpec) -> str:
        return f"https://8080-{deployment_id}.cloudspaces.litng.ai"

    def preflight(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> None:
        self.machine_name(request, profile)
        self.teamspace(request, account)
        options = request.options or DeploymentOptions()
        if options.provider.extra_provider_args:
            raise ProviderPlatformError(
                "Lightning accepts typed custom_args['lightning'] settings; provider_args "
                "pass-through is unsupported."
            )
        if request.dry_run:
            return
        self._require_credentials(account)
        auth = self.endpoint_auth or default_endpoint_auth()
        if not auth.is_configured_for(self.name, account):
            raise ProviderAuthError(
                "Lightning ApiKeyAuth endpoints need the account's user API key for readiness "
                "probes and inference; no endpoint auth resolver is configured."
            )

    @staticmethod
    def _require_credentials(account: ProviderAccount) -> None:
        if account.is_ambient:
            if not os.getenv("LIGHTNING_USER_ID") or not os.getenv("LIGHTNING_API_KEY"):
                raise ProviderAuthError(
                    "Lightning requires LIGHTNING_USER_ID and LIGHTNING_API_KEY from a user "
                    "API key (or a configured Lightning account pool). Do not put credentials "
                    "in provider options."
                )
            return
        account.secret("user_id")
        account.secret("api_key")

    # --- worker plumbing -----------------------------------------------------------------------

    def _account_home(self, account: ProviderAccount) -> Path:
        digest = hashlib.sha256(f"{account.id}/{account.owner_fingerprint}".encode()).hexdigest()[:16]
        return self.state_dir / "lightning" / digest

    async def _run(
        self,
        operation: str,
        payload: dict[str, Any],
        account: ProviderAccount,
        *,
        timeout: float = 240.0,
        deployment_id: str | None = None,
    ) -> Any:
        if account.is_ambient:
            env = ambient_environment()
            secrets: dict[str, str] = {}
        else:
            home = self._account_home(account)
            # The SDK must never fall back to the user's ~/.lightning/credentials.json.
            env = account_environment(
                home, {"LIGHTNING_CREDENTIAL_PATH": str(home / ".lightning" / "none.json")}
            )
            secrets = {
                "LIGHTNING_USER_ID": account.secret("user_id"),
                "LIGHTNING_API_KEY": account.secret("api_key"),
            }
        return await self.runner.run(
            operation,
            payload,
            env=env,
            secrets=secrets,
            timeout=timeout,
            deployment_id=deployment_id,
            account_id=account.id,
        )

    async def verify_credentials(self, account: ProviderAccount) -> None:
        """Cheap read-only identity check; refuse scoped keys for ApiKeyAuth endpoints."""
        identity = await self._run("whoami", {}, account, timeout=120)
        if not isinstance(identity, dict) or identity.get("auth_type") != "user":
            raise ProviderOperationError(
                "Lightning ApiKeyAuth requires a user API key; scoped API keys are not "
                "supported by this provider.",
                FailureKind.PERMISSION,
                resource_may_exist=False,
            )

    # --- lifecycle ---------------------------------------------------------------------------

    async def prepare(self, record: DeploymentRecord, account: ProviderAccount) -> None:
        await self.verify_credentials(account)
        if await self.resource_snapshot(record, account) is not None:
            raise ProviderOperationError(
                "Lightning resource name collision; the existing resource was preserved.",
                FailureKind.INVALID_REQUEST, resource_may_exist=False, deployment_id=record.id,
            )
        assert record.resource is not None
        record.resource.owned = True

    async def provision(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> ProvisionResult:
        assert record.resource is not None
        try:
            await self.verify_credentials(account)
        except ProviderOperationError as err:
            err.resource_may_exist = False  # nothing was created yet
            raise
        scaling = (request.options or DeploymentOptions()).provider.lightning
        start = self._run(
            "start",
            {
                "name": record.resource.name,
                "teamspace": record.resource.scope,
                "machine": self.machine_name(request, profile),
                "image": runtime.docker_image,
                "port": runtime.port,
                "command": runtime_command(runtime),
                "env": dict(runtime.env_vars),
                "healthcheck_path": runtime.healthcheck_path,
                "min_replicas": scaling.min_replicas,
                "max_replicas": scaling.max_replicas,
                "idle_threshold_seconds": scaling.idle_threshold_seconds,
                "discovery_timeout": ENDPOINT_DISCOVERY_TIMEOUT_SECONDS,
            },
            account,
            timeout=600,
            deployment_id=record.id,
        )
        try:
            result = await start
        except ProviderOperationError as err:
            # The worker reports a name collision definitively (resource_may_exist=False);
            # any other uncertain failure happened after the name was confirmed free.
            if err.resource_may_exist:
                record.resource.owned = True
            else:
                record.resource.owned = False
            raise
        except BaseException:
            # Cancelled or crashed after the worker may have created the resource: it is
            # ours, so cleanup must be allowed to delete it.
            record.resource.owned = True
            raise
        record.resource.owned = True
        record.creation_may_continue = False
        urls = result.get("urls") if isinstance(result, dict) else None
        if not urls or len(urls) != 1:
            raise ProviderOperationError(
                "Lightning did not return a single runtime endpoint.",
                FailureKind.TRANSIENT,
                resource_may_exist=True,
                deployment_id=record.id,
            )
        endpoint_url = str(urls[0])
        auth = self.endpoint_auth or default_endpoint_auth()
        if urlsplit(endpoint_url).scheme != "https" or not auth.headers_for(
            self.name, endpoint_url, account
        ):
            raise ProviderOperationError(
                "Lightning endpoint has no safely configured HTTPS auth resolver.",
                FailureKind.INVALID_REQUEST,
                resource_may_exist=True,
                deployment_id=record.id,
            )
        return ProvisionResult(
            state=DeploymentState.STARTING,
            endpoint_url=endpoint_url,
            resource_id=result.get("resource_id"),
        )

    def _target(self, record: DeploymentRecord) -> dict[str, Any]:
        resource = record.resource
        if resource is None or not resource.scope:
            raise ProviderOperationError(
                f"Deployment '{record.id}' has no Lightning resource identity.",
                FailureKind.INVALID_REQUEST,
                resource_may_exist=False,
                deployment_id=record.id,
            )
        return {"target": resource.resource_id or resource.name, "teamspace": resource.scope}

    async def resource_snapshot(
        self, record: DeploymentRecord, account: ProviderAccount
    ) -> dict[str, Any] | None:
        """Read the live resource independently of cached STOPPED state (cleanup audit)."""
        data = await self._run("inspect", self._target(record), account, deployment_id=record.id)
        if data is None or isinstance(data, dict):
            return data
        raise ProviderOperationError("Lightning returned an invalid resource snapshot.", FailureKind.TRANSIENT)

    async def resource_exists(self, record: DeploymentRecord, account: ProviderAccount) -> bool:
        return await self.resource_snapshot(record, account) is not None

    async def stop(
        self,
        record: DeploymentRecord,
        account: ProviderAccount,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Both stop/down fully delete the owned Deployment; scale-to-zero is separate."""
        if record.resource is not None and not record.resource.owned:
            # The name was never confirmed free, so the resource may belong to someone else.
            if await self.resource_exists(record, account):
                raise ProviderOperationError(
                    "Refusing to delete a Lightning resource not owned by InferWeave.",
                    FailureKind.INVALID_REQUEST,
                    resource_may_exist=True,
                    deployment_id=record.id,
                )
            return
        await self._run("delete", self._target(record), account, deployment_id=record.id)
        # Delete is asynchronous and can remain visible for minutes. Confirm absence
        # rather than equating an accepted request with stopped GPU replicas.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + DELETE_CONFIRM_TIMEOUT_SECONDS
        while await self.resource_snapshot(record, account) is not None:
            if loop.time() >= deadline:
                raise ProviderOperationError(
                    "Lightning deletion was requested but removal is not confirmed; retry stop.",
                    FailureKind.TRANSIENT,
                    resource_may_exist=True,
                    deployment_id=record.id,
                )
            await asyncio.sleep(5)

    async def status(self, record: DeploymentRecord, account: ProviderAccount) -> ResourceStatus:
        data = await self.resource_snapshot(record, account)
        if data is None:
            return ResourceStatus(state=DeploymentState.STOPPED, endpoint_url=record.endpoint_url)
        try:
            status = data.get("status") or {}
            desired = str(data.get("desired_state", "")).upper().removeprefix("DEPLOYMENT_STATE_")
            if data.get("deleted_at") or desired in ("STOPPED", "DELETED", "FROZEN", "BALANCE_STOPPED"):
                state = DeploymentState.STOPPED
            elif desired in ("FAILED", "SHADOW_BANNED") or int(status.get("failing_replicas") or 0) > 0:
                state = DeploymentState.FAILED
            elif desired == "PENDING" or int(status.get("pending_replicas") or 0) > 0:
                state = DeploymentState.PROVISIONING
            else:
                # Zero idle replicas remain a deployed, attachable endpoint.
                state = DeploymentState.STARTING
        except (ValueError, TypeError, AttributeError):
            raise ProviderOperationError(
                "Lightning returned an invalid deployment status.",
                FailureKind.TRANSIENT,
                deployment_id=record.id,
            ) from None
        return ResourceStatus(state=state, endpoint_url=record.endpoint_url)
