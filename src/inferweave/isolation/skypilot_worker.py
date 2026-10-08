"""SkyPilot worker: runs one cluster operation for one account in its own process.

Self-contained (stdlib + ``skypilot`` only) so it can run under an isolated interpreter.
Protocol: one JSON request on stdin ``{"op", "payload", "secrets"}``; one JSON line on stdout
tagged ``"iw_worker"``.

Pooled accounts (``payload["mode"] == "account"``) run with ``HOME`` pointing at an
account-private directory prepared by the parent. SkyPilot only reads RunPod/Vast keys from files
under ``~`` and caches them for the lifetime of its API server process, so each account gets:

* its own credential file inside its private home (mode 0600, rewritten when the key rotates);
* its own SkyPilot API server on a dedicated localhost port (the auto-started server always uses
  port 46580 and would be shared by every account), restarted when the key rotates;
* its own SkyPilot state database (``~/.sky``) holding only that account's clusters.

The ambient account uses the user's own SkyPilot setup unchanged.
"""

import contextlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

SERVER_START_TIMEOUT = 120.0
SERVER_STOP_TIMEOUT = 30.0
SERVER_KILL_TIMEOUT = 10.0
posix_os: Any = os
posix_signal: Any = signal


class WorkerError(Exception):
    def __init__(
        self,
        kind: str,
        message: str,
        resource_may_exist: bool | None = None,
        status: int | None = None,
        retry_after: float | None = None,
        operation_may_continue: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.resource_may_exist = resource_may_exist
        self.status = status
        self.retry_after = retry_after
        self.operation_may_continue = operation_may_continue


def _emit(message: dict[str, Any]) -> None:
    print(json.dumps({"iw_worker": 1, **message}), flush=True)


_TEXT_KINDS = (
    (re.compile(r"\b401\b|unauthori[sz]ed|invalid api key", re.IGNORECASE), "auth", 401),
    (re.compile(r"\b403\b|forbidden|permission denied", re.IGNORECASE), "permission", 403),
    (re.compile(r"\b402\b|insufficient (funds|balance|credit)|balance is too low|quota exceeded", re.IGNORECASE), "quota", 402),
    (re.compile(r"\b429\b|rate.?limit|too many requests", re.IGNORECASE), "rate_limit", 429),
)


def classify(err: BaseException, operation: str) -> WorkerError:
    """Maps SkyPilot exceptions to a failure kind; their text never leaves the worker."""
    if isinstance(err, WorkerError):
        return err
    name = type(err).__name__
    if name == "ResourcesUnavailableError":
        # SkyPilot normally tears down failed attempts, but reconcile before failing over.
        return WorkerError("capacity", "No capacity for the requested resources.", True)
    if name == "NoCloudAccessError":
        return WorkerError("permission", "Cloud access is unavailable; invalid credentials were not confirmed.", False, 403)
    if name == "InvalidCloudCredentials":
        return WorkerError("auth", "The cloud rejected the account credentials.", False, 401)
    if name in ("ApiServerConnectionError", "ApiServerAuthenticationError"):
        return WorkerError(
            "transient", "The account's SkyPilot API server is unreachable.", operation == "launch"
        )
    if name in ("InvalidSkyPilotConfigError", "NotSupportedError", "InvalidClusterNameError"):
        return WorkerError("invalid_request", f"SkyPilot rejected the request ({name}).", False)
    text = str(err)
    status_code = getattr(err, "status_code", None) or getattr(err, "status", None) or getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(status_code, int):
        text = f"HTTP {status_code} {text}"
    headers = getattr(err, "headers", None) or getattr(getattr(err, "response", None), "headers", None)
    retry_after = None
    if headers:
        with contextlib.suppress(ValueError, TypeError, AttributeError, OverflowError):
            value = headers.get("Retry-After") or headers.get("retry-after")
            try:
                retry_after = max(0.0, float(value))
            except ValueError:
                retry_after = max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
    for pattern, kind, status in _TEXT_KINDS:
        if pattern.search(text):
            return WorkerError(kind, f"Provider rejected the request (HTTP {status}).", False, status, retry_after)
    return WorkerError(
        "transient",
        f"SkyPilot {operation} failed ({name}); details withheld to protect credentials.",
        True if operation == "launch" else None,
    )


# --- account isolation ---------------------------------------------------------------------


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def write_credentials(home: Path, cloud: str, api_key: str) -> None:
    """Places the key where SkyPilot's RunPod/Vast integrations read it, inside ``home``."""
    if cloud == "runpod":
        escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
        _write_private(home / ".runpod" / "config.toml", f'[default]\napi_key = "{escaped}"\n')
    elif cloud == "vast":
        _write_private(home / ".config" / "vastai" / "vast_api_key", api_key + "\n")
        _write_private(home / ".vast_api_key", api_key + "\n")
    else:
        raise WorkerError("invalid_request", f"Account pools are not supported for '{cloud}'.", False)


def _healthy(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint}/api/health", timeout=3) as response:
            return bool(response.status == 200)
    except Exception:  # noqa: BLE001
        return False


def _server_owned(state: Path, endpoint: str) -> bool:
    """Never reuse or kill a listener solely because its port happens to match."""
    try:
        psutil: Any = importlib.import_module("psutil")  # SkyPilot dependency, lazy

        process = psutil.Process(int((state / "server.pid").read_text().strip()))
        return (
            process.environ().get("HOME") == str(state.parent)
            and process.create_time() == float((state / "server-birth").read_text())
            and posix_os.getpgid(process.pid) == process.pid
        )
    except Exception:  # noqa: BLE001 - inability to verify ownership must fail closed
        return False


def _process_birth(pid: int) -> float:
    psutil: Any = importlib.import_module("psutil")
    return float(psutil.Process(pid).create_time())


def _group_processes(pid: int) -> list[Any]:
    """Live members of the private server group, including orphaned SDK workers."""
    psutil: Any = importlib.import_module("psutil")
    members = []
    for process in psutil.process_iter():
        with contextlib.suppress(psutil.NoSuchProcess, ProcessLookupError):
            if posix_os.getpgid(process.pid) == pid and process.status() != psutil.STATUS_ZOMBIE:
                members.append(process)
    return members


def _wait_server_stopped(pid: int, endpoint: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_processes(pid) or _healthy(endpoint):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)
    return True


def _group_still_owned(members: list[Any], pid: int) -> bool:
    """An original process identity must survive before escalating a group signal.

    psutil.is_running() checks process birth time, protecting against PID/group reuse
    even if the original group leader exited while its children kept running.
    """
    for process in members:
        with contextlib.suppress(ProcessLookupError):
            if process.is_running() and posix_os.getpgid(process.pid) == pid:
                return True
    return False


def _stop_server(state: Path, endpoint: str) -> None:
    if not _server_owned(state, endpoint):
        raise WorkerError("transient", "Cannot verify ownership of the SkyPilot API server.", False)
    pid_file = state / "server.pid"
    pid = int(pid_file.read_text().strip())
    if posix_os.getpgid(pid) != pid:
        raise WorkerError("transient", "SkyPilot server process group is not private.", False)
    members = _group_processes(pid)
    with contextlib.suppress(ProcessLookupError):
        posix_os.killpg(pid, signal.SIGTERM)
    # SkyPilot's graceful drain can outlast our deadline, and idle request workers
    # ignore SIGTERM. Losing /api/health alone does not establish server termination.
    if not _wait_server_stopped(pid, endpoint, SERVER_STOP_TIMEOUT):
        if not _group_still_owned(members, pid):
            raise WorkerError("transient", "Cannot verify ownership of the SkyPilot API server group.", False)
        with contextlib.suppress(ProcessLookupError):
            posix_os.killpg(pid, posix_signal.SIGKILL)
        if not _wait_server_stopped(pid, endpoint, SERVER_KILL_TIMEOUT):
            raise WorkerError("transient", "SkyPilot API server termination was not confirmed.", False)
    with contextlib.suppress(OSError):
        pid_file.unlink()


@contextlib.contextmanager
def _server_lock(state: Path) -> Iterator[None]:
    """Serializes server start/restart decisions of concurrent workers of one account."""
    try:
        fcntl: Any = importlib.import_module("fcntl")
    except ImportError:  # SkyPilot itself only runs on POSIX hosts
        yield
        return
    with open(state / "server.lock", "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_account_server(
    home: Path, cloud: str, api_key: str, port: int, fingerprint: str
) -> str:
    """Starts (or reuses) this account's API server; restarts it when the key rotated."""
    state = home / ".inferweave"
    state.mkdir(parents=True, exist_ok=True)
    with _server_lock(state):
        return _ensure_account_server_locked(home, state, cloud, api_key, port, fingerprint)


def _ensure_account_server_locked(
    home: Path, state: Path, cloud: str, api_key: str, port: int, fingerprint: str
) -> str:
    endpoint = f"http://127.0.0.1:{port}"
    fingerprint_file = state / "server-fingerprint"
    current = fingerprint_file.read_text().strip() if fingerprint_file.exists() else ""
    if _healthy(endpoint):
        if not _server_owned(state, endpoint):
            raise WorkerError("transient", "The SkyPilot account port is occupied by another process.", False)
        if current == fingerprint:
            return endpoint
        _stop_server(state, endpoint)  # it cached the previous key
    write_credentials(home, cloud, api_key)
    _write_private(
        home / ".sky" / "plugins.yaml",
        "plugins:\n  - class: skypilot_queue_plugin.AccountQueuePlugin\n"
        f"    parameters:\n      port: {port + 2}\n",
    )
    server_env = dict(os.environ)
    server_env["IS_SKYPILOT_SERVER"] = "true"
    server_env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    server_env["SKYPILOT_DISABLE_USAGE_COLLECTION"] = "1"
    if cloud == "runpod":
        server_env["RUNPOD_API_KEY"] = api_key  # SkyPilot's adaptor fallback
    # Raw SDK output can contain cloud credentials. Discard it rather than persisting it.
    process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sky.server.server",
                "--host",
                "127.0.0.1",
                f"--port={port}",
                f"--metrics-port={port + 1}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=server_env,
            start_new_session=True,
    )
    (state / "server.pid").write_text(str(process.pid))
    (state / "server-birth").write_text(str(_process_birth(process.pid)))
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while not _healthy(endpoint):
        if process.poll() is not None or time.monotonic() >= deadline:
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    posix_os.killpg(process.pid, signal.SIGTERM)
            raise WorkerError("transient", "The account's SkyPilot API server did not start.", False)
        time.sleep(1)
    fingerprint_file.write_text(fingerprint)
    for stale in state.glob("checked-*"):
        stale.unlink()
    return endpoint


def _ensure_cloud_enabled(sky: Any, home: Path, cloud: str) -> None:
    marker = home / ".inferweave" / f"checked-{cloud}"
    if marker.exists():
        return
    sky.get(sky.check(infra_list=(cloud,), verbose=False))
    marker.write_text("1")


# --- operations ----------------------------------------------------------------------------


def _endpoint_of(sky: Any, cluster: str, port: int | None) -> str | None:
    try:
        endpoints = sky.get(sky.endpoints(cluster, port=port) if port else sky.endpoints(cluster))
    except Exception:  # noqa: BLE001 - endpoints are not ready yet
        return None
    if not endpoints:
        return None
    endpoint = endpoints.get(port) if port else None
    endpoint = endpoint or next(iter(endpoints.values()), None)
    if not endpoint:
        return None
    return endpoint if str(endpoint).startswith("http") else f"http://{endpoint}"


def op_launch(sky: Any, payload: dict[str, Any]) -> dict[str, Any]:
    task = sky.Task(
        name=payload["cluster"],
        setup=payload.get("setup"),
        run=payload["run"],
        envs=payload.get("envs") or {},
        workdir=payload.get("workdir"),
    )
    resources = dict(payload["resources"])
    cloud = resources.pop("cloud")
    registry = getattr(sky.clouds, "CLOUD_REGISTRY", None)
    resources["cloud"] = registry.from_str(cloud) if registry is not None else None
    task.set_resources(sky.Resources(**resources))
    try:
        sky.get(
            sky.launch(
                task,
                cluster_name=payload["cluster"],
                idle_minutes_to_autostop=payload.get("idle_minutes"),
                down=bool(payload.get("down")),
                stream_logs=False,
            )
        )
    except Exception as exc:  # noqa: BLE001 - launch spans several asynchronous API operations
        error = classify(exc, "launch")
        error.resource_may_exist = True
        error.operation_may_continue = True
        raise error from None
    return {"endpoint": _endpoint_of(sky, payload["cluster"], payload.get("port"))}


def op_status(sky: Any, payload: dict[str, Any]) -> dict[str, Any]:
    clusters = sky.get(sky.status(cluster_names=[payload["cluster"]]))
    if not clusters:
        return {"exists": False, "status": None, "endpoint": None}
    record = clusters[0]
    raw = getattr(record, "status", None)
    if raw is None and isinstance(record, dict):
        raw = record.get("status")
    status = str(getattr(raw, "value", raw)).upper()
    endpoint = _endpoint_of(sky, payload["cluster"], payload.get("port")) if "UP" in status else None
    return {"exists": True, "status": status, "endpoint": endpoint}


def _teardown(sky: Any, payload: dict[str, Any], down: bool) -> dict[str, Any]:
    try:
        call = sky.down if down else sky.stop
        sky.get(call(cluster_name=payload["cluster"]))
    except Exception as exc:
        if type(exc).__name__ == "ClusterDoesNotExist":
            return {"existed": False}
        raise
    return {"existed": True}


OPERATIONS = {
    # Network-free check that this interpreter can run the worker (used by CI and doctors).
    "probe": lambda sky, payload: {"sdk": str(getattr(sky, "__version__", "unknown"))},
    "server_probe": lambda sky, payload: {"sdk": str(getattr(sky, "__version__", "unknown"))},
    "launch": op_launch,
    "status": op_status,
    "stop": lambda sky, payload: _teardown(sky, payload, down=False),
    "down": lambda sky, payload: _teardown(sky, payload, down=True),
}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        operation_name = request["op"]
        operation = OPERATIONS[operation_name]
        payload = request.get("payload") or {}
    except (ValueError, KeyError):
        _emit({"ok": False, "error": {"kind": "invalid_request", "message": "bad worker request"}})
        return 2
    try:
        home = Path.home()
        account_mode = payload.get("mode") == "account" and operation_name != "probe"
        if account_mode:
            api_key = (request.get("secrets") or {}).get("api_key") or ""
            if not api_key:
                raise WorkerError("auth", "The account has no API key.", False)
            endpoint = ensure_account_server(
                home, payload["cloud"], api_key, int(payload["server_port"]), payload["fingerprint"]
            )
            os.environ["SKYPILOT_API_SERVER_ENDPOINT"] = endpoint
        try:
            import sky  # type: ignore
        except ImportError:
            raise WorkerError(
                "invalid_request", "SkyPilot is not installed in the SkyPilot worker interpreter.", False
            ) from None
        if account_mode and operation_name != "server_probe":
            _ensure_cloud_enabled(sky, home, payload["cloud"])
        result = operation(sky, payload)
        if operation_name == "server_probe" and account_mode:
            time.sleep(float(payload.get("hold_seconds", 0)))
            _stop_server(home / ".inferweave", endpoint)
    except Exception as exc:  # noqa: BLE001 - classified without its text
        error = classify(exc, operation_name)
        _emit(
            {
                "ok": False,
                "error": {
                    "kind": error.kind,
                    "status": error.status,
                    "message": str(error),
                    "resource_may_exist": error.resource_may_exist,
                    "retry_after": error.retry_after,
                    "operation_may_continue": error.operation_may_continue,
                },
            }
        )
        return 1
    _emit({"ok": True, "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
