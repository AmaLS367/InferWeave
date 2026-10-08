"""Tests for the self-contained SkyPilot worker script (``isolation/skypilot_worker.py``).

The worker is imported as a module. SkyPilot itself is replaced by a fake ``sky`` module in
``sys.modules``; the account API server (``_healthy`` / ``subprocess.Popen``) is faked too, so no
subprocess, network or real SkyPilot is involved.
"""

import io
import json
import os
import signal
import stat
import subprocess
import sys
import time
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

from inferweave.isolation import skypilot_worker as worker


class ResourcesUnavailableError(Exception):
    pass


class NoCloudAccessError(Exception):
    pass


class InvalidCloudCredentials(Exception):
    pass


class ApiServerConnectionError(Exception):
    pass


class InvalidSkyPilotConfigError(Exception):
    pass


class ClusterDoesNotExist(Exception):
    pass


SECRET = "SENTINEL-skypilot-secret"


# --- classify ----------------------------------------------------------------------------------


def test_classify_capacity_means_resource_may_exist():
    error = worker.classify(ResourcesUnavailableError(f"no GPUs {SECRET}"), "launch")

    assert error.kind == "capacity"
    assert error.resource_may_exist is True
    assert SECRET not in str(error)


@pytest.mark.parametrize("exc_class", [InvalidCloudCredentials])
def test_classify_cloud_access_errors_are_auth(exc_class):
    error = worker.classify(exc_class(f"key {SECRET} rejected"), "launch")

    assert error.kind == "auth"
    assert error.status == 401
    assert error.resource_may_exist is False
    assert SECRET not in str(error)


@pytest.mark.parametrize(
    ("operation", "may_exist"), [("launch", True), ("status", False), ("down", False)]
)
def test_classify_api_server_errors_are_transient(operation, may_exist):
    error = worker.classify(ApiServerConnectionError("down"), operation)

    assert error.kind == "transient"
    assert error.resource_may_exist is may_exist


@pytest.mark.parametrize("exc_class", [InvalidSkyPilotConfigError])
def test_classify_config_errors_are_invalid_request(exc_class):
    error = worker.classify(exc_class(SECRET), "launch")

    assert error.kind == "invalid_request"
    assert error.resource_may_exist is False
    assert SECRET not in str(error)


@pytest.mark.parametrize(
    ("text", "kind", "status"),
    [
        ("HTTP 401 from server", "auth", 401),
        ("Unauthorized request", "auth", 401),
        ("Invalid API key supplied", "auth", 401),
        ("403 Forbidden", "permission", 403),
        ("permission denied for resource", "permission", 403),
        ("402 Payment required", "quota", 402),
        ("Insufficient funds on account", "quota", 402),
        ("balance is too low", "quota", 402),
        ("429 Too Many Requests", "rate_limit", 429),
        ("rate limit exceeded", "rate_limit", 429),
    ],
)
def test_classify_text_heuristics(text, kind, status):
    error = worker.classify(RuntimeError(f"{text} key={SECRET}"), "launch")

    assert error.kind == kind
    assert error.status == status
    assert error.resource_may_exist is False
    assert SECRET not in str(error)


def test_classify_unknown_launch_error_is_transient_and_may_exist():
    error = worker.classify(RuntimeError(f"kaboom {SECRET}"), "launch")

    assert error.kind == "transient"
    assert error.resource_may_exist is True
    assert "kaboom" not in str(error)
    assert SECRET not in str(error)
    assert "RuntimeError" in str(error)


@pytest.mark.parametrize("operation", ["status", "stop", "down"])
def test_classify_unknown_non_launch_error_leaves_existence_unknown(operation):
    error = worker.classify(RuntimeError("kaboom"), operation)

    assert error.kind == "transient"
    assert error.resource_may_exist is None


def test_classify_passes_worker_errors_through():
    original = worker.WorkerError("auth", "nope", False, 401)

    assert worker.classify(original, "launch") is original


# --- write_credentials -------------------------------------------------------------------------


def test_write_credentials_runpod_config_toml(tmp_path):
    worker.write_credentials(tmp_path, "runpod", "rp_plain_key")

    config = tmp_path / ".runpod" / "config.toml"
    assert config.read_text(encoding="utf-8") == '[default]\napi_key = "rp_plain_key"\n'
    assert tomllib.loads(config.read_text(encoding="utf-8"))["default"]["api_key"] == "rp_plain_key"


def test_write_credentials_runpod_quotes_special_characters(tmp_path):
    key = 'a"b\\c\'d'

    worker.write_credentials(tmp_path, "runpod", key)

    parsed = tomllib.loads((tmp_path / ".runpod" / "config.toml").read_text(encoding="utf-8"))
    assert parsed == {"default": {"api_key": key}}


def test_write_credentials_runpod_rotation_overwrites_and_leaves_no_temp(tmp_path):
    worker.write_credentials(tmp_path, "runpod", "first")
    worker.write_credentials(tmp_path, "runpod", "second")

    directory = tmp_path / ".runpod"
    assert 'api_key = "second"' in (directory / "config.toml").read_text(encoding="utf-8")
    assert sorted(p.name for p in directory.iterdir()) == ["config.toml"]


def test_write_credentials_vast_files(tmp_path):
    worker.write_credentials(tmp_path, "vast", "vast_key")

    assert (tmp_path / ".config" / "vastai" / "vast_api_key").read_text(encoding="utf-8") == "vast_key\n"
    assert (tmp_path / ".vast_api_key").read_text(encoding="utf-8") == "vast_key\n"


def test_write_credentials_rejects_clouds_without_pools(tmp_path):
    with pytest.raises(worker.WorkerError) as excinfo:
        worker.write_credentials(tmp_path, "aws", "key")

    assert excinfo.value.kind == "invalid_request"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
@pytest.mark.parametrize("cloud", ["runpod", "vast"])
def test_write_credentials_files_are_private(tmp_path, cloud):
    worker.write_credentials(tmp_path, cloud, "key")

    files = [
        path for path in tmp_path.rglob("*") if path.is_file()
    ]
    assert files
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, path.parent


# --- ensure_account_server ---------------------------------------------------------------------


class _Process:
    pid = 4242

    def __init__(self, exit_code: int | None) -> None:
        self._exit_code = exit_code

    def poll(self) -> int | None:
        return self._exit_code


class ServerHarness:
    """Fakes the account API server: health endpoint, stop, and ``subprocess.Popen``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, up: bool, start_fails: bool = False) -> None:
        self.up = up
        self.start_fails = start_fails
        self.popen_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.credential_files_at_start: dict[str, str] = {}
        self.stopped: list[tuple[Path, str]] = []
        self.home: Path | None = None
        monkeypatch.setattr(worker, "_process_birth", lambda pid: 123.0)
        monkeypatch.setattr(worker, "_healthy", lambda endpoint: self.up)
        monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: True)
        monkeypatch.setattr(worker, "_stop_server", self._stop)
        monkeypatch.setattr(
            worker,
            "subprocess",
            types.SimpleNamespace(
                Popen=self._popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT
            ),
        )
        monkeypatch.setattr(
            worker,
            "time",
            types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: None),
        )

    def _stop(self, state: Path, endpoint: str) -> None:
        self.stopped.append((state, endpoint))
        self.up = False

    def _popen(self, argv: list[str], **kwargs: Any) -> _Process:
        self.popen_calls.append((argv, kwargs))
        if self.home is not None:
            config = self.home / ".runpod" / "config.toml"
            if config.exists():
                self.credential_files_at_start["runpod"] = config.read_text(encoding="utf-8")
        if self.start_fails:
            return _Process(exit_code=1)
        self.up = True
        return _Process(exit_code=None)


def _ensure(home: Path, cloud: str = "runpod", key: str = SECRET, port: int = 47010, fp: str = "fp-1"):
    return worker.ensure_account_server(home, cloud, key, port, fp)


def test_ensure_account_server_reuses_healthy_server_with_same_fingerprint(monkeypatch, tmp_path):
    harness = ServerHarness(monkeypatch, up=True)
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server-fingerprint").write_text("fp-1")

    endpoint = _ensure(tmp_path)

    assert endpoint == "http://127.0.0.1:47010"
    assert harness.popen_calls == []
    assert harness.stopped == []
    assert not (tmp_path / ".runpod").exists()  # credentials were not rewritten


def test_ensure_account_server_restarts_when_key_rotated(monkeypatch, tmp_path):
    harness = ServerHarness(monkeypatch, up=True)
    harness.home = tmp_path
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server-fingerprint").write_text("fp-old")
    (state / "checked-runpod").write_text("1")

    endpoint = _ensure(tmp_path, key="SENTINEL-rotated", fp="fp-new")

    assert endpoint == "http://127.0.0.1:47010"
    assert harness.stopped == [(state, endpoint)]  # the old server cached the previous key
    assert len(harness.popen_calls) == 1
    # The new key was on disk before the new server started.
    assert "SENTINEL-rotated" in harness.credential_files_at_start["runpod"]
    assert (state / "server-fingerprint").read_text() == "fp-new"
    assert (state / "server.pid").read_text() == "4242"
    assert not list(state.glob("checked-*"))  # cloud checks are redone for the new key


def test_ensure_account_server_restarts_when_fingerprint_is_unknown(monkeypatch, tmp_path):
    harness = ServerHarness(monkeypatch, up=True)

    _ensure(tmp_path)

    assert len(harness.stopped) == 1
    assert len(harness.popen_calls) == 1


def test_ensure_account_server_starts_server_without_secrets_in_argv(monkeypatch, tmp_path):
    harness = ServerHarness(monkeypatch, up=False)
    state = tmp_path / ".inferweave"

    endpoint = _ensure(tmp_path, key=SECRET, port=47010, fp="fp-1")

    assert endpoint == "http://127.0.0.1:47010"
    assert harness.stopped == []
    ((argv, kwargs),) = harness.popen_calls
    assert argv == [
        sys.executable,
        "-m",
        "sky.server.server",
        "--host",
        "127.0.0.1",
        "--port=47010",
        "--metrics-port=47011",
    ]
    assert not any(SECRET in part for part in argv)
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["IS_SKYPILOT_SERVER"] == "true"
    assert (state / "server-fingerprint").read_text() == "fp-1"
    assert (state / "server.pid").read_text() == "4242"
    assert "port: 47012" in (tmp_path / ".sky" / "plugins.yaml").read_text()
    assert kwargs["env"]["PYTHONPATH"] == str(Path(worker.__file__).resolve().parent)
    assert not (tmp_path / ".sky" / "api_server" / "server.log").exists()
    parsed = tomllib.loads((tmp_path / ".runpod" / "config.toml").read_text(encoding="utf-8"))
    assert parsed["default"]["api_key"] == SECRET


def test_ensure_account_server_passes_runpod_key_only_to_runpod_servers(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    runpod = ServerHarness(monkeypatch, up=False)
    _ensure(tmp_path / "rp", cloud="runpod", key=SECRET)
    assert runpod.popen_calls[0][1]["env"]["RUNPOD_API_KEY"] == SECRET

    vast = ServerHarness(monkeypatch, up=False)
    _ensure(tmp_path / "vast", cloud="vast", key=SECRET)
    vast_env = vast.popen_calls[0][1]["env"]
    assert "RUNPOD_API_KEY" not in vast_env
    assert "VAST_API_KEY" not in vast_env
    assert SECRET not in vast_env.values()
    assert (tmp_path / "vast" / ".vast_api_key").read_text(encoding="utf-8") == f"{SECRET}\n"


def test_ensure_account_server_start_failure_is_a_transient_worker_error(monkeypatch, tmp_path):
    ServerHarness(monkeypatch, up=False, start_fails=True)

    with pytest.raises(worker.WorkerError) as excinfo:
        _ensure(tmp_path)

    assert excinfo.value.kind == "transient"
    assert excinfo.value.resource_may_exist is False
    assert not (tmp_path / ".inferweave" / "server-fingerprint").exists()
    assert SECRET not in str(excinfo.value)


def test_stop_server_terminates_pid_and_waits_for_health_to_drop(monkeypatch, tmp_path):
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server.pid").write_text("31337")
    killed: list[tuple[int, int]] = []
    health = iter([True, True, False])
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False)
    monkeypatch.setattr(worker, "_healthy", lambda endpoint: next(health))
    monkeypatch.setattr(worker, "_group_processes", lambda pid: [])
    monkeypatch.setattr(
        worker, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: None)
    )

    monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: True)
    worker._stop_server(state, "http://127.0.0.1:47010")

    assert killed == [(31337, signal.SIGTERM)]
    assert not (state / "server.pid").exists()


# --- main() with a fake sky module ---------------------------------------------------------


class FakeSky(types.ModuleType):
    """Minimal SkyPilot client SDK: every API returns a request id that ``get`` resolves."""

    def __init__(self) -> None:
        super().__init__("sky")
        self.results: dict[str, Any] = {
            "launch": (1, None),
            "endpoints": {8000: "1.2.3.4:8000"},
            "status": [],
            "stop": None,
            "down": None,
            "check": None,
        }
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.tasks: list[types.SimpleNamespace] = []
        self.resources: list[dict[str, Any]] = []
        self.get_requests: list[str] = []
        self.clouds = types.SimpleNamespace(
            CLOUD_REGISTRY=types.SimpleNamespace(from_str=lambda name: f"cloud:{name}")
        )

    def Task(self, **kwargs: Any) -> types.SimpleNamespace:
        task = types.SimpleNamespace(kwargs=kwargs, resources=None)
        task.set_resources = lambda resources: setattr(task, "resources", resources)
        self.tasks.append(task)
        return task

    def Resources(self, **kwargs: Any) -> dict[str, Any]:
        self.resources.append(kwargs)
        return kwargs

    def _api(self, name: str, *args: Any, **kwargs: Any) -> str:
        self.calls.append((name, args, kwargs))
        return name

    def launch(self, *args: Any, **kwargs: Any) -> str:
        return self._api("launch", *args, **kwargs)

    def endpoints(self, *args: Any, **kwargs: Any) -> str:
        return self._api("endpoints", *args, **kwargs)

    def status(self, *args: Any, **kwargs: Any) -> str:
        return self._api("status", *args, **kwargs)

    def stop(self, *args: Any, **kwargs: Any) -> str:
        return self._api("stop", *args, **kwargs)

    def down(self, *args: Any, **kwargs: Any) -> str:
        return self._api("down", *args, **kwargs)

    def check(self, *args: Any, **kwargs: Any) -> str:
        return self._api("check", *args, **kwargs)

    def get(self, request_id: str) -> Any:
        self.get_requests.append(request_id)
        result = self.results[request_id]
        if isinstance(result, BaseException):
            raise result
        return result

    def call(self, name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        (match,) = [(args, kwargs) for n, args, kwargs in self.calls if n == name]
        return match


@pytest.fixture
def sky(monkeypatch) -> FakeSky:
    fake = FakeSky()
    monkeypatch.setitem(sys.modules, "sky", fake)
    return fake


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    """Isolated home plus a restored-on-teardown SKYPILOT_API_SERVER_ENDPOINT."""
    directory = tmp_path / "home"
    directory.mkdir()
    monkeypatch.setenv("HOME", str(directory))
    monkeypatch.setenv("USERPROFILE", str(directory))
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "placeholder")
    monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT")
    return directory


def run_main(monkeypatch, capsys, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return run_raw(monkeypatch, capsys, json.dumps(request))


def run_raw(monkeypatch, capsys, stdin: str) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = worker.main()
    lines = capsys.readouterr().out.strip().splitlines()
    message = json.loads(lines[-1])
    assert message["iw_worker"] == 1
    return code, message


LAUNCH_PAYLOAD: dict[str, Any] = {
    "mode": "ambient",
    "cloud": "runpod",
    "cluster": "iw-cluster",
    "setup": "echo setup",
    "run": "python3 -m server --model 'a b'",
    "envs": {"MODEL": "m", "PYTHONPATH": ".:$PYTHONPATH"},
    "workdir": "/work",
    "resources": {
        "cloud": "runpod",
        "accelerators": "A100:1",
        "ports": [8000],
        "use_spot": True,
        "image_id": "docker:vllm/vllm-openai:latest",
        "region": "us-east",
    },
    "idle_minutes": 20,
    "down": True,
    "port": 8000,
}


def test_main_launch_builds_task_and_resolves_endpoint(monkeypatch, capsys, sky, home):
    code, message = run_main(monkeypatch, capsys, {"op": "launch", "payload": LAUNCH_PAYLOAD})

    assert code == 0
    assert message == {"iw_worker": 1, "ok": True, "result": {"endpoint": "http://1.2.3.4:8000"}}
    (task,) = sky.tasks
    assert task.kwargs == {
        "name": "iw-cluster",
        "setup": "echo setup",
        "run": "python3 -m server --model 'a b'",
        "envs": {"MODEL": "m", "PYTHONPATH": ".:$PYTHONPATH"},
        "workdir": "/work",
    }
    assert sky.resources == [
        {
            "accelerators": "A100:1",
            "ports": [8000],
            "use_spot": True,
            "image_id": "docker:vllm/vllm-openai:latest",
            "region": "us-east",
            "cloud": "cloud:runpod",
        }
    ]
    assert task.resources == sky.resources[0]
    args, kwargs = sky.call("launch")
    assert args == (task,)
    assert kwargs == {
        "cluster_name": "iw-cluster",
        "idle_minutes_to_autostop": 20,
        "down": True,
        "stream_logs": False,
    }
    assert sky.call("endpoints") == (("iw-cluster",), {"port": 8000})
    # SkyPilot >= 0.9 returns request ids: launch must be awaited with get() before endpoints.
    assert sky.get_requests == ["launch", "endpoints"]
    # The ambient account never starts a private server or touches the endpoint variable.
    assert "SKYPILOT_API_SERVER_ENDPOINT" not in os.environ
    assert not [name for name, _, _ in sky.calls if name == "check"]


def test_main_launch_without_autostop_or_down(monkeypatch, capsys, sky, home):
    payload = {**LAUNCH_PAYLOAD, "idle_minutes": None, "down": False, "port": None}

    code, message = run_main(monkeypatch, capsys, {"op": "launch", "payload": payload})

    assert code == 0
    _, kwargs = sky.call("launch")
    assert kwargs["idle_minutes_to_autostop"] is None
    assert kwargs["down"] is False
    # Without a port the first reported endpoint is used.
    assert sky.call("endpoints") == (("iw-cluster",), {})
    assert message["result"]["endpoint"] == "http://1.2.3.4:8000"


@pytest.mark.parametrize(
    ("endpoints", "expected"),
    [
        ({8000: "https://example.com:8000"}, "https://example.com:8000"),
        ({}, None),
        (None, None),
        (RuntimeError("endpoints not ready"), None),
    ],
)
def test_main_launch_endpoint_variants(monkeypatch, capsys, sky, home, endpoints, expected):
    sky.results["endpoints"] = endpoints

    code, message = run_main(monkeypatch, capsys, {"op": "launch", "payload": LAUNCH_PAYLOAD})

    assert code == 0
    assert message["result"] == {"endpoint": expected}


def test_main_launch_failure_is_classified_and_text_withheld(monkeypatch, capsys, sky, home):
    sky.results["launch"] = ResourcesUnavailableError(f"no capacity {SECRET}")

    code, message = run_main(monkeypatch, capsys, {"op": "launch", "payload": LAUNCH_PAYLOAD})

    assert code == 1
    assert message["ok"] is False
    assert message["error"]["kind"] == "capacity"
    assert message["error"]["resource_may_exist"] is True
    assert SECRET not in json.dumps(message)
    assert sky.get_requests == ["launch"]  # the failure surfaces through get(), not swallowed


def test_main_unknown_error_withholds_text(monkeypatch, capsys, sky, home):
    sky.results["launch"] = RuntimeError(f"boom {SECRET}")

    code, message = run_main(monkeypatch, capsys, {"op": "launch", "payload": LAUNCH_PAYLOAD})

    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert message["error"]["resource_may_exist"] is True
    assert SECRET not in json.dumps(message)


@pytest.mark.parametrize(
    ("cluster", "expected"),
    [
        (types.SimpleNamespace(status=types.SimpleNamespace(value="UP")), "UP"),
        (types.SimpleNamespace(status="INIT"), "INIT"),
        ({"status": types.SimpleNamespace(value="stopped")}, "STOPPED"),
    ],
)
def test_main_status_existing_cluster(monkeypatch, capsys, sky, home, cluster, expected):
    sky.results["status"] = [cluster]

    code, message = run_main(
        monkeypatch,
        capsys,
        {"op": "status", "payload": {"cluster": "iw-cluster", "mode": "ambient", "port": 8000}},
    )

    assert code == 0
    assert sky.call("status") == ((), {"cluster_names": ["iw-cluster"]})
    result = message["result"]
    assert result["exists"] is True
    assert result["status"] == expected
    # Endpoints are only resolved for a running cluster.
    if expected == "UP":
        assert result["endpoint"] == "http://1.2.3.4:8000"
    else:
        assert result["endpoint"] is None
        assert [name for name, _, _ in sky.calls if name == "endpoints"] == []


def test_main_status_missing_cluster(monkeypatch, capsys, sky, home):
    sky.results["status"] = []

    code, message = run_main(
        monkeypatch, capsys, {"op": "status", "payload": {"cluster": "iw-cluster", "mode": "ambient"}}
    )

    assert code == 0
    assert message["result"] == {"exists": False, "status": None, "endpoint": None}


@pytest.mark.parametrize(("op", "sky_call"), [("stop", "stop"), ("down", "down")])
def test_main_stop_and_down_call_matching_sky_api(monkeypatch, capsys, sky, home, op, sky_call):
    code, message = run_main(
        monkeypatch, capsys, {"op": op, "payload": {"cluster": "iw-cluster", "mode": "ambient"}}
    )

    assert code == 0
    assert message["result"] == {"existed": True}
    assert sky.call(sky_call) == ((), {"cluster_name": "iw-cluster"})
    assert sky.get_requests == [sky_call]
    assert [name for name, _, _ in sky.calls] == [sky_call]  # no other teardown call


@pytest.mark.parametrize("op", ["stop", "down"])
def test_main_missing_cluster_is_a_successful_teardown(monkeypatch, capsys, sky, home, op):
    sky.results[op] = ClusterDoesNotExist("gone")

    code, message = run_main(
        monkeypatch, capsys, {"op": op, "payload": {"cluster": "iw-cluster", "mode": "ambient"}}
    )

    assert code == 0
    assert message["result"] == {"existed": False}


def test_main_teardown_failure_is_classified(monkeypatch, capsys, sky, home):
    sky.results["down"] = RuntimeError(f"cannot delete {SECRET}")

    code, message = run_main(
        monkeypatch, capsys, {"op": "down", "payload": {"cluster": "iw-cluster", "mode": "ambient"}}
    )

    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert message["error"]["resource_may_exist"] is None
    assert SECRET not in json.dumps(message)


def _fake_account_server(monkeypatch, home: Path) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def fake(home_arg: Path, cloud: str, api_key: str, port: int, fingerprint: str) -> str:
        calls.append((home_arg, cloud, api_key, port, fingerprint))
        (home_arg / ".inferweave").mkdir(parents=True, exist_ok=True)
        return f"http://127.0.0.1:{port}"

    monkeypatch.setattr(worker, "ensure_account_server", fake)
    return calls


def test_main_account_mode_starts_private_server_and_checks_cloud_once(
    monkeypatch, capsys, sky, home
):
    server_calls = _fake_account_server(monkeypatch, home)
    request = {
        "op": "status",
        "payload": {
            "mode": "account",
            "cloud": "runpod",
            "cluster": "iw-cluster",
            "server_port": 47010,
            "fingerprint": "fp-1",
        },
        "secrets": {"api_key": SECRET},
    }

    code, message = run_main(monkeypatch, capsys, request)

    assert code == 0, message
    assert server_calls == [(home, "runpod", SECRET, 47010, "fp-1")]
    assert os.environ["SKYPILOT_API_SERVER_ENDPOINT"] == "http://127.0.0.1:47010"
    assert sky.call("check") == ((), {"infra_list": ("runpod",), "verbose": False})
    assert (home / ".inferweave" / "checked-runpod").exists()
    assert SECRET not in json.dumps(message)

    # A second operation for the same account does not repeat the (slow) cloud check.
    sky.calls.clear()
    code, _ = run_main(monkeypatch, capsys, request)
    assert code == 0
    assert [name for name, _, _ in sky.calls] == ["status"]


def test_main_account_mode_without_api_key_is_an_auth_error(monkeypatch, capsys, sky, home):
    server_calls = _fake_account_server(monkeypatch, home)
    request = {
        "op": "launch",
        "payload": {**LAUNCH_PAYLOAD, "mode": "account", "server_port": 47010, "fingerprint": "f"},
        "secrets": {},
    }

    code, message = run_main(monkeypatch, capsys, request)

    assert code == 1
    assert message["error"]["kind"] == "auth"
    assert message["error"]["resource_may_exist"] is False
    assert server_calls == []
    assert sky.tasks == []


def test_main_server_start_failure_does_not_reach_launch(monkeypatch, capsys, sky, home):
    def failing(*args: Any) -> str:
        raise worker.WorkerError("transient", "The account's SkyPilot API server did not start.", False)

    monkeypatch.setattr(worker, "ensure_account_server", failing)
    request = {
        "op": "launch",
        "payload": {**LAUNCH_PAYLOAD, "mode": "account", "server_port": 47010, "fingerprint": "f"},
        "secrets": {"api_key": SECRET},
    }

    code, message = run_main(monkeypatch, capsys, request)

    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert message["error"]["resource_may_exist"] is False
    assert sky.tasks == []


def test_main_reports_missing_skypilot_install(monkeypatch, capsys, home):
    monkeypatch.setitem(sys.modules, "sky", None)  # makes ``import sky`` raise ImportError

    code, message = run_main(
        monkeypatch, capsys, {"op": "status", "payload": {"cluster": "c", "mode": "ambient"}}
    )

    assert code == 1
    assert message["error"]["kind"] == "invalid_request"
    assert message["error"]["resource_may_exist"] is False


@pytest.mark.parametrize(
    "stdin",
    ["", "not json", json.dumps({"payload": {}}), json.dumps({"op": "format-disk"})],
)
def test_main_rejects_malformed_requests(monkeypatch, capsys, sky, home, stdin):
    code, message = run_raw(monkeypatch, capsys, stdin)

    assert code == 2
    assert message["ok"] is False
    assert message["error"]["kind"] == "invalid_request"
    assert sky.calls == []


def test_main_probe_reports_sdk_version_without_any_cluster_or_server_work(
    monkeypatch, capsys, sky, home
):
    sky.__version__ = "9.9.9"
    server_calls = _fake_account_server(monkeypatch, home)

    code, message = run_main(
        monkeypatch,
        capsys,
        {
            "op": "probe",
            "payload": {"mode": "account", "cloud": "runpod", "server_port": 47010, "fingerprint": "f"},
        },
    )

    assert code == 0
    assert message["result"] == {"sdk": "9.9.9"}
    assert server_calls == []  # no private API server, even for an account-mode payload
    assert sky.calls == []
    assert "SKYPILOT_API_SERVER_ENDPOINT" not in os.environ


def test_operations_are_exactly_the_documented_set():
    assert set(worker.OPERATIONS) == {"probe", "server_probe", "launch", "status", "stop", "down"}


def test_account_server_rejects_foreign_listener_without_killing_it(monkeypatch, tmp_path):
    harness = ServerHarness(monkeypatch, up=True)
    monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: False)
    with pytest.raises(worker.WorkerError, match="another process"):
        _ensure(tmp_path)
    assert not harness.stopped and not harness.popen_calls


def test_queue_plugin_redacts_sdk_log_records(monkeypatch, tmp_path):
    import importlib.util
    import logging

    context = types.SimpleNamespace(register_queue_backend_factory=lambda factory: factories.append(factory))
    factories = []
    monkeypatch.setitem(sys.modules, "sky.server.plugins", types.SimpleNamespace(BasePlugin=object, PluginContext=object))
    monkeypatch.setitem(sys.modules, "sky.server.requests.queues.base", types.SimpleNamespace(MultiprocessingQueueFactory=lambda **kw: kw))
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(logging, "_logRecordFactory", logging.getLogRecordFactory())
    spec = importlib.util.spec_from_file_location("queue_plugin_test", Path(worker.__file__).with_name("skypilot_queue_plugin.py"))
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    plugin.AccountQueuePlugin(48002).install(context)
    record = logging.getLogRecordFactory()("sky", logging.ERROR, "sdk.py", 1, "key %s", (SECRET,), (RuntimeError, RuntimeError(SECRET), None))
    assert factories == [{"port": 48002}]
    assert SECRET not in record.getMessage()
    assert record.exc_info is None


def test_server_stop_cannot_claim_success_while_listener_remains(monkeypatch, tmp_path):
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server.pid").write_text("31337")
    monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: True)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append(sig), raising=False)
    monkeypatch.setattr(worker, "_group_processes", lambda pid: [])
    monkeypatch.setattr(worker, "_group_still_owned", lambda members, pid: True)
    monkeypatch.setattr(worker, "_wait_server_stopped", lambda pid, endpoint, timeout: False)
    with pytest.raises(worker.WorkerError, match="termination was not confirmed"):
        worker._stop_server(state, "http://127.0.0.1:47010")
    assert (state / "server.pid").exists()
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_server_stop_kills_surviving_workers_even_when_health_is_down(monkeypatch, tmp_path):
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server.pid").write_text("31337")
    # The leader exited but an original child still holds this private process group.
    members = [types.SimpleNamespace(pid=31338, is_running=lambda: True)]
    signals = []

    def killpg(pid, sig):
        assert pid == 31337
        signals.append(sig)
        if sig == signal.SIGKILL:
            members.clear()

    monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: True)
    monkeypatch.setattr(os, "getpgid", lambda pid: 31337, raising=False)
    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    monkeypatch.setattr(worker, "_group_processes", lambda pid: list(members))
    monkeypatch.setattr(worker, "_healthy", lambda endpoint: False)
    moments = iter([0, 31, 31])
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(monotonic=lambda: next(moments), sleep=lambda _: None))
    worker._stop_server(state, "http://127.0.0.1:47010")
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert not (state / "server.pid").exists()


def test_server_stop_never_escalates_after_original_process_identities_disappear(monkeypatch, tmp_path):
    state = tmp_path / ".inferweave"
    state.mkdir()
    (state / "server.pid").write_text("31337")
    original = types.SimpleNamespace(pid=31337, is_running=lambda: False)
    signals = []
    monkeypatch.setattr(worker, "_server_owned", lambda state, endpoint: True)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append(sig), raising=False)
    monkeypatch.setattr(worker, "_group_processes", lambda pid: [original])
    monkeypatch.setattr(worker, "_wait_server_stopped", lambda pid, endpoint, timeout: False)
    with pytest.raises(worker.WorkerError, match="Cannot verify ownership"):
        worker._stop_server(state, "http://127.0.0.1:47010")
    assert signals == [signal.SIGTERM]
    assert (state / "server.pid").exists()


def test_server_group_ignores_zombies_foreign_groups_and_exited_processes(monkeypatch):
    processes = [
        types.SimpleNamespace(pid=11, status=lambda: "sleeping"),
        types.SimpleNamespace(pid=12, status=lambda: "zombie"),
        types.SimpleNamespace(pid=13, status=lambda: "sleeping"),
        types.SimpleNamespace(pid=14, status=lambda: "sleeping"),
    ]

    def getpgid(pid):
        if pid == 14:
            raise ProcessLookupError
        return 31337 if pid in {11, 12} else 999

    psutil = types.SimpleNamespace(
        process_iter=lambda: processes, STATUS_ZOMBIE="zombie", NoSuchProcess=ProcessLookupError,
    )
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: psutil)
    monkeypatch.setattr(os, "getpgid", getpgid, raising=False)
    assert worker._group_processes(31337) == [processes[0]]
