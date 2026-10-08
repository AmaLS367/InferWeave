"""WorkerRunner with real subprocesses and a tiny self-contained worker script."""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

import pytest

from inferweave.core.exceptions import ProviderOperationError
from inferweave.core.failures import FailureKind
from inferweave.isolation.runner import (
    _PASSTHROUGH_ENV,
    WorkerRunner,
    account_environment,
    ambient_environment,
)

pytestmark = pytest.mark.asyncio

SECRET = "SENTINEL-worker-api-key-9f8e7d"
STDERR_SECRET = "SENTINEL-stderr-leak-1a2b3c"

WORKER = r"""
import json
import os
import sys
import time
from pathlib import Path

def reply(message):
    message["iw_worker"] = 1
    print(json.dumps(message), flush=True)

request = json.loads(sys.stdin.read())
op = request["op"]
payload = request["payload"]
if op == "echo":
    print("noise before the result", flush=True)
    print('{"iw_worker": 1, "ok": true, "result": "stale earlier line"}', flush=True)
    reply({"ok": True, "result": {
        "argv": sys.argv,
        "env": dict(os.environ),
        "secrets": request["secrets"],
        "home": str(Path.home()),
        "payload": payload,
    }})
elif op == "sleep":
    Path(payload["pid_file"]).write_text(str(os.getpid()))
    time.sleep(30)
    reply({"ok": True, "result": "too late"})
elif op == "crash":
    sys.stderr.write("Traceback: api_key=" + payload["leak"] + "\n")
    sys.exit(3)
elif op == "silent":
    print("no structured output here")
elif op == "error":
    sys.stderr.write("debug " + payload["leak"] + "\n")
    reply({"ok": False, "error": payload["error"]})
"""


@pytest.fixture
def runner(tmp_path: Path) -> WorkerRunner:
    script = tmp_path / "fake_worker.py"
    script.write_text(WORKER, encoding="utf-8")
    return WorkerRunner("fakecloud", script, python=sys.executable)


async def test_secrets_travel_on_stdin_not_argv_or_env(
    runner: WorkerRunner, tmp_path: Path
) -> None:
    env = account_environment(tmp_path / "acct-home")
    result = await runner.run(
        "echo",
        {"x": 1},
        env=env,
        secrets={"api_key": SECRET},
        timeout=60,
        account_id="acct-a",
    )
    assert result["secrets"] == {"api_key": "***"}
    assert result["payload"] == {"x": 1}
    assert SECRET not in " ".join(runner.command())
    assert runner.command() == [sys.executable, "-I", str(runner.script)]
    assert SECRET not in " ".join(result["argv"])
    assert SECRET not in json.dumps(result["env"])
    assert result["env"]["PYTHONIOENCODING"] == "utf-8"


async def test_child_env_is_exactly_what_is_passed(
    runner: WorkerRunner, tmp_path: Path
) -> None:
    env = account_environment(tmp_path / "home", {"EXTRA_SETTING": SECRET})
    result = await runner.run("echo", {}, env=env, secrets={}, timeout=60)
    assert result["env"]["EXTRA_SETTING"] == SECRET  # only because the caller passed it


async def test_account_environment_is_minimal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: WorkerRunner
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "SENTINEL-parent-modal-token")
    monkeypatch.setenv("LIGHTNING_API_KEY", "SENTINEL-parent-lightning-key")
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    home = tmp_path / "accounts" / "a" / "home"
    env = account_environment(home)
    assert home.is_dir()
    assert env["HOME"] == str(home) and env["USERPROFILE"] == str(home)
    assert "MODAL_TOKEN_ID" not in env and "LIGHTNING_API_KEY" not in env
    assert "SENTINEL" not in json.dumps(env)
    allowed = {"HOME", "USERPROFILE", *_PASSTHROUGH_ENV}
    assert set(env) <= allowed

    result = await runner.run("echo", {}, env=env, secrets={}, timeout=60)
    assert "MODAL_TOKEN_ID" not in result["env"]
    assert "SENTINEL" not in json.dumps(result["env"])
    assert (
        Path(result["home"]) == home
    )  # Path.home() resolves to the account-private home


async def test_ambient_environment_keeps_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTNING_USER_ID", "ambient-user")
    env = ambient_environment({"X_EXTRA": "1"})
    assert env["LIGHTNING_USER_ID"] == "ambient-user" and env["X_EXTRA"] == "1"


async def test_timeout_kills_child_and_is_uncertain(
    runner: WorkerRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes = []
    real_exec = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    pid_file = tmp_path / "pid"
    started = time.monotonic()
    with pytest.raises(ProviderOperationError) as info:
        await runner.run(
            "sleep",
            {"pid_file": str(pid_file)},
            env=account_environment(tmp_path / "home"),
            secrets={"api_key": SECRET},
            timeout=2.5,
            deployment_id="dep-1",
            account_id="acct-a",
        )
    assert time.monotonic() - started < 15
    err = info.value
    assert err.kind is FailureKind.TRANSIENT and err.resource_may_exist is True
    assert err.deployment_id == "dep-1"
    assert "timed out" in str(err) and "acct-a" in str(err)
    assert pid_file.exists()  # the worker really started
    [process] = processes
    assert process.returncode is not None  # killed and reaped, not left running
    assert SECRET not in "".join(traceback.format_exception(err))


async def test_crash_is_transient_and_hides_stderr(
    runner: WorkerRunner, tmp_path: Path
) -> None:
    with pytest.raises(ProviderOperationError) as info:
        await runner.run(
            "crash",
            {"leak": STDERR_SECRET},
            env=account_environment(tmp_path / "h"),
            timeout=60,
        )
    err = info.value
    assert err.kind is FailureKind.TRANSIENT and err.resource_may_exist is True
    assert "exit code 3" in str(err)
    assert STDERR_SECRET not in str(err) + repr(err) + "".join(
        traceback.format_exception(err)
    )


async def test_missing_result_line_is_transient(
    runner: WorkerRunner, tmp_path: Path
) -> None:
    with pytest.raises(ProviderOperationError) as info:
        await runner.run(
            "silent", {}, env=account_environment(tmp_path / "h"), timeout=60
        )
    assert (
        info.value.kind is FailureKind.TRANSIENT
        and info.value.resource_may_exist is True
    )
    assert "without a result" in str(info.value)


@pytest.mark.parametrize(
    ("error", "kind", "status", "retry_after", "may_exist"),
    [
        (
            {
                "kind": "rate_limit",
                "status": 429,
                "retry_after": 12,
                "message": "slow down",
                "resource_may_exist": False,
            },
            FailureKind.RATE_LIMIT,
            429,
            12.0,
            False,
        ),
        (
            {"kind": "auth", "status": 401, "message": "bad key"},
            FailureKind.AUTH,
            401,
            None,
            False,
        ),
        (
            {"kind": "capacity", "message": "no GPUs"},
            FailureKind.CAPACITY,
            None,
            None,
            True,
        ),
        (
            {"kind": "quota", "status": "402", "retry_after": "soon"},
            FailureKind.QUOTA,
            None,
            None,
            False,
        ),
        (
            {"kind": "something-new", "message": "?"},
            FailureKind.TRANSIENT,
            None,
            None,
            True,
        ),
        ({}, FailureKind.TRANSIENT, None, None, True),
    ],
)
async def test_structured_errors_are_mapped(
    runner: WorkerRunner,
    tmp_path: Path,
    error: dict,
    kind: FailureKind,
    status: int | None,
    retry_after: float | None,
    may_exist: bool,
) -> None:
    with pytest.raises(ProviderOperationError) as info:
        await runner.run(
            "error",
            {"error": error, "leak": STDERR_SECRET},
            env=account_environment(tmp_path / "h"),
            secrets={"api_key": SECRET},
            timeout=60,
            deployment_id="dep-9",
            account_id="acct-b",
        )
    err = info.value
    assert (err.kind, err.status_code, err.retry_after, err.resource_may_exist) == (
        kind,
        status,
        retry_after,
        may_exist,
    )
    assert err.deployment_id == "dep-9"
    assert str(err).startswith("fakecloud error (account 'acct-b') failed: ")
    if error.get("message"):
        assert error["message"] in str(err)
    rendered = str(err) + "".join(traceback.format_exception(err))
    assert STDERR_SECRET not in rendered and SECRET not in rendered


async def test_missing_interpreter_is_invalid_request(tmp_path: Path) -> None:
    runner = WorkerRunner(
        "fakecloud", tmp_path / "w.py", python=str(tmp_path / "no-such-python.exe")
    )
    with pytest.raises(ProviderOperationError) as info:
        await runner.run(
            "echo",
            {},
            env={},
            secrets={"api_key": SECRET},
            timeout=10,
            deployment_id="d",
        )
    err = info.value
    assert err.kind is FailureKind.INVALID_REQUEST and err.resource_may_exist is False
    assert "no-such-python" in str(err)
    assert SECRET not in "".join(traceback.format_exception(err))
    assert err.__cause__ is None and err.__suppress_context__
