"""Offline tests of the Lightning SDK worker script, run in-process with fake SDK modules.

``inferweave.isolation.lightning_worker`` is imported as a module; ``main()`` is driven with a
fake stdin/stdout and fake ``lightning_sdk`` modules injected into ``sys.modules``. Nothing here
starts a subprocess or contacts Lightning.
"""

import contextlib
import io
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
from click.exceptions import ClickException, Exit

from inferweave.isolation import lightning_worker as worker

SECRET = "SENTINEL-sdk-body-secret"
USER_ID = "SENTINEL-worker-user"
API_KEY = "SENTINEL-worker-key"
SECRETS = {"LIGHTNING_USER_ID": USER_ID, "LIGHTNING_API_KEY": API_KEY}
URL = "https://8080-dep-x.cloudspaces.litng.ai"


class ApiException(Exception):
    """Mimics the generated REST client's exception: ``.status`` plus a body that echoes input."""

    def __init__(self, status, body=SECRET):
        super().__init__(body)
        self.status = status
        self.body = body


# --- classify ----------------------------------------------------------------------------------


@pytest.mark.parametrize("status,kind", [
    (401, "auth"), (403, "permission"), (402, "quota"), (429, "rate_limit"),
    (400, "invalid_request"), (404, "invalid_request"), (409, "invalid_request"),
    (422, "invalid_request"), (500, "transient"), (502, "transient"), (503, "transient"),
    (418, "transient"),
])
def test_classify_http_status(status, kind):
    error = worker.classify(ApiException(status))
    assert (error.kind, error.status) == (kind, status)
    assert error.resource_may_exist is None
    assert str(error) == f"Lightning SDK error (HTTP {status}); details withheld to protect credentials."
    assert SECRET not in str(error)


@pytest.mark.parametrize("status", [0, -5, "403", None, 4.5])
def test_classify_ignores_invalid_status_attribute(status):
    error = worker.classify(ApiException(status))
    assert (error.kind, error.status) == ("transient", None)
    assert "ApiException" in str(error) and SECRET not in str(error)


def test_classify_authentication_failed_connection_error_is_auth():
    error = worker.classify(ConnectionError(f"Authentication failed: {SECRET}"))
    assert (error.kind, error.status) == ("auth", 401)
    assert SECRET not in str(error)


def test_classify_other_connection_error_is_transient():
    error = worker.classify(ConnectionError(f"connection reset {SECRET}"))
    assert (error.kind, error.status) == ("transient", None)
    assert "ConnectionError" in str(error) and SECRET not in str(error)


@pytest.mark.parametrize("text,kind,status", [
    ("Request failed, response: 429", "rate_limit", 429),
    ("giving up after retries, response:429 body=" + SECRET, "rate_limit", 429),
    ("response: 403", "permission", 403),
    ("response: 401", "auth", 401),
    ("response: 503", "transient", 503),
    ("response: 422", "invalid_request", 422),
])
def test_classify_status_embedded_in_plain_exception_text(text, kind, status):
    error = worker.classify(Exception(text))
    assert (error.kind, error.status) == (kind, status)
    assert SECRET not in str(error)


def test_classify_status_attribute_wins_over_message_text():
    err = ApiException(403, "response: 429")
    assert worker.classify(err).status == 403


@pytest.mark.parametrize("err", [
    RuntimeError(f"boom {SECRET}"),
    ValueError(SECRET),
    KeyError(SECRET),
    Exception("no status here, response: abc"),
])
def test_classify_unknown_errors_are_transient_and_text_is_withheld(err):
    error = worker.classify(err)
    assert (error.kind, error.status) == ("transient", None)
    assert type(err).__name__ in str(error)
    assert SECRET not in str(error)


def test_classify_passes_worker_errors_through():
    original = worker.WorkerError("auth", "nope", 401, False)
    assert worker.classify(original) is original
    assert (original.kind, original.status, original.resource_may_exist) == ("auth", 401, False)


# --- fakes for main() --------------------------------------------------------------------------


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ApiKeyAuth(Recorder):
    pass


class HttpHealthCheck(Recorder):
    pass


class AutoScaleConfig(Recorder):
    pass


@pytest.fixture(autouse=True)
def envs(monkeypatch):
    """Gives the worker a private ``os.environ`` copy so nothing it sets can leak into the run."""
    real = os.environ
    private = real.copy()
    for name in (*SECRETS, "LIGHTNING_DISABLE_VERSION_CHECK", "LIGHTNING_AUTH_TOKEN"):
        private.pop(name, None)
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, "environ", private)
    return SimpleNamespace(real=real, private=private)


@pytest.fixture
def sdk(monkeypatch):
    """Fake ``lightning_sdk`` package plus CLI entrypoint and auth helpers."""
    state = SimpleNamespace(
        lookups=[],
        starts=[],
        cli_calls=[],
        browser_auth=[],
        env_at_start={},
        collision=False,
        start_error=None,
        url_polls=0,
        urls_by_poll=[[URL]],
        cli=lambda args: "",  # stdout text, or raises ClickException/Exit
        sleeps=[],
    )

    class Deployment:
        def __init__(self, name, teamspace=None):
            state.lookups.append((name, teamspace))
            self.name = name
            self.id = "dep_" + name
            self.is_started = state.collision

        def start(self, **kwargs):
            state.starts.append(kwargs)
            state.env_at_start = {
                key: os.environ.get(key)
                for key in (*SECRETS, "LIGHTNING_DISABLE_VERSION_CHECK", "EMPTY")
            }
            if state.start_error is not None:
                raise state.start_error

        @property
        def urls(self):
            state.url_polls += 1
            index = min(state.url_polls, len(state.urls_by_poll)) - 1
            value = state.urls_by_poll[index]
            if isinstance(value, BaseException):
                raise value
            return value

    machine = SimpleNamespace(L4="MACHINE-L4", L40S_X_2="MACHINE-L40S-2")

    class MainCli:
        def main(self, args, prog_name, standalone_mode):
            state.cli_calls.append((list(args), prog_name, standalone_mode))
            sys.stdout.write(state.cli(args))

    @contextlib.contextmanager
    def browser_authentication(enabled):
        state.browser_auth.append(enabled)
        yield

    package = ModuleType("lightning_sdk")
    package.Deployment = Deployment
    package.Machine = machine
    config = ModuleType("lightning_sdk.deployment")
    config.ApiKeyAuth = ApiKeyAuth
    config.HttpHealthCheck = HttpHealthCheck
    config.AutoScaleConfig = AutoScaleConfig
    package.deployment = config
    cli = ModuleType("lightning_sdk.cli")
    entrypoint = ModuleType("lightning_sdk.cli.entrypoint")
    entrypoint.main_cli = MainCli()
    utils = ModuleType("lightning_sdk.cli.utils")
    auth = ModuleType("lightning_sdk.cli.utils.auth")
    auth.browser_authentication = browser_authentication
    for name, module in {
        "lightning_sdk": package,
        "lightning_sdk.deployment": config,
        "lightning_sdk.cli": cli,
        "lightning_sdk.cli.entrypoint": entrypoint,
        "lightning_sdk.cli.utils": utils,
        "lightning_sdk.cli.utils.auth": auth,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Discovery polling must not really sleep.
    monkeypatch.setattr(
        worker,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: state.sleeps.append(seconds)),
    )
    return state


def run_main(monkeypatch, capsys, request):
    """Runs ``worker.main()`` with ``request`` on stdin; returns (exit code, protocol message, stdout)."""
    text = request if isinstance(request, str) else json.dumps(request)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    code = worker.main()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if '"iw_worker"' in line]
    assert len(lines) == 1, out  # the protocol line is the only thing the worker prints
    message = json.loads(lines[-1])
    assert message["iw_worker"] == 1
    return code, message, out


START_PAYLOAD = {
    "name": "iw-lightning-abc",
    "teamspace": "owner/tests",
    "machine": "L4",
    "image": "registry/image:1",
    "port": 8080,
    "command": "-c 'echo hi'",
    "env": {"MODEL": "x"},
    "healthcheck_path": "/v1/health",
    "min_replicas": 0,
    "max_replicas": 2,
    "idle_threshold_seconds": 90,
    "discovery_timeout": 60,
}


def request_for(op, payload=None, secrets=None):
    return {"op": op, "payload": payload or {}, "secrets": secrets if secrets is not None else SECRETS}


# --- main(): protocol errors -------------------------------------------------------------------


@pytest.mark.parametrize("stdin", ["not json", "", "{}", '{"op": "explode"}', '{"payload": {}}'])
def test_bad_request_is_invalid_request_exit_2(sdk, monkeypatch, capsys, stdin):
    code, message, _ = run_main(monkeypatch, capsys, stdin)
    assert code == 2
    assert message["ok"] is False
    assert message["error"] == {"kind": "invalid_request", "message": "bad worker request"}
    assert not sdk.browser_auth  # nothing ran


def test_missing_sdk_is_reported_without_resource(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lightning_sdk", None)  # import raises ImportError
    code, message, _ = run_main(monkeypatch, capsys, request_for("whoami"))
    assert code == 3
    assert message["ok"] is False
    error = message["error"]
    assert error["kind"] == "invalid_request"
    assert error["resource_may_exist"] is False
    assert "lightning-sdk is not installed" in error["message"]


# --- main(): credentials -----------------------------------------------------------------------


def test_secrets_from_stdin_land_only_in_the_worker_environment(sdk, monkeypatch, capsys, envs):
    code, message, out = run_main(
        monkeypatch, capsys,
        request_for("start", START_PAYLOAD, {**SECRETS, "EMPTY": ""}),
    )
    assert code == 0 and message["ok"] is True
    assert sdk.env_at_start["LIGHTNING_USER_ID"] == USER_ID
    assert sdk.env_at_start["LIGHTNING_API_KEY"] == API_KEY
    assert sdk.env_at_start["LIGHTNING_DISABLE_VERSION_CHECK"] == "1"
    assert sdk.env_at_start["EMPTY"] is None  # empty secrets are ignored
    # They were written to the worker's (private copy of the) environment, never echoed back.
    assert envs.private["LIGHTNING_API_KEY"] == API_KEY
    assert USER_ID not in out and API_KEY not in out
    assert "LIGHTNING_API_KEY" not in envs.real and "LIGHTNING_USER_ID" not in envs.real
    # The SDK never opens an interactive browser login inside a worker.
    assert sdk.browser_auth == [False]


def test_request_without_secrets_runs_with_the_inherited_environment(sdk, monkeypatch, capsys):
    request = {"op": "inspect", "payload": {"target": "x", "teamspace": "o/t"}}
    sdk.cli = lambda args: json.dumps({"id": "dep_x"})
    code, message, _ = run_main(monkeypatch, capsys, request)
    assert code == 0 and message["result"] == {"id": "dep_x"}
    assert os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] == "1"


# --- main(): whoami ----------------------------------------------------------------------------


def test_whoami_returns_only_the_auth_type(sdk, monkeypatch, capsys):
    sdk.cli = lambda args: json.dumps({"auth_type": "user", "email": "private@example.test"})
    code, message, out = run_main(monkeypatch, capsys, request_for("whoami"))
    assert code == 0
    assert message == {"iw_worker": 1, "ok": True, "result": {"auth_type": "user"}}
    assert "private@example.test" not in out
    assert sdk.cli_calls == [(["auth", "whoami", "--json"], "lightning", False)]


def test_whoami_with_non_object_identity_has_no_auth_type(sdk, monkeypatch, capsys):
    sdk.cli = lambda args: json.dumps(["unexpected"])
    _, message, _ = run_main(monkeypatch, capsys, request_for("whoami"))
    assert message["result"] == {"auth_type": None}


def test_whoami_generic_cli_failure_does_not_revoke_credentials(sdk, monkeypatch, capsys):
    def cli(args):
        raise ClickException(f"bad credentials {SECRET}")

    sdk.cli = cli
    code, message, out = run_main(monkeypatch, capsys, request_for("whoami"))
    assert code == 1
    assert message["error"].items() >= {
        "kind": "transient",
        "status": None,
        "message": "Lightning identity check failed.",
        "resource_may_exist": False,
    }.items()
    assert SECRET not in out


def test_whoami_invalid_json_is_transient(sdk, monkeypatch, capsys):
    sdk.cli = lambda args: "<html>gateway</html>"
    code, message, _ = run_main(monkeypatch, capsys, request_for("whoami"))
    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert "invalid identity response" in message["error"]["message"]


# --- main(): start -----------------------------------------------------------------------------


def test_start_passes_the_recipe_to_the_sdk_and_polls_urls(sdk, monkeypatch, capsys):
    sdk.urls_by_poll = [[], RuntimeError(f"not ready {SECRET}"), [], [URL]]
    code, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    assert code == 0
    assert message["result"] == {"resource_id": "dep_iw-lightning-abc", "urls": [URL]}
    assert sdk.lookups == [("iw-lightning-abc", "owner/tests")]
    assert sdk.url_polls == 4
    assert sdk.sleeps == [1, 1, 1]
    assert SECRET not in out  # a failing poll is swallowed, never surfaced

    [start] = sdk.starts
    assert start["image"] == "registry/image:1"
    assert start["machine"] == "MACHINE-L4"
    assert start["ports"] == [8080]
    assert start["entrypoint"] == "/bin/sh"
    assert start["command"] == "-c 'echo hi'"
    assert start["env"] == {"MODEL": "x"}
    assert start["include_credentials"] is False
    assert start["spot"] is False
    assert start["replicas"] == 1
    assert isinstance(start["auth"], ApiKeyAuth)
    assert isinstance(start["health_check"], HttpHealthCheck)
    assert start["health_check"].kwargs == {"path": "/v1/health", "port": 8080}
    assert isinstance(start["autoscale"], AutoScaleConfig)
    assert start["autoscale"].kwargs == {
        "min_replicas": 0, "max_replicas": 2, "metric": "GPU", "threshold": 90,
        "idle_threshold_seconds": "90",
    }


def test_start_returns_no_urls_when_discovery_times_out(sdk, monkeypatch, capsys):
    sdk.urls_by_poll = [[]]
    payload = {**START_PAYLOAD, "discovery_timeout": 0}
    code, message, _ = run_main(monkeypatch, capsys, request_for("start", payload))
    assert code == 0
    assert message["result"] == {"resource_id": "dep_iw-lightning-abc", "urls": []}
    assert sdk.url_polls == 1 and sdk.sleeps == []


def test_start_name_collision_is_definitive_and_preserves_the_resource(sdk, monkeypatch, capsys):
    sdk.collision = True
    code, message, _ = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    assert code == 1
    assert message["error"].items() >= {
        "kind": "invalid_request",
        "status": None,
        "message": "Lightning resource name collision; the existing resource was preserved.",
        "resource_may_exist": False,
    }.items()
    assert sdk.starts == []  # never touched the pre-existing Deployment


@pytest.mark.parametrize("status,kind", [
    (400, "invalid_request"), (401, "auth"), (402, "quota"), (403, "permission"),
    (404, "invalid_request"), (409, "invalid_request"), (422, "invalid_request"),
    (429, "rate_limit"),
])
def test_start_rejection_can_follow_an_accepted_create(sdk, monkeypatch, capsys, status, kind):
    sdk.start_error = ApiException(status, f"echoed request body {SECRET}")
    code, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    assert code == 1
    error = message["error"]
    assert (error["kind"], error["status"]) == (kind, status)
    assert error["resource_may_exist"] is True
    assert error["operation_may_continue"] is True
    assert error["message"] == (
        f"Lightning SDK error (HTTP {status}); details withheld to protect credentials."
    )
    assert SECRET not in out and API_KEY not in out


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_start_server_error_may_have_created_the_resource(sdk, monkeypatch, capsys, status):
    sdk.start_error = ApiException(status)
    code, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    assert code == 1
    error = message["error"]
    assert (error["kind"], error["status"], error["resource_may_exist"]) == ("transient", status, True)
    assert SECRET not in out


@pytest.mark.parametrize("err", [
    RuntimeError(f"socket closed {SECRET}"),
    TimeoutError(SECRET),
    Exception("weird"),
])
def test_start_unknown_error_may_have_created_the_resource(sdk, monkeypatch, capsys, err):
    sdk.start_error = err
    code, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    assert code == 1
    error = message["error"]
    assert (error["kind"], error["status"], error["resource_may_exist"]) == ("transient", None, True)
    assert type(err).__name__ in error["message"]
    assert SECRET not in out


def test_start_retried_4xx_in_message_text_is_a_definitive_rejection(sdk, monkeypatch, capsys):
    sdk.start_error = Exception(f"Max retries exceeded, response: 403 {SECRET}")
    _, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    error = message["error"]
    assert (error["kind"], error["status"], error["resource_may_exist"]) == ("permission", 403, True)
    assert SECRET not in out


def test_start_authentication_failed_connection_error_is_auth(sdk, monkeypatch, capsys):
    sdk.start_error = ConnectionError(f"Authentication failed {SECRET}")
    _, message, out = run_main(monkeypatch, capsys, request_for("start", START_PAYLOAD))
    error = message["error"]
    assert (error["kind"], error["status"], error["resource_may_exist"]) == ("auth", 401, True)
    assert SECRET not in out


# --- main(): inspect / delete ------------------------------------------------------------------

TARGET = {"target": "dep_x", "teamspace": "owner/tests"}


def test_inspect_returns_the_deployment_json(sdk, monkeypatch, capsys):
    data = {"desired_state": "RUNNING", "status": {"ready_replicas": 1}}
    sdk.cli = lambda args: json.dumps(data)
    code, message, out = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 0 and message["result"] == data
    assert sdk.cli_calls == [
        (["deployment", "inspect", "dep_x", "--teamspace", "owner/tests", "--json"],
         "lightning", False)
    ]
    assert out.count("\n") == 1  # CLI stdout was captured, not leaked into the protocol stream


def test_inspect_not_found_is_none(sdk, monkeypatch, capsys):
    def cli(args):
        raise ClickException("Deployment 'dep_x' was not found in teamspace owner/tests")

    sdk.cli = cli
    code, message, _ = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 0
    assert message["ok"] is True and message["result"] is None


@pytest.mark.parametrize("failure", [
    ClickException(f"server exploded {SECRET}"),
    Exit(1),
])
def test_inspect_other_failure_is_transient_and_text_withheld(sdk, monkeypatch, capsys, failure):
    def cli(args):
        raise failure

    sdk.cli = cli
    code, message, out = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert message["error"]["message"] == "Lightning deployment inspect failed."
    assert SECRET not in out


@pytest.mark.parametrize("text,fragment", [
    ("not json", "invalid deployment status response"),
    (json.dumps(["list"]), "invalid deployment status"),
])
def test_inspect_invalid_payload_is_transient(sdk, monkeypatch, capsys, text, fragment):
    sdk.cli = lambda args: text
    code, message, _ = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert fragment in message["error"]["message"]


def test_inspect_exit_zero_with_output_is_success(sdk, monkeypatch, capsys):
    def cli(args):
        sys.stdout.write(json.dumps({"id": "dep_x"}))
        raise Exit(0)

    sdk.cli = cli
    code, message, _ = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 0 and message["result"] == {"id": "dep_x"}


def test_delete_confirms_with_yes_flag(sdk, monkeypatch, capsys):
    code, message, _ = run_main(monkeypatch, capsys, request_for("delete", TARGET))
    assert code == 0 and message["result"] == {"deleted": True}
    assert sdk.cli_calls == [
        (["deployment", "delete", "dep_x", "--teamspace", "owner/tests", "--yes"],
         "lightning", False)
    ]


def test_delete_not_found_is_not_deleted(sdk, monkeypatch, capsys):
    def cli(args):
        raise ClickException("Deployment 'dep_x' was not found")

    sdk.cli = cli
    code, message, _ = run_main(monkeypatch, capsys, request_for("delete", TARGET))
    assert code == 0
    assert message["result"] == {"deleted": False}


def test_delete_failure_is_transient_and_text_withheld(sdk, monkeypatch, capsys):
    def cli(args):
        raise ClickException(f"denied for key {API_KEY} {SECRET}")

    sdk.cli = cli
    code, message, out = run_main(monkeypatch, capsys, request_for("delete", TARGET))
    assert code == 1
    assert message["error"]["kind"] == "transient"
    assert message["error"]["message"] == "Lightning deployment delete failed."
    assert SECRET not in out and API_KEY not in out


def test_unexpected_exception_in_an_operation_is_classified_not_raised(sdk, monkeypatch, capsys):
    def cli(args):
        raise ApiException(429, f"slow down {SECRET}")

    sdk.cli = cli
    code, message, out = run_main(monkeypatch, capsys, request_for("inspect", TARGET))
    assert code == 1
    error = message["error"]
    assert (error["kind"], error["status"]) == ("rate_limit", 429)
    assert SECRET not in out


def test_probe_reports_the_sdk_version_without_network(sdk, monkeypatch, capsys):
    sys.modules["lightning_sdk"].__version__ = "2026.10.1"
    code, message, _ = run_main(monkeypatch, capsys, request_for("probe", secrets={}))
    assert code == 0
    assert message["result"] == {"sdk": "2026.10.1"}
    assert sdk.cli_calls == [] and sdk.lookups == []


def test_probe_without_a_version_attribute_reports_unknown(sdk, monkeypatch, capsys):
    code, message, _ = run_main(monkeypatch, capsys, request_for("probe", secrets={}))
    assert code == 0
    assert message["result"] == {"sdk": "unknown"}


def test_operations_table_matches_the_provider_protocol():
    assert set(worker.OPERATIONS) == {"probe", "whoami", "start", "inspect", "delete"}
