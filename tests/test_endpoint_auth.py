"""Authenticated Modal endpoints: proxy auth for inference, probes, and secret hygiene."""

import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import modal
import pytest
from conftest import TEST_PROXY_TOKEN_ID, TEST_PROXY_TOKEN_SECRET
from support import AUDIO_MODEL, ENDPOINT, deploy_on_modal, make_weave, patched_modal

from inferweave import InferWeave
from inferweave.adapters.auth import (
    CompositeEndpointAuth,
    ModalProxyAuth,
    StaticHeaderAuth,
)
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.core.exceptions import InferenceError, ProviderAuthError
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.healthcheck import ProbeResult
from inferweave.domain.options import DeploymentOptions, RuntimeOptions
from inferweave.models.enums import DeploymentState

SECRETS = (TEST_PROXY_TOKEN_ID, TEST_PROXY_TOKEN_SECRET)
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16
EXPECTED_AUTH = {"modal-key": TEST_PROXY_TOKEN_ID, "modal-secret": TEST_PROXY_TOKEN_SECRET}


class HeaderRecordingProbe(MockHealthcheckProbeAdapter):
    """Mock probe that also records the headers each probe request carried."""

    def __init__(self) -> None:
        super().__init__(default_healthy=True)
        self.headers_seen: list[dict[str, str]] = []

    async def probe(self, url, method="GET", timeout_seconds=5.0, headers=None, expected_status_codes=None) -> ProbeResult:
        self.headers_seen.append(dict(headers or {}))
        return await super().probe(url, method, timeout_seconds, headers, expected_status_codes)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


def _assert_auth(headers: dict[str, str]) -> None:
    lowered = {k.lower(): v for k, v in headers.items()}
    assert {k: lowered.get(k) for k in EXPECTED_AUTH} == EXPECTED_AUTH


# ------------------------------------------------------------- auth resolvers


def test_modal_proxy_auth_resolves_headers_from_environment():
    auth = ModalProxyAuth()
    assert auth.headers_for("modal", ENDPOINT) == {
        "Modal-Key": TEST_PROXY_TOKEN_ID,
        "Modal-Secret": TEST_PROXY_TOKEN_SECRET,
    }
    assert auth.is_configured_for("modal")


def test_modal_proxy_auth_is_scoped_to_provider_and_modal_hosts():
    auth = ModalProxyAuth()
    assert auth.headers_for("runpod", ENDPOINT) == {}
    assert auth.headers_for("modal", "https://attacker.example.com") == {}
    assert auth.headers_for("modal", "http://203.0.113.9:8080") == {}
    assert auth.headers_for("modal", "https://x--serve.modal.run/path")


def test_modal_proxy_auth_without_tokens_yields_nothing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    assert ModalProxyAuth().headers_for("modal", ENDPOINT) == {}
    assert not ModalProxyAuth().is_configured_for("modal")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "only-id")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    assert ModalProxyAuth().headers_for("modal", ENDPOINT) == {}


def test_explicit_tokens_and_composite_and_static_auth():
    explicit = ModalProxyAuth(token_id="wk-explicit", token_secret="ws-explicit")
    assert explicit.headers_for("modal", ENDPOINT)["Modal-Key"] == "wk-explicit"
    combined = CompositeEndpointAuth(
        explicit, StaticHeaderAuth({"X-Api-Key": "abc123456"}, providers=("runpod",))
    )
    assert combined.headers_for("modal", ENDPOINT) == {
        "Modal-Key": "wk-explicit",
        "Modal-Secret": "ws-explicit",
    }
    assert combined.headers_for("runpod", "http://1.2.3.4:8080") == {"X-Api-Key": "abc123456"}


def test_auth_reprs_never_contain_secrets():
    explicit = ModalProxyAuth(token_id="wk-visible-id-xyz", token_secret="ws-hidden-secret-xyz")
    texts = [
        repr(explicit),
        str(explicit),
        repr(CompositeEndpointAuth(explicit, StaticHeaderAuth({"Authorization": "Bearer hunter2hunter2"}))),
        repr(InferWeave().endpoint_auth),
    ]
    for text in texts:
        for secret in ("wk-visible-id-xyz", "ws-hidden-secret-xyz", "hunter2hunter2", *SECRETS):
            assert secret not in text


# ---------------------------------------------------- inference + health probes


@pytest.mark.asyncio
async def test_inference_requests_carry_modal_proxy_auth(db_path: Path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})

    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    attached = await make_weave(db_path, handler=handler).attach(dep.id)
    await attached.synthesize("hello")
    _assert_auth(dict(seen[0].headers))


@pytest.mark.asyncio
async def test_readiness_and_health_probes_carry_modal_proxy_auth(db_path: Path):
    probe = HeaderRecordingProbe()
    weave = make_weave(db_path, probe=probe)
    with patched_modal():
        # deploy(wait_for_ready=True) must keep working with proxy auth enabled
        dep = await weave.deploy(model=AUDIO_MODEL, provider="modal", wait_for_ready=True)
    assert dep.state == DeploymentState.HEALTHY
    assert probe.headers_seen
    for headers in probe.headers_seen:
        _assert_auth(headers)

    probe.headers_seen.clear()
    await dep.check_health()
    await dep.wait_for_ready()
    with patched_modal():
        await dep.refresh()
        await weave.get_status(dep.id)
    assert len(probe.headers_seen) >= 4
    for headers in probe.headers_seen:
        _assert_auth(headers)


@pytest.mark.asyncio
async def test_attached_handle_probes_are_authenticated_too(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    probe = HeaderRecordingProbe()
    attached = await make_weave(db_path, probe=probe).attach(dep.id)
    await attached.check_health()
    await attached.wait_for_ready()
    with patched_modal():
        await attached.refresh()
    assert len(probe.headers_seen) == 3
    for headers in probe.headers_seen:
        _assert_auth(headers)


@pytest.mark.asyncio
async def test_auth_does_not_mutate_shared_healthcheck_config(db_path: Path):
    probe = HeaderRecordingProbe()
    weave = make_weave(db_path, probe=probe)
    with patched_modal():
        dep = await weave.deploy(model=AUDIO_MODEL, provider="modal", wait_for_ready=True)
    await dep.check_health()
    profile = weave.registry.get(AUDIO_MODEL)
    assert profile.healthcheck.headers == {}
    assert weave.registry.get(AUDIO_MODEL).healthcheck.headers == {}


@pytest.mark.asyncio
async def test_custom_auth_resolver_is_used_for_probes_and_inference(db_path: Path):
    auth = StaticHeaderAuth({"Authorization": "Bearer custom-token-123"}, providers=("modal",))
    probe = HeaderRecordingProbe()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})

    weave = make_weave(db_path, probe=probe, endpoint_auth=auth, handler=handler)
    dep = await deploy_on_modal(weave)
    await dep.check_health()
    await dep.synthesize("hello")
    assert probe.headers_seen[-1]["Authorization"] == "Bearer custom-token-123"
    assert seen[0].headers["authorization"] == "Bearer custom-token-123"
    assert "modal-key" not in seen[0].headers


# ------------------------------------------------------------ secret hygiene


@pytest.mark.asyncio
async def test_secrets_are_never_persisted_in_sqlite(db_path: Path):
    probe = HeaderRecordingProbe()
    weave = make_weave(db_path, probe=probe)
    with patched_modal():
        dep = await weave.deploy(
            model=AUDIO_MODEL,
            provider="modal",
            wait_for_ready=True,
            custom_args={"extra_env": {"HF_TOKEN": "hf_supersecretvalue"}},
        )
    await dep.check_health()
    with patched_modal():
        await dep.refresh()

    raw = b"".join(
        f.read_bytes() for f in db_path.parent.glob("deployments.db*") if f.is_file()
    )
    assert b"iw-modal-" in raw  # sanity: the database really holds the record
    for secret in (*SECRETS, "hf_supersecretvalue"):
        assert secret.encode() not in raw


def test_sanitization_redacts_credential_like_provider_args_and_env():
    record = DeploymentRecord(
        id="x",
        model="m",
        provider="modal",
        options=DeploymentOptions(
            runtime=RuntimeOptions(
                extra_env={"MODAL_PROXY_TOKEN_SECRET": "s1", "SERVICE_AUTH": "s2", "PLAIN": "ok"}
            ),
        ),
    )
    record.options.provider.extra_provider_args = {
        "headers": {"Modal-Secret": "s3", "Authorization": "Bearer s4", "X-Trace": "ok"},
        "proxy_credentials": {"user": "u", "password": "p"},
        "api-key": "s5",
        "cpu": 4,
    }
    clean = record.to_sanitized_record()
    dumped = clean.model_dump_json()
    for secret in ("s1", "s2", "s3", "Bearer s4", "s5", '"p"'):
        assert f'"{secret}"' not in dumped
    assert clean.options.runtime.extra_env["PLAIN"] == "ok"
    assert clean.options.provider.extra_provider_args["cpu"] == 4
    assert clean.options.provider.extra_provider_args["headers"]["X-Trace"] == "ok"


@pytest.mark.asyncio
async def test_secrets_not_exposed_in_errors_logs_or_reprs(
    db_path: Path, caplog: pytest.LogCaptureFixture
):
    def echoing_handler(request: httpx.Request) -> httpx.Response:
        # A hostile/buggy endpoint echoing the request credentials back in its error body
        return httpx.Response(
            401,
            text=f"bad creds: {request.headers['modal-key']} / {request.headers['modal-secret']}",
        )

    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    attached = await make_weave(db_path, handler=echoing_handler).attach(dep.id)

    with caplog.at_level(logging.DEBUG), pytest.raises(InferenceError) as excinfo:
        await attached.synthesize("hello")
    err = excinfo.value
    assert err.status_code == 401
    assert "MODAL_PROXY_TOKEN" in str(err)  # actionable hint
    visible = " ".join(
        [str(err), repr(err), err.response_body or "", err.endpoint or "", repr(attached), repr(attached.status)]
        + [r.getMessage() for r in caplog.records]
    )
    for secret in SECRETS:
        assert secret not in visible
    assert "[REDACTED]" in (err.response_body or "")


# --------------------------------------- provider-side enforcement (web_server)


@pytest.mark.asyncio
async def test_modal_web_server_requires_proxy_auth_by_default(db_path: Path):
    weave = make_weave(db_path)
    with patch("modal.web_server", wraps=modal.web_server) as web_server:
        await deploy_on_modal(weave)
    assert web_server.call_args.kwargs["requires_proxy_auth"] is True


@pytest.mark.asyncio
async def test_modal_public_endpoint_is_an_explicit_opt_out(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    weave = make_weave(db_path)
    with patch("modal.web_server", wraps=modal.web_server) as web_server:
        dep = await deploy_on_modal(weave, custom_args={"requires_proxy_auth": False})
    assert web_server.call_args.kwargs["requires_proxy_auth"] is False
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record is not None and record.options.provider.requires_proxy_auth is False


@pytest.mark.asyncio
async def test_modal_deploy_fails_fast_without_proxy_tokens(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    weave = make_weave(db_path)
    with patched_modal() as mocks, pytest.raises(ProviderAuthError, match="MODAL_PROXY_TOKEN_ID"):
        await weave.deploy(model=AUDIO_MODEL, provider="modal", wait_for_ready=False)
    assert await weave.list_records() == []  # nothing half-created
    mocks.stop_app.assert_not_awaited()


@pytest.mark.asyncio
async def test_modal_dry_run_does_not_require_proxy_tokens(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    dep = await make_weave(db_path).deploy(model=AUDIO_MODEL, provider="modal", dry_run=True)
    assert dep.state == DeploymentState.PROVISIONING


@pytest.mark.asyncio
async def test_sdk_level_endpoint_auth_satisfies_modal_provider_check(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID")
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    auth = ModalProxyAuth(token_id="wk-from-code", token_secret="ws-from-code")
    weave = make_weave(db_path, endpoint_auth=auth)
    dep = await deploy_on_modal(weave)
    assert dep.endpoint_url == ENDPOINT
