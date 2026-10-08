"""Regression gaps found while auditing the 0.2.0 release."""

import asyncio
import logging
import sqlite3

import httpx
import pytest
from support import deploy_on_modal, make_weave

from inferweave import (
    Deployment,
    DeploymentRecord,
    DeploymentStatus,
    InferenceConfig,
    InferenceTransport,
    SqliteDeploymentRepository,
)
from inferweave.adapters.healthcheck import HttpxHealthcheckProbeAdapter
from inferweave.clients.audio import FishSpeechClient
from inferweave.core.exceptions import InferenceError


@pytest.mark.parametrize(
    "field",
    [
        "timeout_seconds",
        "connect_timeout_seconds",
        "backoff_base_seconds",
        "backoff_max_seconds",
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_inference_limits_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        InferenceConfig(**{field: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
async def test_invalid_per_call_timeout_does_not_make_request(value):
    def forbidden(request):
        pytest.fail("Invalid timeouts must not issue HTTP requests")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        transport = InferenceTransport(
            "x", lambda: "https://example.test", http_client=client
        )
        with pytest.raises(ValueError, match="positive and finite"):
            await FishSpeechClient(transport).synthesize("hi", timeout=value)


@pytest.mark.asyncio
async def test_text_error_is_not_accepted_as_headerless_pcm():
    from inferweave import InvalidInferenceResponseError

    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, text="proxy error"),
    )) as client:
        transport = InferenceTransport("x", lambda: "https://app.test", http_client=client)
        with pytest.raises(InvalidInferenceResponseError):
            await FishSpeechClient(transport).synthesize("hi", format="pcm")


@pytest.mark.asyncio
async def test_health_probe_follows_same_origin_redirect_with_auth():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["modal-secret"] == "placeholder"
        return (httpx.Response(303, headers={"location": "/ready"})
                if len(seen) == 1 else httpx.Response(200))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HttpxHealthcheckProbeAdapter(client=client).probe(
            "https://app.modal.run/health", headers={"Modal-Secret": "placeholder"},
        )
    assert result.is_healthy and len(seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["https://foreign.test/health", "http://app.modal.run/health"]
)
async def test_health_probe_refuses_cross_origin_credentials(target):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": target})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        result = await HttpxHealthcheckProbeAdapter(client=client).probe(
            "https://app.modal.run/health",
            headers={"Modal-Secret": "test-placeholder"},
        )
    assert len(seen) == 1
    assert not result.is_healthy and "cross-origin" in result.error_message


@pytest.mark.asyncio
async def test_health_probe_keeps_auth_for_same_origin_and_bounds_redirects():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["modal-secret"] == "placeholder"
        return httpx.Response(302, headers={"location": "/loop"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HttpxHealthcheckProbeAdapter(client=client).probe(
            "https://app.modal.run/health",
            headers={"Modal-Secret": "placeholder"},
        )
    assert len(seen) == 6
    assert "Too many" in result.error_message


@pytest.mark.asyncio
async def test_probe_diagnostics_redact_headers_and_endpoint(caplog):
    endpoint = "https://user:password@app.modal.run/health?token=hidden"

    def handler(request):
        raise httpx.ConnectError(f"Cannot connect to {endpoint} with short-key xy")

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await HttpxHealthcheckProbeAdapter(client=client).probe(
                endpoint,
                headers={"X-Api-Key": "xy"},
            )
    visible = str(result) + caplog.text
    for secret in ("password", "hidden", "xy"):
        assert secret not in visible


def test_endpoint_reprs_do_not_show_url_secrets():
    endpoint = "https://user:password@app.modal.run/path?token=hidden#fragment"
    transport = InferenceTransport("x", lambda: endpoint)
    deployment = Deployment(
        DeploymentStatus(id="x", model="m", provider="modal", endpoint_url=endpoint)
    )
    for obj in (transport, deployment):
        assert "password" not in repr(obj) and "hidden" not in repr(obj)


@pytest.mark.asyncio
async def test_short_custom_auth_secret_is_redacted_from_error_body():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, text="bad key xy"),
        )
    ) as client:
        transport = InferenceTransport(
            "x",
            lambda: "https://app.test",
            auth_headers_fn=lambda: {"X-Api-Key": "xy"},
            http_client=client,
        )
        with pytest.raises(InferenceError) as error:
            await transport.post("/v1/tts")
        assert "xy" not in error.value.response_body


@pytest.mark.asyncio
async def test_cancelled_inference_releases_idle_guard(tmp_path):
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.Event().wait()

    weave = make_weave(tmp_path / "state.db", handler=handler)
    try:
        deployment = await deploy_on_modal(weave)
        task = asyncio.create_task(deployment.synthesize("hi"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert weave.lifecycle_service.get_state(deployment.id).in_flight_requests == 0
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_sqlite_record_with_missing_release_fields_can_recover(tmp_path):
    path = tmp_path / "state.db"
    weave = make_weave(path)
    try:
        deployed = await deploy_on_modal(weave)
    finally:
        await weave.close()
    # Model a 0.1.0 serialized row, where optional fields are absent, not null.
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                "UPDATE deployments SET data_json = json_remove(data_json, '$.last_activity_at', '$.workload_type')"
            )
    finally:
        conn.close()
    restarted = make_weave(path)
    try:
        attached = await restarted.attach(deployed.id)
        assert attached.last_activity_at is not None
        assert attached.workload_type.value == "audio"
        assert (await restarted.list_records())[
            0
        ].last_activity_at == attached.last_activity_at
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_sqlite_connections_close_after_every_operation(tmp_path, monkeypatch):
    connections = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs["check_same_thread"] = False  # Inspect closure from the test thread.
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    repo = SqliteDeploymentRepository(tmp_path / "state.db")
    await repo.save(DeploymentRecord(id="x", model="m", provider="modal"))
    assert await repo.get("x")
    assert await repo.get("absent") is None
    assert await repo.list_all()
    await repo.delete("x")
    repo.clear()
    assert len(connections) == 7
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_pre_accounts_record_defaults_to_ambient_account_and_can_be_stopped(tmp_path):
    """A 0.2.0 row has no account/resource fields: it is owned by the ambient account."""
    from unittest.mock import patch

    path = tmp_path / "state.db"
    weave = make_weave(path)
    try:
        deployed = await deploy_on_modal(weave)
    finally:
        await weave.close()
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                "UPDATE deployments SET data_json = json_remove(data_json, "
                "'$.account', '$.resource', '$.needs_reconciliation')"
            )
    finally:
        conn.close()

    restarted = make_weave(path)
    try:
        attached = await restarted.attach(deployed.id)
        assert attached.account == "ambient"
        with patch("modal.experimental.stop_app") as mock_stop:
            await attached.stop()
        assert mock_stop.call_args.args == (deployed.id,)
        assert mock_stop.call_args.kwargs == {"environment_name": None, "client": None}
    finally:
        await restarted.close()
