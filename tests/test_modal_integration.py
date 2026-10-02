"""Opt-in protected Modal TTS/recovery flow; provisions a real GPU and incurs cost."""

import io
import os
import wave

import httpx
import pytest

from inferweave import DeploymentState, InferWeave


def _skip_reason() -> str | None:
    required = (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
    )
    missing = [name for name in required if not os.getenv(name)]
    return "Missing live Modal credentials: " + ", ".join(missing) if missing else None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or ""),
]


def _assert_wav(audio: bytes) -> None:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() > 0
        assert wav.getframerate() > 0
        assert wav.getnframes() > 0
        assert wav.readframes(wav.getnframes())


@pytest.mark.asyncio
async def test_live_modal_synthesize_and_reattach_after_restart(tmp_path, monkeypatch):
    """Readiness, denied anonymous access, valid TTS, SQLite recovery and full stop."""
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "live.db"))
    weave1 = InferWeave()
    weave2 = None
    try:
        deployment = await weave1.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=True,
            destroy_after_idle_mins=60,
            scaledown_window_seconds=300,
        )
        assert deployment.state == DeploymentState.HEALTHY
        assert (await deployment.check_health()).is_healthy
        assert deployment.endpoint_url
        async with httpx.AsyncClient() as client:
            anonymous = await client.get(deployment.endpoint_url + "/v1/health")
            assert anonymous.status_code in (401, 403)
        _assert_wav(await deployment.synthesize("InferWeave integration test"))
        records = await weave1.list_records()
        assert len(records) == 1 and records[0].id == deployment.id
        assert records[0].options.provider.requires_proxy_auth
        assert records[0].last_activity_at is not None
        await weave1.close()  # Simulate shutdown: no old watchdog or connection pools.

        weave2 = InferWeave()
        attached = await weave2.attach(deployment.id)
        assert attached is not deployment
        assert attached.endpoint_url == deployment.endpoint_url
        assert attached.status.created_at == deployment.status.created_at
        assert attached.status.ready_at == deployment.status.ready_at
        assert attached.autostop_mins == 60
        assert attached.last_activity_at == records[0].last_activity_at
        assert (await weave2.find(model="fish-s2-pro", provider="modal")) is attached
        _assert_wav(await attached.synthesize("Recovered deployment"))
        await attached.stop()
        assert attached.state == DeploymentState.STOPPED
        assert (await weave2.list_records())[0].state == DeploymentState.STOPPED
    finally:
        # Also clean up if deploy() raised during readiness, before returning a handle.
        try:
            for record in await weave1.list_records():
                if record.state != DeploymentState.STOPPED:
                    await (weave2 or weave1).stop(record.id)
        finally:
            try:
                if weave2 is not None:
                    await weave2.close()
            finally:
                await weave1.close()
