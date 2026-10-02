"""Live integration tests for Modal deployment.

These tests communicate with real Modal cloud infrastructure and require
valid Modal credentials (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET).
They are skipped in CI / local test runs if credentials are not configured.

Modal endpoints are protected with proxy auth, so these tests additionally need a Modal proxy
auth token (workspace settings -> Proxy Auth Tokens) exported as MODAL_PROXY_TOKEN_ID and
MODAL_PROXY_TOKEN_SECRET.
"""

import os

import pytest

from inferweave import DeploymentState, InferWeave, UnsupportedWorkloadError

_LIVE_MODAL = pytest.mark.skipif(
    not (
        os.getenv("MODAL_TOKEN_ID")
        and os.getenv("MODAL_PROXY_TOKEN_ID")
        and os.getenv("MODAL_PROXY_TOKEN_SECRET")
    ),
    reason=(
        "Live Modal integration test requires MODAL_TOKEN_ID/MODAL_TOKEN_SECRET and "
        "MODAL_PROXY_TOKEN_ID/MODAL_PROXY_TOKEN_SECRET environment variables"
    ),
)


@pytest.mark.integration
@_LIVE_MODAL
@pytest.mark.asyncio
async def test_live_modal_deployment():
    """Performs a live deployment to Modal cloud and verifies real healthcheck probe."""
    weave = InferWeave()

    # Use a lightweight audio model or custom profile for quick cold start
    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="modal",
        autostop_mins=5,
        wait_for_ready=True,
    )

    try:
        assert deployment.id.startswith("iw-modal-")
        assert deployment.endpoint_url is not None
        assert deployment.state == DeploymentState.HEALTHY
        assert deployment.is_healthy is True

        # Check real health probe
        probe = await deployment.check_health()
        assert probe.is_healthy is True
    finally:
        await deployment.stop()


@pytest.mark.integration
@_LIVE_MODAL
@pytest.mark.asyncio
async def test_live_modal_synthesize_and_reattach_after_restart():
    """Deploy, synthesize, "restart" (fresh InferWeave), attach, synthesize again, stop.

    Exercises proxy-authenticated readiness + inference, cold-start retries, scale-to-zero
    config independent from the destroy timer, and recovery from the persisted record.
    """
    weave1 = InferWeave()
    deployment = await weave1.deploy(
        model="fish-s2-pro",
        provider="modal",
        wait_for_ready=True,
        destroy_after_idle_mins=60,
        scaledown_window_seconds=300,
    )
    attached = None
    try:
        assert deployment.state == DeploymentState.HEALTHY
        assert (await deployment.check_health()).is_healthy is True

        audio = await deployment.synthesize("InferWeave integration test")
        assert audio[:4] == b"RIFF"
        with pytest.raises(UnsupportedWorkloadError):
            await deployment.render("not an image model")

        # Simulated process restart: a brand new SDK instance recovers the deployment.
        weave2 = InferWeave()
        attached = await weave2.attach(deployment.id)
        assert attached is not deployment
        assert attached.endpoint_url == deployment.endpoint_url
        assert attached.status.created_at == deployment.status.created_at
        assert attached.autostop_mins == 60
        assert attached.last_activity_at is not None

        recovered_audio = await attached.synthesize("Recovered deployment")
        assert recovered_audio[:4] == b"RIFF"
        assert (await weave2.find(model="fish-s2-pro", provider="modal")) is attached
    finally:
        await (attached or deployment).stop()
