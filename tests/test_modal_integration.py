"""Live integration tests for Modal deployment.

These tests communicate with real Modal cloud infrastructure and require
valid Modal credentials (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET).
They are skipped in CI / local test runs if credentials are not configured.
"""

import os

import pytest

from inferweave import DeploymentState, InferWeave


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MODAL_TOKEN_ID"),
    reason="Live Modal integration test requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables",
)
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
