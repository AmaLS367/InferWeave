"""Unit tests for healthcheck adapters (HTTPX and Mock)."""

from unittest.mock import AsyncMock

import httpx
import pytest

from inferweave.adapters.healthcheck.httpx_probe import HttpxHealthcheckProbeAdapter
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult


@pytest.mark.asyncio
async def test_mock_probe_default_healthy():
    adapter = MockHealthcheckProbeAdapter()
    res = await adapter.probe("http://localhost:8000/health")

    assert res.is_healthy is True
    assert res.outcome == ProbeOutcome.SUCCESS
    assert res.status_code == 200
    assert adapter.probe_call_count == 1
    assert adapter.probed_urls == ["http://localhost:8000/health"]


@pytest.mark.asyncio
async def test_mock_probe_canned_sequence():
    p1 = ProbeResult(is_healthy=False, outcome=ProbeOutcome.CONNECTION_REFUSED)
    p2 = ProbeResult(is_healthy=True, outcome=ProbeOutcome.SUCCESS, status_code=200)

    adapter = MockHealthcheckProbeAdapter(canned_results=[p1, p2])

    res1 = await adapter.probe("http://localhost:8000/health")
    assert res1.is_healthy is False

    res2 = await adapter.probe("http://localhost:8000/health")
    assert res2.is_healthy is True

    # Fallback to default
    res3 = await adapter.probe("http://localhost:8000/health")
    assert res3.is_healthy is True


@pytest.mark.asyncio
async def test_httpx_probe_success():
    # Use httpx.MockTransport to test without actual network traffic
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ready"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = HttpxHealthcheckProbeAdapter(client=client)
        result = await adapter.probe("http://app.test/health")

        assert result.is_healthy is True
        assert result.outcome == ProbeOutcome.SUCCESS
        assert result.status_code == 200
        assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_httpx_probe_connection_error():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.request.side_effect = httpx.ConnectError("Connection refused")

    adapter = HttpxHealthcheckProbeAdapter(client=mock_client)
    result = await adapter.probe("http://app.test/health")

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.CONNECTION_REFUSED
    assert "Connection refused" in (result.error_message or "")


@pytest.mark.asyncio
async def test_httpx_probe_timeout():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.request.side_effect = httpx.ReadTimeout("Request timed out")

    adapter = HttpxHealthcheckProbeAdapter(client=mock_client)
    result = await adapter.probe("http://app.test/health")

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.TIMEOUT
    assert "timed out" in (result.error_message or "")


@pytest.mark.asyncio
async def test_httpx_probe_close():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()

    adapter = HttpxHealthcheckProbeAdapter(client=mock_client)
    await adapter.close()
    mock_client.aclose.assert_called_once()
