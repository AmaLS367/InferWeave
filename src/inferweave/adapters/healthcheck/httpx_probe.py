"""HTTPX-backed health probe adapter."""

import logging
import time
from types import TracebackType
from typing import Self
from urllib.parse import urljoin, urlsplit

import httpx

from inferweave.clients.transport import redact, safe_endpoint
from inferweave.domain.healthcheck import HealthEvaluator, ProbeResult
from inferweave.ports.healthcheck import HealthcheckProbePort

logger = logging.getLogger(__name__)


class HttpxHealthcheckProbeAdapter(HealthcheckProbePort):
    """Executes non-blocking HTTP probes using HTTPX with persistent connection reuse."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        evaluator: HealthEvaluator | None = None,
        verify_ssl: bool = True,
    ) -> None:
        self._custom_client = client is not None
        self._client = client
        self._evaluator = evaluator or HealthEvaluator()
        self._verify_ssl = verify_ssl

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client and self._client is not None:
            return self._client
        is_closed = getattr(self._client, "is_closed", False)
        if isinstance(is_closed, bool) and is_closed:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                verify=self._verify_ssl,
                follow_redirects=True,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def probe(
        self,
        url: str,
        method: str = "GET",
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
        expected_status_codes: list[int] | None = None,
    ) -> ProbeResult:
        """Executes a single HTTP probe against the specified URL."""
        client = self._get_client()
        start = time.perf_counter()

        timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 3.0))

        try:
            origin = urlsplit(url)[:2]
            for _ in range(6):
                response = await client.request(
                    method=method.upper(), url=url, timeout=timeout,
                    headers=headers or {}, follow_redirects=False,
                )
                location = response.headers.get("location")
                if not response.is_redirect or not location:
                    break
                target = urljoin(url, location)
                if urlsplit(target)[:2] != origin:
                    raise httpx.HTTPError("Refusing a cross-origin health probe redirect.")
                url = target
            else:
                raise httpx.HTTPError("Too many health probe redirects.")
            latency = (time.perf_counter() - start) * 1000.0
            return self._evaluator.evaluate_probe(
                status_code=response.status_code,
                latency_ms=latency,
                expected_status_codes=expected_status_codes,
            )
        except Exception as err:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000.0
            result = self._evaluator.evaluate_probe(
                latency_ms=latency, error=err,
                expected_status_codes=expected_status_codes,
            )
            result.error_message = (result.error_message or "").replace(url, safe_endpoint(url))
            result.error_message = redact(result.error_message, headers or {})
            logger.debug(
                "Probe to '%s' failed (%.2fms): %s", safe_endpoint(url), latency,
                result.error_message,
            )
            return result

    async def close(self) -> None:
        """Closes the underlying HTTP client if managed internally."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
