"""Application service orchestrating healthcheck polling, intervals, retries, and readiness probing."""

import asyncio
import logging
import time
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

from inferweave.adapters.healthcheck.httpx_probe import HttpxHealthcheckProbeAdapter
from inferweave.core.exceptions import HealthcheckError, HealthcheckTimeoutError
from inferweave.domain.healthcheck import (
    HealthEvaluator,
    ProbeResult,
    ReadinessReport,
    ReadinessState,
)
from inferweave.models.profile import HealthcheckConfig
from inferweave.ports.healthcheck import HealthcheckProbePort

logger = logging.getLogger(__name__)


class HealthcheckService:
    """Coordinates endpoint readiness probing, timeouts, retries, and health verification."""

    def __init__(
        self,
        probe_port: HealthcheckProbePort | None = None,
        evaluator: HealthEvaluator | None = None,
    ) -> None:
        self._probe_port = probe_port or HttpxHealthcheckProbeAdapter()
        self._evaluator = evaluator or HealthEvaluator()

    @property
    def probe_port(self) -> HealthcheckProbePort:
        return self._probe_port

    def build_probe_url(self, endpoint_url: str, config: HealthcheckConfig) -> str:
        """Constructs the canonical HTTP/HTTPS probe URL from base endpoint and config."""
        raw_url = endpoint_url.strip()
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"http://{raw_url}"

        parts = urlsplit(raw_url)
        scheme = parts.scheme or "http"
        netloc = parts.netloc

        # If netloc does not contain a port and scheme is http, inject config.port if specified
        if ":" not in netloc and scheme == "http" and config.port:
            netloc = f"{netloc}:{config.port}"

        # Combine path cleanly
        base_path = parts.path.rstrip("/")
        probe_path = config.path if config.path.startswith("/") else f"/{config.path}"
        final_path = f"{base_path}{probe_path}" if base_path else probe_path

        return urlunsplit((scheme, netloc, final_path, parts.query, parts.fragment))

    async def check_health(
        self, endpoint_url: str, config: HealthcheckConfig
    ) -> ProbeResult:
        """Executes a single probe check against an endpoint."""
        probe_url = self.build_probe_url(endpoint_url, config)
        return await self._probe_port.probe(
            url=probe_url,
            method=config.method,
            timeout_seconds=config.request_timeout_seconds,
            headers=config.headers,
            expected_status_codes=config.expected_status_codes,
        )

    async def wait_for_ready(
        self,
        endpoint_url: str | None,
        config: HealthcheckConfig,
        deployment_id: str | None = None,
        on_poll: Callable[[ReadinessReport], None] | None = None,
        timeout_override: float | None = None,
    ) -> ReadinessReport:
        """Polls the endpoint until it responds with a healthy status or timeout expires.

        Args:
            endpoint_url: Base endpoint URL of the running deployment.
            config: Healthcheck configuration defining path, timeouts, and thresholds.
            deployment_id: Optional deployment ID for diagnostics and logging.
            on_poll: Optional callback invoked after every probe attempt.
            timeout_override: Optional custom timeout in seconds overriding config.timeout_seconds.

        Returns:
            ReadinessReport: Final report indicating successful readiness.

        Raises:
            HealthcheckError: If endpoint_url is missing.
            HealthcheckTimeoutError: If the endpoint fails to satisfy readiness before deadline.
        """
        if not config.enabled:
            logger.info("Healthcheck polling disabled in config; skipping probe.")
            return ReadinessReport(
                state=ReadinessState.READY,
                total_probes=0,
                consecutive_successes=0,
                elapsed_seconds=0.0,
            )

        if not endpoint_url or not endpoint_url.strip():
            raise HealthcheckError(
                f"Cannot wait for readiness: deployment '{deployment_id}' has no endpoint URL.",
                deployment_id=deployment_id,
            )

        probe_url = self.build_probe_url(endpoint_url, config)
        total_timeout = (
            float(timeout_override)
            if timeout_override is not None
            else float(config.timeout_seconds)
        )

        logger.info(
            "Starting readiness polling for deployment '%s' at '%s' (timeout=%ss, interval=%ss)",
            deployment_id or "unknown",
            probe_url,
            total_timeout,
            config.probe_interval_seconds,
        )

        # Initial delay before starting the probe loop
        if config.initial_delay_seconds > 0:
            logger.debug(
                "Waiting initial delay of %ss for deployment '%s'...",
                config.initial_delay_seconds,
                deployment_id,
            )
            await asyncio.sleep(config.initial_delay_seconds)

        start_time = time.monotonic()
        deadline = start_time + total_timeout
        history: list[ProbeResult] = []
        last_probe: ProbeResult | None = None

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            probe_timeout = min(config.request_timeout_seconds, remaining)

            probe_result = await self._probe_port.probe(
                url=probe_url,
                method=config.method,
                timeout_seconds=probe_timeout,
                headers=config.headers,
                expected_status_codes=config.expected_status_codes,
            )
            history.append(probe_result)
            last_probe = probe_result
            elapsed = time.monotonic() - start_time

            is_ready, consecutive = self._evaluator.evaluate_readiness(
                history, config.consecutive_successes
            )

            report = ReadinessReport(
                state=ReadinessState.READY if is_ready else ReadinessState.POLLING,
                total_probes=len(history),
                consecutive_successes=consecutive,
                elapsed_seconds=elapsed,
                last_probe=last_probe,
                history=history,
            )

            if on_poll:
                try:
                    on_poll(report)
                except Exception as err:  # noqa: BLE001
                    logger.debug("on_poll callback error: %s", err)

            if is_ready:
                logger.info(
                    "Deployment '%s' readiness verified at '%s' in %.2fs (%d probes)",
                    deployment_id or "unknown",
                    probe_url,
                    elapsed,
                    len(history),
                )
                return report

            # Sleep between probe attempts, avoiding sleeping past the deadline
            if time.monotonic() >= deadline:
                break

            sleep_duration = min(
                float(config.probe_interval_seconds),
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

        elapsed = time.monotonic() - start_time
        last_error = (
            last_probe.error_message
            if last_probe and last_probe.error_message
            else "Readiness probe failed without specific error message"
        )
        logger.warning(
            "Deployment '%s' readiness probe timed out after %.2fs at '%s'. Last error: %s",
            deployment_id or "unknown",
            elapsed,
            probe_url,
            last_error,
        )
        raise HealthcheckTimeoutError(
            endpoint_url=probe_url,
            timeout_seconds=total_timeout,
            total_probes=len(history),
            last_error=last_error,
            deployment_id=deployment_id,
        )

    async def close(self) -> None:
        """Closes the underlying probe port resources."""
        await self._probe_port.close()
