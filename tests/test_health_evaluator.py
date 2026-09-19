"""Unit tests for pure domain HealthEvaluator logic."""

import httpx

from inferweave.domain.healthcheck import (
    HealthEvaluator,
    ProbeOutcome,
    ProbeResult,
)


def test_evaluator_probe_success_default_code():
    evaluator = HealthEvaluator()
    result = evaluator.evaluate_probe(status_code=200, latency_ms=12.5)

    assert result.is_healthy is True
    assert result.outcome == ProbeOutcome.SUCCESS
    assert result.status_code == 200
    assert result.latency_ms == 12.5
    assert result.error_message is None


def test_evaluator_probe_custom_status_codes():
    evaluator = HealthEvaluator()
    result = evaluator.evaluate_probe(
        status_code=204, latency_ms=8.0, expected_status_codes=[200, 204]
    )

    assert result.is_healthy is True
    assert result.outcome == ProbeOutcome.SUCCESS
    assert result.status_code == 204


def test_evaluator_probe_unexpected_status_code():
    evaluator = HealthEvaluator()
    result = evaluator.evaluate_probe(
        status_code=503, latency_ms=15.0, expected_status_codes=[200]
    )

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.HTTP_ERROR
    assert result.status_code == 503
    assert "503" in (result.error_message or "")


def test_evaluator_probe_connection_refused_error():
    evaluator = HealthEvaluator()
    err = httpx.ConnectError("Connection refused by peer")
    result = evaluator.evaluate_probe(latency_ms=2.0, error=err)

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.CONNECTION_REFUSED
    assert "Connection refused" in (result.error_message or "")


def test_evaluator_probe_timeout_error():
    evaluator = HealthEvaluator()
    err = httpx.ReadTimeout("The read operation timed out")
    result = evaluator.evaluate_probe(latency_ms=5000.0, error=err)

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.TIMEOUT
    assert "timed out" in (result.error_message or "")


def test_evaluator_probe_unknown_error():
    evaluator = HealthEvaluator()
    err = RuntimeError("Unexpected internal crash")
    result = evaluator.evaluate_probe(latency_ms=1.0, error=err)

    assert result.is_healthy is False
    assert result.outcome == ProbeOutcome.UNKNOWN_ERROR


def test_evaluator_readiness_empty_history():
    evaluator = HealthEvaluator()
    is_ready, consecutive = evaluator.evaluate_readiness([], required_consecutive_successes=1)
    assert is_ready is False
    assert consecutive == 0


def test_evaluator_readiness_consecutive_threshold():
    evaluator = HealthEvaluator()
    success = ProbeResult(is_healthy=True, outcome=ProbeOutcome.SUCCESS, status_code=200)
    failure = ProbeResult(is_healthy=False, outcome=ProbeOutcome.CONNECTION_REFUSED)

    # 1 success, target 1 -> ready
    is_ready, count = evaluator.evaluate_readiness([success], required_consecutive_successes=1)
    assert is_ready is True
    assert count == 1

    # 1 success, target 2 -> not ready
    is_ready, count = evaluator.evaluate_readiness([success], required_consecutive_successes=2)
    assert is_ready is False
    assert count == 1

    # failure then 2 successes, target 2 -> ready
    is_ready, count = evaluator.evaluate_readiness(
        [failure, success, success], required_consecutive_successes=2
    )
    assert is_ready is True
    assert count == 2

    # 2 successes then failure -> consecutive resets to 0
    is_ready, count = evaluator.evaluate_readiness(
        [success, success, failure], required_consecutive_successes=2
    )
    assert is_ready is False
    assert count == 0
