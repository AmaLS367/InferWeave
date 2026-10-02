# Handle cold starts

Modal scale-to-zero removes idle GPU containers but keeps the app and URL.
The next request loads runtime/weights; connection failures or temporary 502/503
can precede success. A stopped app requires a new deployment.

```python
from inferweave import InferWeave, InferenceConfig

weave = InferWeave(inference_config=InferenceConfig(
    timeout_seconds=180, connect_timeout_seconds=10,
    max_retries=4, backoff_base_seconds=1, backoff_max_seconds=8,
    jitter_ratio=0.25,
))
```

Defaults: response timeout 300 seconds, connect timeout 10 seconds, 6 retries
after the initial attempt, exponential backoff from 1 second capped at 30 seconds,
with up to 25% downward jitter. Numeric `Retry-After` can increase a delay to the
same cap. Durations must be finite. Per-call `timeout=` overrides response timeout,
not connect timeout or readiness policy.

Only connection errors/timeouts, dropped remote connections (`RemoteProtocolError`)
and configured statuses (default 502/503) retry. Read/write/pool timeouts raise
`InferenceTimeoutError` immediately; rejected requests/other HTTP failures raise
`InferenceError` without retries. Exhaustion, missing URLs and stopped handles
raise `EndpointNotReadyError`. Malformed successful bodies raise
`InvalidInferenceResponseError`.

```python
from inferweave import EndpointNotReadyError, InferenceTimeoutError

async def speak(deployment):
    try:
        return await deployment.synthesize("Hello", timeout=120)
    except EndpointNotReadyError:
        # Present "Starting the model" and allow a deliberate later retry.
        raise
    except InferenceTimeoutError:
        # Present a timeout; avoid automatically replaying costly generation.
        raise
```

Timeouts are HTTP phase limits per attempt/redirect, not one total deadline.
Use `asyncio.timeout` when you need a wall-clock limit. Cancellation releases
the local in-flight marker. Replaying POST after a dropped connection may
duplicate work if the server already accepted it; there is no idempotency key.

Keep the event loop responsive and show progress while the model starts.
Do not provision another app just because an existing one is cold.
Readiness is separate: `deploy(wait_for_ready=True)` uses `HealthcheckConfig`;
`wait_for_ready(timeout_seconds=...)` overrides its polling deadline. Failure
raises `HealthcheckTimeoutError`; `cleanup_on_failure=True` opts into cleanup.
Deterministic tests cover wakeup retries without waiting minutes for live scale-down.
