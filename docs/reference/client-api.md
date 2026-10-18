# Client API

The names below are lazy exports from `inferweave`. Provider implementations and
`ProviderRouter` live in `inferweave.providers.*`. Base imports do not load either
provider SDK. Supported Python: 3.11 and 3.12.

## InferWeave

```text
InferWeave(registry: ModelRegistry | None = None,
           router: ProviderRouter | None = None,
           healthcheck_service: HealthcheckService | None = None,
           lifecycle_service: LifecycleService | None = None,
           endpoint_auth: EndpointAuthPort | None = None,
           inference_config: InferenceConfig | None = None,
           inference_http_client: httpx.AsyncClient | None = None,
           accounts: AccountsConfig | AccountManager | str | Path | None = None)

await weave.deploy(
    model: str, provider: str = "auto", strategy: str = "cheapest",
    gpu_type: str | None = None, num_gpus: int | None = None,
    env: dict[str, str] | None = None, autostop_mins: int | None | _Unset = <unset>,
    custom_args: dict[str, Any] | None = None, dry_run: bool = False,
    wait_for_ready: bool = True,
    destroy_after_idle_mins: int | None | _Unset = <unset>,
    scaledown_window_seconds: int | None = None,
    account: str | None = None,
) -> Deployment
```

`accounts` configures per-provider account pools (default: `$INFERWEAVE_ACCOUNTS_FILE`, else
every provider uses its native credentials as the `ambient` account). Deploy leases an account
from the provider's pool, fails over within `max_attempts`, and binds the deployment to that
account; `account=` pins one account of an explicit provider. See
[multiple accounts](../how-to/use-multiple-accounts.md).

The internal sentinel distinguishes omission from `None`; idle default resolves
to 30 minutes. Do not import/pass `_Unset`. Defaults construct registry, router,
health service and SQLite lifecycle service; auth is environment-based and
inference uses `InferenceConfig()`. Deploy resolves/validates/renders/provisions,
registers persistence/lifecycle and optionally waits for readiness.
Readiness failure retains resources unless `cleanup_on_failure=True`.

Injected inference HTTP clients are caller-owned; default transports own pools.
The health adapter closes supplied probe clients on shutdown, so account for
sharing. The SDK has no async context manager; use `try/finally` and `close()`.

| SDK method | Return / behavior |
| --- | --- |
| `await attach(deployment_id: str)` | `Deployment`; rebuild stored handle, no cloud call |
| `await find(model: str \| None = None, provider: str \| None = None)` | `Deployment \| None`; exact model, case-insensitive provider; ambiguity raises |
| `await close()` | `None`; release local resources, retain remote apps |
| `register_model(profile: ModelProfile)` | `None`; register before deploy/attach |
| `list_deployments()` | `list[Deployment]`; local tracked handles |
| `await list_records()` | `list[DeploymentRecord]`; persisted records |
| `await get_usage(deployment_id: str, *, compute_seconds: float \| None = None)` | `UsageReport`; read persisted counters and pricing without cloud calls, including stopped/failed/dry-run records |
| `await stop(deployment_id: str, action: Any \| None = None)` | `None`; stop by ID across restarts |
| `await get_status(deployment_id: str)` | `DeploymentStatus`; provider reconciliation and known-profile probe |
| `await reconcile(min_age_seconds: float = 1800.0, *, confirmed_settled: tuple[str, ...] = ())` | `list[DeploymentRecord]`; clean settled leftovers under their owners; pending remote creation requires explicit provider-history confirmation |
| `account_health()` | `list[AccountHealth]`; nonsecret CredWeave state of pooled accounts |
| `accounts` | `AccountManager`; `await accounts.reset(provider, id)` / `await accounts.authorize(provider, id)` |

Every operation on an existing deployment (status, stop, attach, autostop, inference headers)
uses the account recorded on it, never another one; a removed account raises
`AccountUnavailableError`. Changed control keys also fail for pooled records bound to a
different credential generation. `confirmed_settled` acknowledges that remote creation has
finished or was cancelled; it never bypasses an active provisioning lock.

## Deployment

Properties: `id`, `model`, `provider`, `account`, `state`, `endpoint_url`, `status`,
`workload_type`, `is_healthy`, `autostop_mins`, `last_activity_at`. The status
snapshot includes identity/state/URL/error and `created_at`/`ready_at`.
Constructing `Deployment(status)` alone does not wire SDK callbacks.

```text
await deployment.synthesize(
    text: str, reference_id: str | None = None, format: str = "wav", *,
    references: Sequence[ReferenceAudio] = (), seed: int | None = None,
    temperature: float | None = None, top_p: float | None = None,
    repetition_penalty: float | None = None, chunk_length: int | None = None,
    max_new_tokens: int | None = None, normalize: bool | None = None,
    timeout: float | None = None,
) -> bytes

await deployment.render(
    prompt: str, width: int = 1024, height: int = 1024,
    steps: int | None = None, seed: int | None = None, n: int = 1, *,
    guidance_scale: float | None = None, timeout: float | None = None,
) -> list[bytes]

await deployment.request(
    method: str, path: str, *,
    params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    json: object | None = None, content: bytes | str | None = None,
    headers: Mapping[str, str] | None = None, timeout: float | None = None,
) -> httpx.Response
```

Audio only: `ReferenceAudio(audio: bytes, text: str)` supplies in-context
audio/transcript; `reference_id` selects a voice on the runtime. Formats:
`wav`, `mp3`, headerless `pcm`. Unset tuning fields defer to Fish Speech.
Validation checks signatures/non-empty data, not full audio decoding.

Image only: dimensions/steps/seed/n map to FLUX JSON, returning decoded image
bytes. Validation checks JSON/base64/count, not full image integrity. Wrong
workloads fail before activity; other calls count at start/end, including failure,
and protect local in-flight requests from idle shutdown.

`request()` works on every workload, including `ModelProfile.for_container()`
deployments recovered through `attach()` or `find()` without re-registering the
profile. The response is fully buffered: use `.status_code`, `.headers`, `.json()`
or `.content` for JSON and binary APIs. HTTP failures raise the existing inference
errors. For example, using an active deployment handle:

```python
status = await deployment.request("GET", "/v1/status", params={"verbose": True})
result = await deployment.request("POST", "/v1/predict", json={"inputs": [1, 2]})
binary = await deployment.request(
    "POST", "/v1/process", content=b"raw input",
    headers={"Content-Type": "application/octet-stream"}, timeout=120,
)
output = binary.content
```

Paths are relative to the endpoint's base path, with or without a leading `/`;
query strings are supported. Explicit `params` replace a query string in `path`,
as in httpx; pass a sequence of pairs for repeated query keys. Absolute URLs,
protocol-relative URLs, URL userinfo, backslashes and control characters are
refused. Redirects are bounded and must retain the same scheme, host and port.
Provider auth headers override caller headers regardless of case; `Host` cannot
be overridden. JSON and raw `content` are mutually exclusive. Positive finite
`timeout` overrides the response/write/pool timeout; the configured connection
timeout still applies.

Supported methods: GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS and TRACE.
GET/HEAD/PUT/DELETE/OPTIONS/TRACE retry configured cold-start conditions with
bounded backoff; POST/PATCH are never automatically retried, including when a
redirected result fetch fails. Read/write/pool timeouts are never retried.
Existing audio/image retry behavior is unchanged. Stopped, failed, pending,
provisioning, dry-run and endpoint-less handles cannot make generic requests.
Requests count as activity at start and end, including failure and cancellation;
in-flight requests prevent idle shutdown.

| Handle method | Return / behavior |
| --- | --- |
| `await stop(action: Any \| None = None)` | `None`; default `AutostopAction.STOP`, accepts `"stop"`/`"down"` |
| `await refresh()` | `DeploymentStatus`; provider reconciliation/probe |
| `await check_health()` | `ProbeResult`; authenticated probe (source annotation: `Any`) |
| `await wait_for_ready(timeout_seconds: int \| None = None)` | `DeploymentStatus`; authenticated polling |
| `await usage(*, compute_seconds: float \| None = None)` | `UsageReport`; same reader as `weave.get_usage(id)` |
| `record_activity()` | `None`; synchronous in-memory reset |
| `is_idle()` | `bool`; evaluate policy/in-flight count |

## Transport and auth

`InferenceConfig` is immutable; [configuration](configuration.md) lists all defaults.

```text
InferenceTransport(deployment_id, endpoint_fn, auth_headers_fn=None, config=None,
                   http_client=None, default_port=None, sleep=asyncio.sleep,
                   random_fn=random.random)
InferenceClient(deployment_id, workload_type, transport, on_activity=None,
                on_request_start=None, on_request_end=None, on_usage=None,
                clock=time.perf_counter)
FishSpeechClient(transport)
ImageGenerationClient(transport)
```

Transport: `await post(path, *, content=None, json=None, headers=None, timeout=None)
-> httpx.Response`, `await request(method, path, *, params=None, json=None,
content=None, headers=None, timeout=None) -> httpx.Response`,
`await aclose() -> None`. Audio/image clients have the handle
inference signatures. Unified client dispatches/tracks activity and has `aclose()`.

`EndpointAuthPort.headers_for(provider: str, endpoint_url: str | None,
account: ProviderAccount | None = None) -> dict[str, str]` resolves headers from the
deployment's owning account (pooled accounts carry their own Modal proxy tokens / Lightning
key; the ambient account uses environment variables); `is_configured_for(provider: str,
account: ProviderAccount | None = None) -> bool` checks availability. Implementations: `ModalProxyAuth(token_id=None,
token_secret=None, token_getter=None)`, `StaticHeaderAuth(headers, providers=())`,
`CompositeEndpointAuth(*auths)`, `NoEndpointAuth()`. See [security](../how-to/secure-your-endpoint.md).

## Usage and estimates

`UsageReport`, `DeploymentUsage`, `ComputePricing`, and `CostEstimate` are public,
immutable models exported from `inferweave`. Reports are detached snapshots; read
again for changes from other handles/processes. Reading does not reset idle timers.
The owning account must still be resolvable for live records, as with attach/status.

```python
report = await deployment.usage()
if report.usage is not None:
    print(report.usage.requests, report.usage.successes, report.usage.failures)
    print(report.usage.cancellations, report.usage.unfinished_requests)
    print(report.usage.average_latency_seconds)
print(report.cost.estimated_cost_usd)  # None for live deployments without duration
print(report.cost.unavailable_reasons)

# A scenario using an externally obtained compute duration, NOT request latency.
scenario = await weave.get_usage(deployment.id, compute_seconds=90)
print(scenario.pricing, scenario.cost.estimated_cost_usd)
```

`DeploymentUsage` counts generic HTTP, audio and image calls at the unified client
boundary. Wrong-workload/stopped/dry-run handle rejection is excluded. Calls that
enter that boundary but fail local payload validation count as failures. Success
requires the entire operation, including audio/image decoding, to succeed. Retries
and redirects are one logical request; a render of multiple images is one request.
Health probes, direct low-level client/transport calls and requests from outside
this SDK are excluded.

`total_latency_seconds` sums monotonic elapsed time of completed calls, including
retry delays and client-side start bookkeeping, excluding terminal persistence.
`average_latency_seconds` divides by successes + failures + cancellations and is
`None` until one call finishes. Concurrent request latencies overlap and cannot be
added up to derive GPU consumption.

`request_body_bytes` counts serialized HTTP body bytes submitted to the transport;
delivery is not guaranteed. `response_body_bytes` counts fully buffered decoded
HTTP response body bytes, including failed attempts/redirects. These include JSON,
base64 or MessagePack envelopes; they exclude headers, compression/TLS/network
overhead and image-result decoding. `incomplete_transfers` counts calls with an
unmeasurable transport outcome (connection/read failure or cancellation). Partial
response bodies are not included. These counters are not provider egress billing.

Counters persist at start and completion, independently of throttled idle activity.
A hard process kill can leave an `unfinished_requests` entry. It remains unknown
rather than being replayed or converted to a failure; this field is not a distributed
live in-flight gauge. Legacy records have `usage=None` until their first observed
call; `tracking_started_at` identifies the start of available history. Earlier
requests cannot be recovered. Persistence failure is logged by error type and does
not change inference results; usage may then be incomplete. Cancellation drains
pending metering writes before returning. SQLite transactions and JSON file locks
protect increments across processes; lifecycle saves preserve current usage. Custom
repositories must implement atomic `update()` and preserve usage in `save()` to
support metering; the default port refuses an unsafe read/modify/save fallback.
Older custom repositories keep local activity tracking and attach support, but
cannot persist activity updates or meter requests until they implement that contract;
their report's `usage` is `None`.

`ComputePricing` snapshots the candidate's provider, requested GPU type/count,
catalog instance/region, whole-configuration USD hourly rate, spot/on-demand kind,
billing increment and capture time. `source="catalog"` is an indicative quote,
including when the built-in static catalog supplies it; it is not a current provider
price confirmation. Unmatched/ambiguous GPU overrides, count/region mismatches and
missing spot rates remain unavailable; another GPU or on-demand rate is never
substituted. Provider failover records each candidate's own quote. Arbitrary catalog
metadata, endpoint URLs, headers and credentials are not included.

`compute_seconds` must be finite and nonnegative and represents one allocation's
duration, not GPU-seconds. Its scenario rounds **once** upward to the captured billing
increment and multiplies by the configuration rate / 3600; GPU count is already in
that rate. For multiple containers, replicas or billing sessions, calculate separate
scenarios using their externally known durations and applicable quotes. This API
does not infer replicas or aggregate rounding across sessions. Missing rate or
duration returns `estimated_cost_usd=None` with `unavailable_reasons`; uncertain
provisioning/cleanup also prevents an estimate. Dry runs report a zero compute
estimate without making cloud calls.

`measured_gpu_seconds` and `actual_billed_cost_usd` remain `None`: no provider
consumption/invoice integration is implemented. See [lifecycle and cost](../explanation/lifecycle-and-cost.md)
for provider differences and exclusions.

## Errors

These public errors subclass `InferWeaveError`:

| Exception | Meaning |
| --- | --- |
| `DeploymentNotFoundError` | Missing ID; `deployment_id` |
| `DeploymentNotActiveError` | Stopped/failed/dry-run attach; `deployment_id`, `reason` |
| `AmbiguousDeploymentError` | Multiple matches; `model`, `provider`, `candidate_ids` |
| `ProviderAuthError` | Required endpoint credentials missing |
| `AccountUnavailableError` | ProviderAuthError: the deployment's owning account is not configured; `provider`, `account_id`, `deployment_id` |
| `AccountConfigurationError` | Invalid accounts configuration (never echoes secrets) |
| `NoAccountAvailableError` | Every attempt failed or no account was eligible; `attempts`, `retry_after` |
| `ProviderOperationError` | Classified control-plane failure; `kind` (`FailureKind`), `status_code`, `retry_after`, `resource_may_exist` |
| `ProvisioningUncertainError` | DeploymentError: a billed resource may exist; nothing else was tried; run `reconcile()` |
| `InferenceError` | Base error; optional sanitized `deployment_id`, `status_code`, `endpoint`, `response_body` |
| `EndpointNotReadyError` | InferenceError: missing/stopped URL or exhausted retries |
| `InferenceTimeoutError` | InferenceError: HTTP timeout, no replay |
| `InvalidInferenceResponseError` | InferenceError: malformed response |
| `UnsupportedWorkloadError` | InferenceError: wrong workload; `workload_type`, `operation` |
| `HealthcheckError` | Probe/readiness configuration error |
| `HealthcheckTimeoutError` | Readiness exhausted; `endpoint_url`, `timeout_seconds`, `total_probes`, `last_error` |

Existing model/provider/hardware/deployment errors remain exported. Invalid local
arguments raise `ValueError`. Cancellation propagates normally.
