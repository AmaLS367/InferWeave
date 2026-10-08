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
```

Audio only: `ReferenceAudio(audio: bytes, text: str)` supplies in-context
audio/transcript; `reference_id` selects a voice on the runtime. Formats:
`wav`, `mp3`, headerless `pcm`. Unset tuning fields defer to Fish Speech.
Validation checks signatures/non-empty data, not full audio decoding.

Image only: dimensions/steps/seed/n map to FLUX JSON, returning decoded image
bytes. Validation checks JSON/base64/count, not full image integrity. Wrong
workloads fail before activity; other calls count at start/end, including failure,
and protect local in-flight requests from idle shutdown.

| Handle method | Return / behavior |
| --- | --- |
| `await stop(action: Any \| None = None)` | `None`; default `AutostopAction.STOP`, accepts `"stop"`/`"down"` |
| `await refresh()` | `DeploymentStatus`; provider reconciliation/probe |
| `await check_health()` | `ProbeResult`; authenticated probe (source annotation: `Any`) |
| `await wait_for_ready(timeout_seconds: int \| None = None)` | `DeploymentStatus`; authenticated polling |
| `record_activity()` | `None`; synchronous in-memory reset |
| `is_idle()` | `bool`; evaluate policy/in-flight count |

## Transport and auth

`InferenceConfig` is immutable; [configuration](configuration.md) lists all defaults.

```text
InferenceTransport(deployment_id, endpoint_fn, auth_headers_fn=None, config=None,
                   http_client=None, default_port=None, sleep=asyncio.sleep,
                   random_fn=random.random)
InferenceClient(deployment_id, workload_type, transport, on_activity=None,
                on_request_start=None, on_request_end=None)
FishSpeechClient(transport)
ImageGenerationClient(transport)
```

Transport: `await post(path, *, content=None, json=None, headers=None, timeout=None)
-> httpx.Response`, `await aclose() -> None`. Audio/image clients have the handle
inference signatures. Unified client dispatches/tracks activity and has `aclose()`.

`EndpointAuthPort.headers_for(provider: str, endpoint_url: str | None,
account: ProviderAccount | None = None) -> dict[str, str]` resolves headers from the
deployment's owning account (pooled accounts carry their own Modal proxy tokens / Lightning
key; the ambient account uses environment variables); `is_configured_for(provider: str,
account: ProviderAccount | None = None) -> bool` checks availability. Implementations: `ModalProxyAuth(token_id=None,
token_secret=None, token_getter=None)`, `StaticHeaderAuth(headers, providers=())`,
`CompositeEndpointAuth(*auths)`, `NoEndpointAuth()`. See [security](../how-to/secure-your-endpoint.md).

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
