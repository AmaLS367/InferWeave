# Configuration

## Persistence and credentials

| Setting | Default / meaning |
| --- | --- |
| `INFERWEAVE_DEPLOYMENTS_PATH` | `~/.inferweave/deployments.db`; read on repository construction |
| `SqliteDeploymentRepository(db_path=...)` | Overrides environment; inject via `LifecycleService(repository=...)` |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | SDK credentials; Modal also reads `modal setup` profiles |
| `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` | HTTP Proxy Token pair, resolved per request |
| `endpoint_auth=` | Default `CompositeEndpointAuth(ModalProxyAuth())` |

There is no `.env` loader. Export/load values before creating the SDK.
Recovery does not load secrets from SQLite. See [security](../how-to/secure-your-endpoint.md).

## Deployment and runtime

`deploy()` accepts container `env` and structured `custom_args`.
`DeploymentRequest.options` supports lower-level providers; SDK `deploy()`
does not accept `options=`.

| Option | Default / behavior |
| --- | --- |
| `destroy_after_idle_mins` | Omitted: 30; `None`/`0` disables full stop; negatives rejected |
| `autostop_mins` | Legacy alias; differing explicit SDK values raise `ValueError` |
| `scaledown_window_seconds` | Modal container idle: default 1800 seconds; independent of full stop |
| `wait_for_ready` | `True`; poll model endpoint after launch |
| `dry_run` | `False`; `True` simulates without live inference |
| `custom_args['requires_proxy_auth']` | `True` for Modal; `False` exposes HTTP endpoint |
| `custom_args['cleanup_on_failure']` | `False`; `True` stops after readiness timeout |
| `custom_args['timeout_seconds']` | Modal function timeout: 86400; separate from client/readiness |
| `custom_args['cpu']`, `['memory']` | Modal resource overrides |
| `custom_args['allow_spot']` | `True`; permit spot routing offers |
| `custom_args['max_price_per_hour']` | `None`; routing constraint, not billing budget |
| `custom_args['preferred_regions']` | Empty list; routing preferences |
| `custom_args['disk_size_gb']` | `None`; minimum 10 if set |
| `custom_args['autodown']` | `False`; SkyPilot terminate instead of stop |
| `custom_args['autostop_action']` | `stop`, or `down` with autodown |

Legacy `scaledown_window`, `timeout`, `disk_size`, `extra_args` and nested
`provider_args`/`runtime_args` remain accepted. Top-level provider values generally
override nested values; SDK `scaledown_window_seconds=` overrides its custom arg.
Legacy custom idle keys normalize via `DeploymentOptions.from_custom_args`;
prefer SDK parameters and avoid specifying both forms.

`engine_args` and known engine keys (`dtype`, `max_model_len`,
`gpu_memory_utilization`, `tensor_parallel_size`, etc.) render CLI flags.
`extra_cli_args` is literal tokens or shlex-parsed string, not shell code.
`extra_env` merges runtime variables; `provider_args` are backend-specific.
Not all providers honor all settings. Built-in templates pin images/dependencies.
SkyPilot also receives native legacy idle/autodown; Modal scaling is independent
of InferWeave's local watchdog. See [lifecycle](../explanation/lifecycle-and-cost.md).

## Inference and readiness

`InferWeave(inference_config=InferenceConfig(...))` controls both clients:

| Field | Default |
| --- | --- |
| `timeout_seconds` | `300.0` |
| `connect_timeout_seconds` | `10.0` |
| `max_retries` | `6` plus initial attempt |
| `backoff_base_seconds` | `1.0` |
| `backoff_max_seconds` | `30.0` |
| `jitter_ratio` | `0.25` |
| `retry_statuses` | `(502, 503)` |

Timeouts are positive/finite, backoff finite/nonnegative, retries nonnegative,
jitter between 0 and 1. Per-call `timeout` overrides response phase timeout,
not a total deadline. See [cold starts](../how-to/handle-cold-starts.md).

`HealthcheckConfig` defaults: enabled, initial delay 10 seconds, polling timeout
300 seconds, interval 3 seconds, request timeout 5 seconds, expected codes `[200]`,
consecutive successes 1, method `GET`, headers `{}`. Paths/ports vary by
[model](supported-models.md). Readiness uses the profile, not `InferenceConfig`.

`LifecycleService(activity_persist_interval_seconds=30.0)` throttles inference
activity writes; `0` persists every touch. Synchronous activity and successful
probes update memory only. Watchdog intervals depend on the idle policy.
