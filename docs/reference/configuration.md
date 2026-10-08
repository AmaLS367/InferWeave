# Configuration

## Persistence and credentials

| Setting | Default / meaning |
| --- | --- |
| `INFERWEAVE_DEPLOYMENTS_PATH` | `~/.inferweave/deployments.db`; read on repository construction |
| `SqliteDeploymentRepository(db_path=...)` | Overrides environment; inject via `LifecycleService(repository=...)` |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | SDK credentials; Modal also reads `modal setup` profiles |
| `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` | HTTP Proxy Token pair, resolved per request |
| `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY` | User SDK credentials; same key authenticates Lightning HTTP endpoints as Bearer |
| `LIGHTNING_TEAMSPACE` | Selected `owner/teamspace` for Lightning, unless typed option overrides it |
| `LIGHTNING_ORG` | Optional owner for a bare teamspace; unnecessary with a full slug |
| `INFERWEAVE_LIGHTNING_INTEGRATION` | Unset/`0` skips paid Lightning tests; `1` explicitly opts in |
| `endpoint_auth=` | Default `CompositeEndpointAuth(ModalProxyAuth(), LightningEndpointAuth())`; resolvers receive the deployment's owning account |
| `INFERWEAVE_ACCOUNTS_FILE` | Accounts configuration (YAML/JSON) used when `InferWeave(accounts=...)` is omitted; CLI `--accounts` |
| `INFERWEAVE_LIGHTNING_PYTHON` | Interpreter for Lightning worker processes (an isolated venv with `lightning-sdk`); default: current Python |
| `INFERWEAVE_SKYPILOT_PYTHON` | Interpreter for SkyPilot worker processes (an isolated venv with `skypilot`); default: current Python |

The single-account variables above are the **ambient** account of each provider. They are used
for providers without an account pool and for deployments created before pools existed.

There is no `.env` loader. Export/load values before creating the SDK.
Recovery does not load secrets from SQLite. See [security](../how-to/secure-your-endpoint.md).

## Accounts

`InferWeave(accounts=...)` accepts an `AccountsConfig`, a path to a YAML/JSON file, or an
`AccountManager`. See [multiple accounts](../how-to/use-multiple-accounts.md) for examples.

| Field | Default / behavior |
| --- | --- |
| `max_attempts` | `3`; provisioning attempts per deploy across accounts and (`auto`) providers |
| `state_dir` | `~/.inferweave/accounts`; private per-account homes for Lightning/SkyPilot workers |
| `providers.<name>.strategy` | `round_robin`; also `weighted`, `failover`, `least_used`, `least_recently_used`, `random` |
| `providers.<name>.accounts[]` | `id`, `env` (`secret field: VARIABLE_NAME`), `weight`, `priority`, `max_concurrency`, `metadata` and provider metadata keys (`teamspace`, `environment`) |
| `providers.<name>.credentials_file` | CredWeave JSON credentials file, hot-reloaded; relative to the config file |
| `providers.<name>.cooldown_seconds` | `60`; cooldown after a transient failure |
| `providers.<name>.max_consecutive_failures` | `3`; transient failures before an account is unhealthy |
| `providers.<name>.permission_cooldown_seconds` | `900`; parking time after HTTP 403 |
| `providers.<name>.quota_cooldown_seconds` | `3600`; parking time after quota exhaustion without a reset hint |

Secret fields per provider: Modal `token_id`, `token_secret`, optional pair `proxy_token_id`,
`proxy_token_secret`; Lightning `user_id`, `api_key`; RunPod and Vast.ai `api_key`.

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
| `custom_args['cleanup_on_failure']` | `False`; `True` stops after readiness timeout; Lightning always cleans failed deploy/readiness |
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
Modal clears the inherited image entrypoint and launches `RuntimeSpec.run_args`.
Local Python source is mounted after all image build steps. Custom templates can
set `RuntimeSpec.metadata['modal_setup_dockerfile_commands']` to a list of
Dockerfile directives passed to `Image.from_registry()` before runtime setup.
The Fish S2 template uses this to expose the image's system `python3` as `python`
for Modal; its model server still runs under `/app/.venv/bin/python`.
SkyPilot also receives native legacy idle/autodown; Modal scaling is independent
of InferWeave's local watchdog. See [lifecycle](../explanation/lifecycle-and-cost.md).

## Inference and readiness

Lightning settings use `custom_args={"lightning": {...}}`, mapped to typed
`LightningOptions`: `teamspace=None` (environment slug), `min_replicas=0`,
`max_replicas=1`, `idle_threshold_seconds=300`. Minimum must not exceed maximum.
Replica scaling is independent of `destroy_after_idle_mins`.
Deployment is on-demand (`spot=False`), endpoints always require user `ApiKeyAuth`,
and SDK credential injection is disabled. Generic Modal scale/timeout/proxy-auth
opt-out, and arbitrary provider pass-through are not Lightning settings.
See [Lightning](../how-to/use-lightning.md) for machine mappings, separate installation,
CLI deletion and initial-readiness timeout configuration.

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

Pooled records bind control credentials by a nonsecret digest. Changing keys affects new
deployments; managing existing records requires the original generation. Modal proxy tokens
rotate independently. CredWeave cooldown/health/concurrency state is in-memory by default.
SkyPilot account isolation uses its public 0.13 plugin API (`>=0.13.0,<0.14`) and three private
ports per account generation. See [recovery and rotation](../how-to/use-multiple-accounts.md).
