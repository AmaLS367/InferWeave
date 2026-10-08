# Changelog

## 0.3.0

Multi-account provider credentials backed by CredWeave 0.1 (`credweave>=0.1.0,<0.2`).

- Configure any number of named accounts for Modal, Lightning AI, RunPod and Vast.ai with
  `AccountsConfig` / an accounts YAML or JSON file (`InferWeave(accounts=...)`,
  `INFERWEAVE_ACCOUNTS_FILE`, CLI `--accounts`). Secrets come from environment variables,
  hot-reloaded CredWeave JSON credentials files or custom CredWeave sources; configuration
  never contains secret values.
- New deployments lease an account through the provider's CredWeave strategy (round robin,
  weighted, failover, least used, LRU, random) and report outcomes: 401 revokes, 403 parks
  without health damage, 402/quota and 429 cool down using provider hints, GPU stock-outs and
  invalid requests never degrade an account, transient errors back off. Failover across
  accounts (and, for `provider="auto"`, across feasible providers) is bounded by
  `max_attempts`; `deploy(account=...)` pins one account.
- Deployments are bound to the account that created them. Records persist only the nonsecret
  account id, control-credential digest and remote resource identity; status, stop, autostop, attach, inference headers
  and reconciliation always use that account and raise `AccountUnavailableError` instead of
  falling back to another one. Changed control keys support new deployments; existing resources reject them unless the
  original keys are restored. Endpoint proxy tokens can rotate independently.
- Duplicate-free provisioning recovery: records are written before the create call; a failure
  that may have created a resource is reconciled (and the resource destroyed) under the same
  account before any retry, otherwise `ProvisioningUncertainError` stops all retries.
  Process locks protect active provisioning across restarts/concurrent recovery. Pending remote
  creates require provider-history confirmation via `confirmed_settled` / `--confirm-settled`
  before recovery; age or a resource-absence snapshot cannot authorize a retry.
- Credential isolation: Modal uses an explicit `modal.Client.from_credentials` per account;
  Lightning and SkyPilot operations run in worker processes that receive only their account's
  secrets on stdin, with private homes (and a private SkyPilot API server per RunPod/Vast
  account, including its request queue via the public SkyPilot 0.13 plugin API). The process environment is never modified. Workers can run under separate
  interpreters (`INFERWEAVE_LIGHTNING_PYTHON`, `INFERWEAVE_SKYPILOT_PYTHON`), so Lightning SDK
  and SkyPilot no longer need to share an environment.
- Platform credentials (any configured secret or variables such as `MODAL_TOKEN_ID`) are
  refused in model runtimes and persisted options; SDK error text is withheld or redacted.
- New CLI commands `inferweave accounts` and `inferweave reconcile`; `list`/`status` show the
  owning account.

Breaking changes:

- Provider adapters implement a new stateless port (`provision`, `status`, `resource_exists`,
  `stop` taking the record and its `ProviderAccount`); `deploy()`/`get_status(id)` on
  providers, `bind_repository()` and provider-side persistence were removed.
- `DeploymentRecord.lightning` (`LightningDeploymentMetadata`) was replaced by the generic
  `DeploymentRecord.resource` (`ResourceRef`); records gained `account` and
  `owner_fingerprint`, `creation_may_continue` and `needs_reconciliation`. Records written by 0.2 load as owned by the `ambient` account.
- `EndpointAuthPort.headers_for()` / `is_configured_for()` take an optional `account`.
- `LifecycleService` requires a persisted record to stop or refresh a live deployment.

## 0.2.0

- Add optional Lightning AI container Deployment provider, shared TTS/image inference,
  user-key Bearer auth, typed autoscaling/resource metadata, recovery and verified
  full deletion/failure cleanup without Studios. Add opt-in L4 live testing.
  SDK 2026.10.1 requires a separate environment from SkyPilot because of upstream
  Click bounds; `all` retains its previous composition.

- Recover persisted deployments with `attach()` and unique-match `find()`.
- Synthesize Fish Speech audio/render FLUX images with typed errors, references,
  per-call options, timeouts and bounded cold-start retries.
- Protect Modal endpoints by default with separate Proxy Tokens for probes/inference;
  credentials resolve at runtime rather than being persisted by endpoint auth.
- Separate container scaling from full idle stop, persist inference activity
  and protect local in-flight calls from idle shutdown.
- Add tutorials, guides, references and credential-free runnable example validation.
  Strengthen live recovery and clean-wheel checks.
- Block cross-origin probe credential forwarding; sanitize repr/probe diagnostics
  and reject non-finite inference timeout/backoff settings.
- Close SQLite connections deterministically to prevent leaked file handles,
  particularly during Windows recovery/cleanup.
- Fix Modal image construction: mount local source after build steps, clear the
  inherited server entrypoint, and expose system Python in the Fish S2 image.

Compatibility: existing imports, provider options, old SQLite records and
`autostop_mins` remain supported. The alias defaults to 30 minutes but no longer
determines Modal's container window (independent default: 1800 seconds).
Protected Modal requires Proxy Tokens in addition to SDK credentials. Successful
probes reset local activity only; inference writes throttle to 30 seconds.
`close()` cancels local monitoring without destroying apps. No unified LLM/video client.

## 0.1.0

Initial registry, runtimes, cloud routing, readiness, persistent state, lifecycle and CLI.
