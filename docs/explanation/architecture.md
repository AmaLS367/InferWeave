# Architecture

InferWeave separates metadata, orchestration, provisioning and inference.

```text
Application -> InferWeave -> ModelRegistry / ProviderRouter / runtime templates
                       -> ProvisioningService -> AccountManager (CredWeave pools) + providers
                       -> LifecycleService -> DeploymentRepositoryPort -> SQLite
                       -> HealthcheckService -> authenticated HTTP probe
Deployment             -> InferenceClient -> FishSpeechClient / ImageGenerationClient
                                         -> InferenceTransport -> authenticated HTTP
ProviderRouter         -> ModalProvider | LightningProvider | SkyPilotProvider per registered cloud
LightningProvider      -> lightning_worker.py (one process per account operation)
SkyPilotProvider       -> skypilot_worker.py -> per-account SkyPilot home + API server
```

## Accounts and ownership

`AccountManager` owns one CredWeave `CredentialPool` per provider with configured accounts.
Credentials come from environment variables (each account read independently, so an unset
account drops out instead of breaking the pool), hot-reloaded CredWeave JSON files or custom
CredWeave sources, and are validated against the provider's credential schema. A
`ScopedStrategy` wraps the configured CredWeave strategy so one acquire can exclude accounts
already tried or pin one account.

`ProvisioningService` runs each deploy attempt: lease, write-ahead record (owner, remote
resource name, owner digest, `creation_may_continue`, `needs_reconciliation`), provider create call, failure classification
(`FailureKind`), reconciliation by name under the same account when the resource may exist,
CredWeave outcome report, then bounded failover. Providers are stateless adapters that receive
the record and the `ProviderAccount` for every call; they never persist anything or choose
credentials. `LifecycleService` resolves the owning account from the record for status, stop,
autostop and `reconcile()`, and never substitutes another account. Records persist only
nonsecret ownership data; secret values exist in process memory, worker stdin and the
providers' own credential stores.

## Registry, routing and runtimes

`ModelRegistry` holds IDs, external artifacts, workload, hardware, runtime and
health policy. Friendly and artifact IDs differ for Fish S2/WAN. Register custom
profiles again after restart. `ProviderRouter.aresolve()` handles explicit
providers or `SmartRoutingService` for auto/cheapest/free-first selection.
Catalog/hardware services validate GPU/VRAM. Offers are hints, not capacity or
billing guarantees. Use async resolution inside event loops.

Templates render `RuntimeSpec`: pinned image, environment, dependencies, port
and literal argv. Built-ins: vLLM, Fish Speech v1/S2, FLUX diffusers, WAN.
`runtimes/manifest.py` records digests/dependencies. Templates do not imply
registered models. Old plans mentioning SGLang/ComfyUI are not implemented templates.

## Providers and recovery

Modal creates a named app/runtime via protected `web_server`. Its scaledown
window controls containers; stop controls the app. Persistent records allow
provider status/stop recovery. SkyPilot manages registered cloud launch,
stop/down and normalized status. SDKs load lazily; SkyPilot provisioning on
native Windows needs WSL2/POSIX, while Modal supports native Windows.
Base installs import APIs without provider SDKs. No local-Docker or Hugging Face
provisioning adapter is implemented.

`LifecycleService` defaults to SQLite/WAL; JSON/in-memory adapters are injectable.
Records store identity, URL, options/state, timestamps, workload/activity with
sensitive configuration redacted. Old optional fields load. `attach()` rebuilds
callbacks without deployment; `find()` attaches a unique stored match and
rejects ambiguity. Neither automatically checks live provider truth.

## Auth, health, lifecycle and inference

`LightningProvider` translates the shared `RuntimeSpec` into one public-image
container Deployment, with sequential setup, foreground server and bundled worker
sources. It creates no Studio/snapshot. The SDK authenticates once per process, so
every operation runs in a worker process holding only the owning account's key;
cancellation drains the worker before cleanup. Full deletion uses the official CLI
shipped in that SDK (invoked inside the worker) because `Deployment.delete()` calls the
model HTTP endpoint. Deletion is confirmed independently. No private Lightning APIs are
used. The record's `resource` holds only name, teamspace, resource ID and ownership.
`include_credentials=False` prevents model injection. See [Lightning](../how-to/use-lightning.md).

`EndpointAuthPort` resolves runtime headers from the deployment's owning account, not
persisted secrets. Resolvers cover Modal Proxy Tokens, Lightning user-key Bearer auth,
static headers, composites/no auth. Readiness,
explicit/status probes and recovered inference share them. Redirects never
forward auth across origins.

Health polling uses model readiness; infrastructure alone cannot establish
HEALTHY. Successful probes reset local activity. Readiness timeout marks FAILED;
optional cleanup stops compute. Watchdogs monitor only registered/attached handles
in their own process. In-flight inference prevents local idle stop.
See [lifecycle and cost](lifecycle-and-cost.md).

Transport owns pools unless injected, resolves auth per attempt, follows bounded
same-origin redirects and applies phase timeouts/backoff with typed/redacted errors.
Fish Speech sends MessagePack `/v1/tts` for raw audio; FLUX sends JSON
`/v1/images/generations` and decodes base64. `InferenceClient` validates workloads
and tracks activity. Handles expose `synthesize()`/`render()`; LLM/WAN lack
unified high-level inference methods.

## Secret scanner scope

GitGuardian's Authentication Tuple detector mistakes auth class/module identifiers
in the lazy export map for credentials. The existing `__init__.py` path exception
is retained; auth implementations and all other files remain scanned. Never store
credentials in this entrypoint.

[ggshield configuration](https://docs.gitguardian.com/ggshield-docs/configuration)
supports path exclusions and exact occurrence fingerprints, but no line-scoped
rule. A smaller declaration-only module with a local path exclusion was tested;
the GitHub app still reported all six identifier pairs. Local ggshield exclusions
[are not shared with the dashboard](https://docs.gitguardian.com/ggshield-docs/reference/secret/ignore).
A reliable narrower rule therefore needs scanner-generated fingerprints for CLI
use and separate dashboard incident configuration for the GitHub app. Neither is
available in this release workflow. Preserve the original export map and exception
rather than disabling the Authentication Tuple detector or obscuring identifiers.

Recovery uses per-deployment process locks in built-in repositories. Remote creation that may
still run requires explicit provider-history confirmation before absence/cleanup can settle
the record. Credential generations are bound conservatively because a configured account name
does not prove cloud identity. SkyPilot accounts have private API, metrics and request queues
through its public 0.13 plugin API. CredWeave pool health/concurrency state remains in-memory
by default; durable deployment ownership does not imply distributed account scheduling.
