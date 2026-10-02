# Architecture

InferWeave separates metadata, orchestration, provisioning and inference.

```text
Application -> InferWeave -> ModelRegistry / ProviderRouter / runtime templates
                       -> LifecycleService -> DeploymentRepositoryPort -> SQLite
                       -> HealthcheckService -> authenticated HTTP probe
Deployment             -> InferenceClient -> FishSpeechClient / ImageGenerationClient
                                         -> InferenceTransport -> authenticated HTTP
ProviderRouter         -> ModalProvider | SkyPilotProvider per registered cloud
```

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
Base installs import APIs without either SDK. No local-Docker or Hugging Face
provisioning adapter is implemented.

`LifecycleService` defaults to SQLite/WAL; JSON/in-memory adapters are injectable.
Records store identity, URL, options/state, timestamps, workload/activity with
sensitive configuration redacted. Old optional fields load. `attach()` rebuilds
callbacks without deployment; `find()` attaches a unique stored match and
rejects ambiguity. Neither automatically checks live provider truth.

## Auth, health, lifecycle and inference

`EndpointAuthPort` resolves runtime headers, not persisted secrets. Resolvers
cover Modal Proxy Tokens, static headers, composites/no auth. Readiness,
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

GitGuardian's Authentication Tuple detector flags auth class/module identifiers.
The path exception is limited to the six declarative pairs in `_auth_exports.py`;
the entrypoint and auth implementations remain scanned. Do not put credentials
in this map. [GitGuardian configuration](https://docs.gitguardian.com/ggshield-docs/configuration)
supports exact-match fingerprints and path exclusions; existing scanner fingerprints
are unavailable locally, so the small path exception is used rather than inventing
hashes or disabling a detector globally.
