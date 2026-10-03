# Roadmap

- [x] **Core Model Registry:** Pre-configured specs for popular Audio, LLM, Image, and Video models with separation of InferWeave IDs and external model repository artifacts.
- [x] **Runtime Templates:**
  - Built-in runtime images are pinned by immutable `@sha256` digest (the upstream tag each digest was resolved from is recorded in `inferweave/runtimes/manifest.py`)
  - LLM: `vLLM` (`vllm/vllm-openai:v0.7.3`)
  - Audio: `fish-s2-pro` on `fish-speech-s2` (**beta** — upstream `fishaudio/fish-speech:server-cuda-v2.0.0-beta`, a GitHub pre-release; weights pinned to an exact `fishaudio/s2-pro` revision); legacy Fish Speech v1.x models stay on `fish-speech` (`fishaudio/fish-speech:v1.5.1`)
  - Image: `FLUX` via unified worker (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`, exact-pinned diffusers stack incl. transitive dependencies)
  - Video: `WAN` 2.1 via unified worker (same PyTorch base image, exact-pinned diffusers/video stack incl. transitive dependencies)
  - Custom user-defined runtimes may specify arbitrary container images (including mutable tags) and dependencies
- [x] **Smart Compute Routing:**
  - `provider="auto"` with `strategy="cheapest"`
  - `strategy="free_first"` (spot instances / community compute)
  - Latency and VRAM-aware GPU matching and validation.
- [x] **Unified Client Protocol:** Audio `.synthesize()` and image `.render()` with typed errors, authentication, retries and activity tracking. Unified LLM/video inference remains future work.
- [x] **Lifecycle Management:** Auto-shutdown on idle, healthcheck polling, and persistent cross-process deployment state repository (`~/.inferweave/deployments.db`, SQLite with WAL mode, configurable via `INFERWEAVE_DEPLOYMENTS_PATH`).
