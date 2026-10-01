<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/inferweave_banner.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/inferweave_banner_light.png">
    <img alt="InferWeave Logo" src="assets/inferweave_banner.png" width="580">
  </picture>
</p>

<p align="center">
  <em>Unified AI inference deployment SDK across cloud GPU providers</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

> **Unified AI inference deployment SDK across cloud GPU providers.**  
> Choose your model or workload (Audio, LLM, Image, Video) — InferWeave provisions the GPU infrastructure, sets up the container environment, manages CUDA/VRAM requirements, and exposes a ready-to-use endpoint.

---

## 🌟 The Vision

Deploying open-weight AI models into production is notoriously complex:
- Different hardware specs (VRAM sizing, tensor parallelism, compute capability).
- Fragmented container runtimes (vLLM, SGLang, ComfyUI, custom FastAPI/PyTorch wrappers).
- Provider lock-in and disparate provisioning APIs (RunPod, Lambda, AWS, GCP, Modal, etc.).

**InferWeave hides all infrastructural complexity behind a unified Python interface:**

```python
import asyncio
from inferweave import InferWeave


async def main():
    weave = InferWeave()

    # Deploy an audio voice model directly to RunPod
    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="runpod",
    )

    print(f"Endpoint live at: {deployment.endpoint_url}")


asyncio.run(main())
```

InferWeave automatically handles model profile detection, VRAM budgeting, container runtime templates, healthcheck polling, and endpoint proxying.

---

## 🏗️ Architecture

```text
               ┌────────────────────────┐
               │    Your Application    │
               └───────────┬────────────┘
                           │
               ┌───────────▼────────────┐
               │     InferWeave SDK     │
               │ (Registry, Router, API)│
               └───────────┬────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│ Runtime Layer │                       │ Compute Layer │
├───────────────┤                       ├───────────────┤
│ • Audio (TTS) │                       │ • SkyPilot:   │
│ • LLM (vLLM)  │                       │   RunPod/Vast │
│ • Image (FLUX)│                       │   AWS/GCP/etc.│
│ • Video (WAN) │                       │ • Modal       │
│ • Custom      │                       │ • Local GPU   │
└───────────────┘                       └───────────────┘
```

---

## 📦 Installation

InferWeave provides a modular installation model so you only install dependencies for the clouds you actually use.

### 1. Base Installation (Core SDK & CLI)
Installs the unified client, schema definitions, model registry, and CLI tools:
```bash
pip install inferweave
```

### 2. Provider-Specific Extras
Install drivers for the clouds you plan to target:
```bash
# Individual cloud providers (via SkyPilot engine)
pip install "inferweave[runpod]"
pip install "inferweave[aws]"
pip install "inferweave[gcp]"
pip install "inferweave[azure]"
pip install "inferweave[lambda]"
pip install "inferweave[nebius]"
pip install "inferweave[kubernetes]"
pip install "inferweave[vast]"

# Serverless compute engine (verified with modal>=1.6,<1.7)
pip install "inferweave[modal]"

# Inference worker runtime (FastAPI / Uvicorn stack)
pip install "inferweave[workers]"
```

### 3. Multi-Cloud Bundles
```bash
# All cloud providers supported by SkyPilot
pip install "inferweave[clouds]"

# Complete suite (all cloud engines + Modal + workers runtime)
pip install "inferweave[all]"
```

---

## ⚡ CLI Quickstart

InferWeave includes a high-performance command-line interface for deploying and inspecting models across GPU providers:

```bash
# 1. Deploy models directly to cloud GPUs
inferweave deploy fish-s2-pro --provider modal
inferweave deploy meta-llama/Meta-Llama-3-8B-Instruct --provider runpod --gpu A100

# 2. Dry-run deployment to preview configuration and hardware budgets
inferweave deploy fish-s2-pro --provider modal --dry-run

# 3. Explore supported models and cloud providers
inferweave models
inferweave models --workload audio
inferweave providers

# 4. Inspect, monitor, and stop deployments across processes
inferweave list
inferweave status <deployment-id>
inferweave stop <deployment-id>
```

---

## 💻 Platform & OS Support Guidelines

| Operating System | InferWeave Core / Client | Modal Engine | SkyPilot Engine (Multi-Cloud) |
| :--- | :---: | :---: | :---: |
| **Linux (Ubuntu/Debian/RHEL)** | ✅ Native | ✅ Native | ✅ Native |
| **macOS (Apple Silicon / Intel)** | ✅ Native | ✅ Native | ✅ Native |
| **Windows (WSL2)** | ✅ Native (WSL) | ✅ Native (WSL) | ✅ Native (WSL) |
| **Windows (Native PowerShell/CMD)** | ✅ Native | ✅ Native | ⚠️ Requires WSL2 / Remote Runner |

### ⚠️ Important Note for Windows Users
SkyPilot’s underlying execution engine relies on POSIX system primitives (`termios`, `resource`) that are not present in native Windows. 

- **If you are on Windows:** We recommend running InferWeave inside **WSL2** (Windows Subsystem for Linux), where all multi-cloud provisioning features work seamlessly.
- **InferWeave Graceful Fallback:** InferWeave utilizes lazy loading for SkyPilot modules. If invoked directly on native Windows, InferWeave will safely guide you to WSL2 or allow targeting native-supported backends (such as Modal or remote runners) without crashing during import.

---

## 🛣️ Roadmap

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
- [ ] **Unified Client Protocol:** Standardized `.generate()`, `.synthesize()`, and `.render()` methods.
- [x] **Lifecycle Management:** Auto-shutdown on idle, healthcheck polling, and persistent cross-process deployment state repository (`~/.inferweave/deployments.db`, SQLite with WAL mode, configurable via `INFERWEAVE_DEPLOYMENTS_PATH`).

---

## 🛠️ Contributing & Development Setup

InferWeave uses [`uv`](https://docs.astral.sh/uv/) for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/AmaLS367/InferWeave.git
cd InferWeave

# Setup virtual environment with all extras and dev dependencies
uv sync --all-extras --group dev

# Run linting and type checking
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration"
```

#### Live integration tests (opt-in, spend real money)

Tests marked `integration` talk to real clouds. They are excluded from CI and skip
cleanly unless explicitly enabled and configured:

| Test | Required environment |
| --- | --- |
| `tests/test_modal_integration.py` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| `tests/test_runpod_integration.py` | `INFERWEAVE_RUNPOD_INTEGRATION=1`, `RUNPOD_API_KEY` (or `~/.runpod/config.toml`), `inferweave[runpod]` on Linux/macOS/WSL2 |

The RunPod test also reads optional `INFERWEAVE_RUNPOD_GPU` (default `L4`),
`INFERWEAVE_RUNPOD_MODEL` (default `facebook/opt-125m`), `INFERWEAVE_RUNPOD_READY_TIMEOUT_SECS`,
`INFERWEAVE_RUNPOD_AUTOSTOP_MINS` and `INFERWEAVE_RUNPOD_ALLOW_SPOT`; see the module docstring.

```bash
INFERWEAVE_RUNPOD_INTEGRATION=1 RUNPOD_API_KEY=... \
  uv run --extra runpod pytest -m integration tests/test_runpod_integration.py -s
```

### 🚀 Release Process

Publishing to PyPI is automated via GitHub Actions using PyPI Trusted Publishing (OIDC):

1. Ensure the CI suite is green on `master`.
2. Set the package version in `src/inferweave/__init__.py`.
3. Create and push a Git tag: `vX.Y.Z`.
4. Publish a GitHub Release from that tag.
5. GitHub Actions (`publish.yml`) verifies artifact metadata, validates that the release tag matches `inferweave.__version__`, and publishes the artifacts to PyPI via Trusted Publishing.

---

## 📄 License

InferWeave's source code is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Third-party licenses

The Apache-2.0 license covers InferWeave's own source code only. It does **not** cover anything InferWeave deploys or references:

- **Model weights** (including those named by the built-in model profiles) keep their upstream licenses.
- **Container images and third-party runtimes** (vLLM, Fish Speech, PyTorch, and the Python packages installed into them) keep their upstream licenses.
- You are responsible for reviewing and complying with the upstream terms of every model, image and runtime you deploy. This is not legal advice.

**Fish Audio S2 Pro (`fish-s2-pro`):** the `fishaudio/s2-pro` weights are published under Fish Audio's own license (`fish-audio-research-license` on [Hugging Face](https://huggingface.co/fishaudio/s2-pro)), not Apache-2.0. The built-in S2 runtime is **beta**: it uses the upstream `v2.0.0-beta` pre-release of [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). Review the upstream model and runtime licenses before any production or commercial use.
