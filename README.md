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

# Serverless compute engine
pip install "inferweave[modal]"
```

### 3. Multi-Cloud Bundles
```bash
# All cloud providers supported by SkyPilot
pip install "inferweave[clouds]"

# Complete suite (all clouds + Modal + dev tools)
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

# 4. Check status or stop active deployments
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

- [ ] **Core Model Registry:** Pre-configured specs for popular Audio, LLM, Image, and Video models.
- [ ] **Runtime Templates:**
  - LLM: `vLLM`, `SGLang`, `llama.cpp`
  - Audio: `fish-speech` / `fish-s2-pro`, Whisper, Chatterbox
  - Image: `FLUX`, `ComfyUI` headless, `Diffusers`
  - Video: `WAN`, `HunyuanVideo`, `CogVideoX`
- [ ] **Smart Compute Routing:**
  - `provider="auto"` with `strategy="cheapest"`
  - `strategy="free_first"` (spot instances / community compute)
  - Latency and VRAM-aware GPU matching.
- [ ] **Unified Client Protocol:** Standardized `.generate()`, `.synthesize()`, and `.render()` methods.
- [ ] **Lifecycle Management:** Auto-shutdown on idle to prevent cloud overspending.

---

## 🛠️ Contributing & Development Setup

InferWeave uses [`uv`](https://docs.astral.sh/uv/) for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/inferweave.git
cd inferweave

# Setup virtual environment with all extras and dev dependencies
uv sync --all-extras --dev

# Run linting and type checking
uv run ruff check .
uv run mypy src
uv run pytest

```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
