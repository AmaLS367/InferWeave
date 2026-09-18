# InferWeave Architecture & Design Principles

This document describes the architectural layout, core subsystems, and design constraints of **InferWeave**.

---

## 1. High-Level Architecture

InferWeave acts as an intelligent abstraction layer between end-user applications and heterogenous GPU compute providers:

```
[ Application Layer ]
        │  (User specifies: model, workload type, optional provider / strategy)
        ▼
[ InferWeave Core Orchestrator ]
   ├── Model Registry          (Tracks VRAM, CUDA requirements, default runtimes)
   ├── Routing & Sizing Engine (Resolves minimum GPU requirements, selects provider)
   ├── Deployment Manager      (Tracks deployment state, health, lifecycle)
   └── Unified Inference Client (Presents typed APIs: LLM, Audio, Image, Video)
        │
        ├── [ Runtime Templates Engine ]
        │      Generates startup scripts, environment configs, and container definitions
        │      (vLLM, SGLang, ComfyUI, Fish-Speech, etc.)
        │
        ▼
[ Compute Adapters Layer ]
   ├── SkyPilot Backend Adapter (Multi-cloud IaaS: RunPod, Vast, AWS, GCP, Lambda, Nebius...)
   ├── Modal Backend Adapter    (Serverless GPU functions & containers)
   ├── Hugging Face Adapter     (Dedicated Inference Endpoints / Spaces)
   └── Local Docker Adapter     (Local NVIDIA GPUs for dev & testing)
```

---

## 2. Core Subsystems

### 2.1. Model Registry (`inferweave.registry`)
The registry maps high-level model identifiers (e.g. `"fish-s2-pro"`, `"meta-llama/Meta-Llama-3-8B-Instruct"`, `"black-forest-labs/FLUX.1-schnell"`) to physical runtime requirements:
- **Workload Category**: `AUDIO`, `LLM`, `IMAGE`, `VIDEO`.
- **Hardware Profile**:
  - Minimum VRAM (e.g., 16 GB, 24 GB, 80 GB).
  - Recommended GPU families (e.g., `A10G`, `L4`, `A100`, `H100`).
  - Supported compute capabilities (e.g., sm_80, sm_89, sm_90).
- **Default Runtime Template**: Associated default container runtime (e.g., `vllm`, `sglang`, `comfyui`, `fish-speech`).
- **Healthcheck Specification**: Expected endpoint path (e.g., `/health`, `/v1/models`) and timeout bounds.

### 2.2. Runtime Templates (`inferweave.runtimes`)
Instead of forcing users to write Dockerfiles or shell scripts:
- Templates know the exact launch parameters, environment variables, and ports for each model category.
- Supports customizable runtime parameters (e.g. `--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`).
- Standardizes container outputs to expose HTTP / WebSocket endpoints.

### 2.3. Compute Layer Adapters (`inferweave.compute`)
All compute providers implement a common asynchronous interface:

```python
from abc import ABC, abstractmethod
from typing import Optional
from inferweave.models import DeploymentRequest, DeploymentStatus

class ComputeProvider(ABC):
    @abstractmethod
    async def deploy(self, request: DeploymentRequest) -> DeploymentStatus:
        """Provisions compute, starts the workload, and waits for health readiness."""
        pass

    @abstractmethod
    async def stop(self, deployment_id: str) -> None:
        """Terminates or pauses the deployment."""
        pass

    @abstractmethod
    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Returns the current status, endpoint URL, and resource utilization."""
        pass
```

---

## 3. Platform & Cross-Compatibility Considerations

### 3.1. SkyPilot POSIX Dependency on Windows
- **Issue**: SkyPilot relies on Unix system calls (`import resource`, `import termios`) for process isolation and subshell execution.
- **Architectural Solution in InferWeave**:
  1. **Lazy Loading**: `import sky` is strictly forbidden in the top-level package namespace (`inferweave.__init__`). Compute backends are loaded dynamically on demand.
  2. **Clear Diagnostics**: When a user attempts to trigger a SkyPilot-backed provider (`runpod`, `aws`, etc.) directly on native Windows, InferWeave captures the platform mismatch and suggests:
     - Running the workload under **WSL2** (supported natively).
     - Or using providers that support native Windows clients (e.g., Modal or remote API proxies).

### 3.2. Extensible Extra Packaging
To minimize installation footprint:
- The base package `inferweave` contains zero cloud SDK bloat.
- Cloud drivers are grouped via PEP 508 optional dependencies:
  - `inferweave[runpod]`
  - `inferweave[aws]`
  - `inferweave[gcp]`
  - `inferweave[clouds]` (all SkyPilot-supported clouds)
  - `inferweave[modal]`
