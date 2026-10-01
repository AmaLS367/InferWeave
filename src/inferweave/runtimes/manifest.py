"""Pinned container images and exact dependency pins for built-in runtimes.

Reproducibility policy for built-in InferWeave runtimes: the same InferWeave release must
deploy the same runtime software later. Therefore:

* Container images use explicit version tags (never ``:latest``).
* Every direct Python dependency is pinned to an exact ``package==x.y.z`` version.
* The transitive dependency closure of the diffusion/video worker stack is pinned as well
  (``DIFFUSION_TRANSITIVE_PINS``) so a later ``pip install`` cannot drift to newer
  numpy / huggingface-hub / pydantic / ... releases.

Trade-off: exact pins mean security and bug-fix updates to these packages arrive only with a
new InferWeave release (re-resolve and re-test the lock, see below) instead of automatically.
Custom user-defined runtimes remain free to specify their own container images, mutable tags,
or custom setup scripts.

Regenerating the diffusion lock (linux, Python 3.11 = the interpreter in ``PYTORCH_IMAGE``).
Put the direct pins from this file plus ``torch==2.4.0`` and ``numpy==1.26.4`` into ``req.in``
and run::

    uv pip compile req.in --python-version 3.11 --python-platform linux \\
        --no-annotate --no-header

Drop torch itself and its exclusive dependencies (triton, nvidia-*, sympy, mpmath, networkx,
jinja2, markupsafe): they are supplied by ``PYTORCH_IMAGE``.
"""

# Base container images (explicit versioned tags, never :latest)
VLLM_IMAGE = "vllm/vllm-openai:v0.7.3"
# Legacy Fish Speech v1.x runtime (pre-S2 models). Not compatible with S2 Pro.
FISH_SPEECH_IMAGE = "fishaudio/fish-speech:v1.5.1"
# Official Fish Speech S2 server image (CUDA 12.6, Python 3.12, uv env in /app/.venv, checkpoints
# expected under /app/checkpoints). Built from upstream tag v2.0.0-beta, the first release with
# S2 Pro support (amd64 index digest sha256:882a5541959a3dc2ac7e72fee04a8c4e740cb282f2bdb6c282346b021b99bbf0).
FISH_SPEECH_S2_IMAGE = "fishaudio/fish-speech:server-cuda-v2.0.0-beta"
PYTORCH_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

# Fish Audio S2 Pro weights: Hugging Face repo and the immutable commit that matches
# FISH_SPEECH_S2_IMAGE (checkpoint layout: sharded safetensors + codec.pth).
FISH_S2_PRO_ARTIFACT = "fishaudio/s2-pro"
FISH_S2_PRO_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"

# Exact direct dependencies for diffusion and worker runtimes.
# diffusers>=0.33 is required for WanPipeline (Wan 2.1 video); FluxPipeline is also in 0.33.
DIFFUSERS_SPEC = "diffusers==0.33.1"
TRANSFORMERS_SPEC = "transformers==4.51.3"
ACCELERATE_SPEC = "accelerate==1.6.0"
SENTENCEPIECE_SPEC = "sentencepiece==0.2.0"
PROTOBUF_SPEC = "protobuf==5.29.5"
FASTAPI_SPEC = "fastapi==0.115.12"
UVICORN_SPEC = "uvicorn==0.34.2"

# Video worker dependencies
IMAGEIO_SPEC = "imageio==2.37.0"
IMAGEIO_FFMPEG_SPEC = "imageio-ffmpeg==0.6.0"

# Exact pins for the transitive closure of the specs above (linux, Python 3.11), excluding
# torch and its exclusive dependencies which come from PYTORCH_IMAGE.
# numpy is held on 1.x on purpose: torch 2.4.0 wheels are built against the NumPy 1 ABI and
# fail to initialize NumPy under numpy 2.x ("_ARRAY_API not found").
DIFFUSION_TRANSITIVE_PINS: tuple[str, ...] = (
    "annotated-types==0.8.0",
    "anyio==4.15.1",
    "certifi==2026.7.22",
    "charset-normalizer==3.5.2",
    "click==8.5.0",
    "filelock==4.0.8",
    "fsspec==2026.9.0",
    "h11==0.16.0",
    "hf-xet==1.6.0",
    "huggingface-hub==0.36.2",
    "idna==3.20",
    "importlib-metadata==9.0.1",
    "numpy==1.26.4",
    "packaging==26.3",
    "pillow==12.3.0",
    "psutil==7.2.2",
    "pydantic==2.13.5",
    "pydantic-core==2.46.5",
    "pyyaml==6.0.3",
    "regex==2026.9.29",
    "requests==2.34.2",
    "safetensors==0.8.0",
    "starlette==0.46.2",
    "tokenizers==0.21.4",
    "tqdm==4.70.1",
    "typing-extensions==4.16.0",
    "typing-inspection==0.4.4",
    "urllib3==2.8.0",
    "zipp==4.1.0",
)
