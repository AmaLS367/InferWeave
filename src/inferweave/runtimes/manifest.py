"""Pinned container images and bounded dependency specifications for built-in runtimes.

These constants guarantee reproducible, deterministic deployments across cloud GPU providers.
Custom user-defined runtimes remain free to specify their own container images, mutable tags,
or custom setup scripts.
"""

# Base container images (explicit versioned tags, never :latest)
VLLM_IMAGE = "vllm/vllm-openai:v0.7.3"
FISH_SPEECH_IMAGE = "fishaudio/fish-speech:v1.5.1"
PYTORCH_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

# Tested, compatible package specifications for diffusion and worker runtimes
DIFFUSERS_SPEC = "diffusers>=0.31.0,<0.33.0"
TRANSFORMERS_SPEC = "transformers>=4.46.0,<4.49.0"
ACCELERATE_SPEC = "accelerate>=1.0.0,<1.3.0"
SENTENCEPIECE_SPEC = "sentencepiece>=0.2.0,<0.3.0"
PROTOBUF_SPEC = "protobuf>=5.28.0,<6.0.0"
FASTAPI_SPEC = "fastapi>=0.115.0,<0.116.0"
UVICORN_SPEC = "uvicorn>=0.32.0,<0.35.0"

# Video worker dependencies
IMAGEIO_SPEC = "imageio>=2.36.0,<2.37.0"
IMAGEIO_FFMPEG_SPEC = "imageio-ffmpeg>=0.5.0,<0.6.0"
