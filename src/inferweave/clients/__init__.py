"""Inference clients for calling deployed models (audio and image workloads)."""

from inferweave.clients.audio import FishSpeechClient, ReferenceAudio
from inferweave.clients.image import ImageGenerationClient
from inferweave.clients.inference import InferenceClient
from inferweave.clients.transport import InferenceConfig, InferenceTransport

__all__ = [
    "FishSpeechClient",
    "ImageGenerationClient",
    "InferenceClient",
    "InferenceConfig",
    "InferenceTransport",
    "ReferenceAudio",
]
