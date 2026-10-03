# Supported models

Lightning AI deploys the same container runtimes with `provider="lightning"`.
Fish S2 Pro uses L4 (24 GB) in the opt-in live TTS/recovery test; T4 is too small.
FLUX shares the generic translation and existing `render()` transport with offline
coverage; real FLUX validation remains separate. Setup/weights repeat on cold replicas.
Machine support does not guarantee account entitlement or availability.
See [Lightning guide](../how-to/use-lightning.md) for mappings and limits.

Built-in registry (custom profiles may add models).

| InferWeave model | Workload | Runtime | Inference endpoint/protocol | Deployment method | Status/notes |
| --- | --- | --- | --- | --- | --- |
| `fish-s2-pro` | audio | `fish-speech-s2` | `/v1/tts` MessagePack / raw audio | `synthesize()` | Pinned upstream beta; review model license |
| `meta-llama/Meta-Llama-3-8B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `black-forest-labs/FLUX.1-schnell` | image | `flux-diffusers` | `/v1/images/generations` JSON / base64 PNG | `render()` | InferWeave FLUX worker |
| `wan-video/wan-2.1` | video | `wan-video` | `/v1/videos/generations` JSON / base64 video | No unified method | Deployment only; direct endpoint client required |

## Artifacts and readiness

| ID | External artifact | Minimum VRAM | Health endpoint |
| --- | --- | --- | --- |
| `fish-s2-pro` | `fishaudio/s2-pro` | 24 GB | `/v1/health` (port 8080) |
| `meta-llama/Meta-Llama-3-8B-Instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` | 16 GB | `/health` (port 8000) |
| `black-forest-labs/FLUX.1-schnell` | `black-forest-labs/FLUX.1-schnell` | 24 GB | `/health` (port 8000) |
| `wan-video/wan-2.1` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 24 GB | `/health` (port 8000) |

Registry facts are validated against source by the example smoke harness. GPU recommendations do not guarantee capacity.
Audio supports WAV/PCM/MP3; images return decoded PNG bytes. LLM/video are deployable but lack unified inference.
Fish Speech S2 uses pinned upstream beta runtime/weights. Review upstream licenses as described in the [README](../../README.md#third-party-licenses).
