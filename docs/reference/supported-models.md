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
| `Qwen/Qwen2.5-0.5B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `Qwen/Qwen2.5-1.5B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `Qwen/Qwen2.5-7B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `mistralai/Mistral-7B-Instruct-v0.3` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `microsoft/Phi-3.5-mini-instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required |
| `meta-llama/Llama-3.2-1B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required; ungated `unsloth` re-upload |
| `meta-llama/Llama-3.2-3B-Instruct` | llm | `vllm` | vLLM OpenAI-compatible HTTP | No unified method | Deployment only; direct endpoint client required; ungated `unsloth` re-upload |
| `black-forest-labs/FLUX.1-schnell` | image | `flux-diffusers` | `/v1/images/generations` JSON / base64 PNG | `render()` | InferWeave FLUX worker |
| `wan-video/wan-2.1` | video | `wan-video` | `/v1/videos/generations` JSON / base64 video | No unified method | Deployment only; direct endpoint client required |

## Artifacts and readiness

| ID | External artifact | Minimum VRAM | Health endpoint |
| --- | --- | --- | --- |
| `fish-s2-pro` | `fishaudio/s2-pro` | 24 GB | `/v1/health` (port 8080) |
| `meta-llama/Meta-Llama-3-8B-Instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` | 16 GB | `/health` (port 8000) |
| `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | 6 GB | `/health` (port 8000) |
| `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 8 GB | `/health` (port 8000) |
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 20 GB | `/health` (port 8000) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` | 20 GB | `/health` (port 8000) |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | 20 GB | `/health` (port 8000) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 8 GB | `/health` (port 8000) |
| `microsoft/Phi-3.5-mini-instruct` | `microsoft/Phi-3.5-mini-instruct` | 16 GB | `/health` (port 8000) |
| `meta-llama/Llama-3.2-1B-Instruct` | `unsloth/Llama-3.2-1B-Instruct` | 8 GB | `/health` (port 8000) |
| `meta-llama/Llama-3.2-3B-Instruct` | `unsloth/Llama-3.2-3B-Instruct` | 12 GB | `/health` (port 8000) |
| `black-forest-labs/FLUX.1-schnell` | `black-forest-labs/FLUX.1-schnell` | 24 GB | `/health` (port 8000) |
| `wan-video/wan-2.1` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 24 GB | `/health` (port 8000) |

## Aliases and lookup

`weave.deploy(model=...)` and `registry.get(...)` accept the exact id, a registered alias, any
letter case, or the repository-less name when it is unique (`Qwen2.5-1.5B-Instruct`). Deployments
are recorded under the canonical id. Unknown names raise `ModelNotFoundError`, which lists the
registered models and close matches.

| Alias | Model |
| --- | --- |
| `flux` | `black-forest-labs/FLUX.1-schnell` |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `phi-3.5` | `microsoft/Phi-3.5-mini-instruct` |
| `phi-3.5-mini` | `microsoft/Phi-3.5-mini-instruct` |
| `qwen2.5` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen2.5-coder` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `qwen2.5-coder-7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `smollm2` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `wan` | `wan-video/wan-2.1` |
| `wan2.1` | `wan-video/wan-2.1` |

The Llama 3.2 profiles keep the `meta-llama/...` id but download the ungated `unsloth/...`
re-upload, because the official repositories are gated and InferWeave does not pass a Hugging Face
token. The Llama 3.2 Community License applies. Long-context profiles (Llama 3.2, Phi-3.5) cap
`--max-model-len` at 16384 through `metadata["vllm_engine_args"]` so the KV cache fits a 24 GB GPU;
pass `engine_args={"max_model_len": ...}` in the deployment options to override it. The built-in
vLLM image is v0.7.3, so newer architectures (for example Qwen3) need a custom container or an
updated runtime pin. The new LLM profiles have mocked/unit coverage only; no live GPU deployment
has been run for them.

Registry facts are validated against source by the example smoke harness. GPU recommendations do not guarantee capacity.
Audio supports WAV/PCM/MP3; images return decoded PNG bytes. LLM/video are deployable but lack unified inference.
Fish Speech S2 uses pinned upstream beta runtime/weights. Review upstream licenses as described in the [README](../../README.md#third-party-licenses).
