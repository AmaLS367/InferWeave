# Generate an image

Install `inferweave[modal]` and configure both credential pairs as in the
[TTS tutorial](../tutorials/first-tts-on-modal.md). The registered image model is
`black-forest-labs/FLUX.1-schnell`, served by InferWeave's FLUX diffusers worker.
Save and run the program below, or run [examples/image.py](../../examples/image.py).

<!-- example: image -->
```python
"""Run with python examples/image.py after configuring Modal and proxy credentials."""

import asyncio
from pathlib import Path

from inferweave import InferWeave


async def main() -> None:
    weave = InferWeave()
    deployment = None
    try:
        deployment = await weave.deploy(
            model="black-forest-labs/FLUX.1-schnell",
            provider="modal",
            custom_args={"cleanup_on_failure": True},
        )
        images = await deployment.render(
            "A small red fox in a snowy forest",
            width=512,
            height=768,
            steps=4,
            seed=42,
            n=2,
        )
        for index, image in enumerate(images):
            Path(f"image-{index}.png").write_bytes(image)
    finally:
        try:
            if deployment is not None:
                await deployment.stop()
        finally:
            await weave.close()


if __name__ == "__main__":
    asyncio.run(main())
```


`render()` returns `list[bytes]`, one PNG per image. Width/height map to
`size="512x768"`; `steps` maps to `num_inference_steps`. `seed` and `n` are
forwarded, and `b64_json` is decoded for you. Unset tuning fields defer to the
worker (Schnell defaults: 4 steps and 0.0 guidance). Choose dimensions and
batches within your GPU budget.

Invalid JSON/base64, empty data and an unexpected image count raise
`InvalidInferenceResponseError`. Retry/timeout semantics match audio inference.
WAN/video can be deployed but has no unified video client. `render()` is
image-only; calling it on WAN raises `UnsupportedWorkloadError`.
See [supported models](../reference/supported-models.md).
