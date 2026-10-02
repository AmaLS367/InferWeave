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
