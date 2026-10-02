"""Run with python examples/tts.py after configuring Modal and proxy credentials."""

import asyncio
from pathlib import Path

from inferweave import InferWeave


async def main() -> None:
    weave = InferWeave()
    deployment = None
    try:
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            custom_args={"cleanup_on_failure": True},
        )
        audio = await deployment.synthesize("Hello from InferWeave", format="wav")
        Path("speech.wav").write_bytes(audio)
    finally:
        try:
            if deployment is not None:
                await deployment.stop()
        finally:
            await weave.close()


if __name__ == "__main__":
    asyncio.run(main())
