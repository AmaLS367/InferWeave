"""Configure Lightning user credentials and teamspace, then run this paid example."""

import asyncio
from pathlib import Path

from inferweave import InferWeave


async def main() -> None:
    weave = InferWeave()
    previous_ids = {record.id for record in await weave.list_records()}
    try:
        deployment = await weave.deploy("fish-s2-pro", provider="lightning", gpu_type="L4")
        Path("speech.wav").write_bytes(await deployment.synthesize("Hello from Lightning AI"))
    finally:
        # Also covers readiness failure before deploy() returns a handle.
        try:
            for record in await weave.list_records():
                if record.id not in previous_ids and record.state.value != "stopped":
                    await weave.stop(record.id)
        finally:
            await weave.close()


if __name__ == "__main__":
    asyncio.run(main())
