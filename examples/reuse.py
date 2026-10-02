"""Reuse persisted state. Serialize startup externally when running multiple replicas."""

import asyncio
from pathlib import Path

from inferweave import AmbiguousDeploymentError, InferWeave


async def main() -> None:
    weave = InferWeave()
    try:
        try:
            deployment = await weave.find(model="fish-s2-pro", provider="modal")
        except AmbiguousDeploymentError as error:
            raise RuntimeError(
                f"Choose a deployment ID: {error.candidate_ids}"
            ) from error
        if deployment is None:
            deployment = await weave.deploy(
                model="fish-s2-pro",
                provider="modal",
                scaledown_window_seconds=300,
                destroy_after_idle_mins=1440,
                custom_args={"cleanup_on_failure": True},
            )
        deployment_id = deployment.id  # Store this in your application configuration.
    finally:
        await weave.close()  # Keeps the remote app deployed; stops the local watchdog.

    restarted = InferWeave()  # Uses the same SQLite database.
    try:
        attached = await restarted.attach(deployment_id)
        audio = await attached.synthesize("Recovered deployment")
        Path("recovered.wav").write_bytes(audio)
        # This demonstration is finished. A service would retain and reuse the handle.
        await attached.stop()
    finally:
        await restarted.close()


if __name__ == "__main__":
    asyncio.run(main())
