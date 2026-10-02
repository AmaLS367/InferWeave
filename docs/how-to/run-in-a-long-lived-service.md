# Run in a long-lived service

A bot, API server or background assistant should keep one SDK/handle per workload,
find persisted deployments before provisioning, and close local resources on shutdown.
Keep the same persistent SQLite directory and inject credentials on each restart.

1. Call `find(model=..., provider=...)`.
2. Deploy when absent. On ambiguity choose a configured ID and `attach()` it.
3. Retain the handle for `synthesize()`/`render()` on incoming requests.
4. Call `close()` on shutdown and `stop()` when intentionally retiring remote compute.

This generic runnable example demonstrates find-first, close and fresh SDK recovery:

<!-- example: reuse -->
```python
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
```


The demonstration stops the recovered app when finished. A service normally keeps
it deployed across shutdown: `close()` only closes local HTTP resources/watchdogs.

## Coordinate startup and shutdown

`find()` and `deploy()` are separate operations. Serialize startup through one
provisioning owner or an external lock across replicas to avoid duplicates.
`attach()` has a per-SDK lock, not a distributed deployment lock. Never silently
pick an arbitrary `AmbiguousDeploymentError.candidate_ids` entry.

Register custom profiles and resolve credentials in each process. Recovery trusts
stored metadata; reconcile out-of-band provider changes with `refresh()`.
`DeploymentNotActiveError` rejects terminal records. Deliberately deploy again
when appropriate. `find()` cannot discover apps absent from the local database.

Drain application requests before `close()` or explicit `stop()`. Idle protection
only covers requests in the owning SDK instance. Use one lifecycle owner per
deployment or disable full destruction (`destroy_after_idle_mins=None`) when
multiple replicas share it.

## Idle policy

`scaledown_window_seconds=300` releases idle containers while retaining the URL.
`destroy_after_idle_mins=1440` allows a longer idle interval before stopping the app.
The watchdog runs only while its process lives and a handle is registered/attached;
it is not a remote destroy schedule. Recovery resumes persisted activity.
Requests reset activity on start/end, including errors. Successful explicit probes
count as in-memory activity; aggressive polling can keep deployments alive and
wake GPUs. See [lifecycle and cost](../explanation/lifecycle-and-cost.md).
