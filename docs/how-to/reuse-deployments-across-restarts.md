# Reuse deployments across restarts

`InferWeave()` defaults to SQLite at `~/.inferweave/deployments.db`.
Set `INFERWEAVE_DEPLOYMENTS_PATH` before constructing the SDK to select a writable
shared path. Records survive application/process restarts.

```python
from inferweave import InferWeave

async def recover(deployment_id: str):
    weave = InferWeave()
    try:
        deployment = await weave.attach(deployment_id)
        return await deployment.synthesize("Back after a restart")
    finally:
        await weave.close()
```

`attach()` reconstructs callbacks, clients, identity, URL, timestamps and stored
options without provisioning or calling the provider. Use `refresh()` to
reconcile provider state and run an authenticated health probe. Stopped, failed
and dry-run records cannot be attached. Register custom profiles again first.

```python
from inferweave import AmbiguousDeploymentError, InferWeave

async def discover():
    weave = InferWeave()
    try:
        try:
            deployment = await weave.find(model="fish-s2-pro", provider="modal")
        except AmbiguousDeploymentError as error:
            raise RuntimeError(f"Choose one of {error.candidate_ids}") from error
        return deployment.id if deployment else None
    finally:
        await weave.close()
```

`find()` returns an attached handle, `None` when absent, or raises on multiple
matches. It filters persisted state, not live cloud truth; records may be stale
after an out-of-band stop. See [service startup](run-in-a-long-lived-service.md).

## Containers

Mount the directory, including SQLite's WAL/SHM side files, on a persistent volume:

```yaml
services:
  app:
    image: your-service-image
    environment:
      INFERWEAVE_DEPLOYMENTS_PATH: /app/data/inferweave/deployments.db
    volumes:
      - inference-state:/app/data/inferweave
volumes:
  inference-state:
```

Inject credentials separately at runtime. A read-only root needs this writable
mount. Keeping only the deployment ID is insufficient if the database vanishes.
SQLite WAL needs filesystem locking; prefer local persistent disks over unsupported
network filesystems. Shared storage does not provide distributed in-flight tracking
or an atomic find-or-deploy lock.

A runnable fresh-SDK demonstration is [examples/reuse.py](../../examples/reuse.py).
