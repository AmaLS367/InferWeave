# Use Lightning AI

Install on Python 3.11 or 3.12:

```bash
pip install "inferweave[lightning]"
```

Validated against the installed `lightning-sdk==2026.10.1` API. The extra is bounded
to this release (`>=2026.10.1,<2026.10.2`). Ordinary imports and dry runs do not
import Lightning. Lightning and SkyPilot currently have incompatible Click
requirements (`>=8.2` versus `<8.2`); use separate environments. The `all` extra
retains SkyPilot, Modal and workers and therefore excludes Lightning.
Lightning can coexist with Modal and worker extras.

## Configure local credentials

Put these in your ignored local `.env`, or export them before creating the SDK:

```dotenv
LIGHTNING_USER_ID=your-user-id
LIGHTNING_API_KEY=your-user-api-key
LIGHTNING_TEAMSPACE=owner/teamspace
INFERWEAVE_LIGHTNING_INTEGRATION=0
```

Use a **user API key** from Lightning settings. Platform requests use the SDK's
environment authentication. Deployment `ApiKeyAuth()` accepts that same key as
`Authorization: Bearer …`; no separate endpoint key is needed. Scoped org API
keys cannot call these endpoints and are rejected by a read-only identity check
before creation. InferWeave never creates keys.

`LIGHTNING_TEAMSPACE` must identify your chosen `owner/teamspace`; the owner
can be an organization or a user. This avoids ambiguous defaults and deprecated
separate SDK `org=`/`user=` parameters. Alternatively set a bare teamspace name
and `LIGHTNING_ORG`, or override the slug with
`custom_args={"lightning": {"teamspace": "owner/teamspace"}}`.
Do not store credentials in model env, custom args, SQLite or code.
The library does not load `.env` automatically.

The default `LightningEndpointAuth` sends the Bearer key only to managed HTTPS
`*.cloudspaces.litng.ai` endpoints. Readiness, status health probes and inference
use the same resolver, including after recovery. Custom domain auth requires an
explicit `endpoint_auth=` resolver and is not configured by this provider.

## Deploy and recover

```python
import asyncio
from pathlib import Path
from inferweave import InferWeave

async def main():
    weave1 = InferWeave()
    weave2 = None
    previous_ids = {record.id for record in await weave1.list_records()}
    try:
        deployment = await weave1.deploy(
            "fish-s2-pro", provider="lightning", gpu_type="L4",
            destroy_after_idle_mins=60,
            custom_args={"lightning": {
                "min_replicas": 0, "max_replicas": 1,
                "idle_threshold_seconds": 300,
            }},
        )
        Path("speech.wav").write_bytes(await deployment.synthesize("Hello from Lightning"))
        deployment_id = deployment.id
        await weave1.close()
        weave2 = InferWeave()
        attached = await weave2.attach(deployment_id)
        await attached.refresh()
        Path("recovered.wav").write_bytes(await attached.synthesize("Recovered deployment"))
        await attached.stop()
    finally:
        for record in await weave1.list_records():
            if record.id not in previous_ids and record.state.value != "stopped":
                await (weave2 or weave1).stop(record.id)
        if weave2:
            await weave2.close()
        await weave1.close()

asyncio.run(main())
```

`find(model="fish-s2-pro", provider="lightning")` uses the same persisted records.
Fresh-process attach restores the shared inference clients, health/readiness,
timestamps, idle activity and lifecycle callbacks. Credentials resolve afresh.

CLI:

```bash
inferweave providers
inferweave deploy fish-s2-pro --provider lightning --gpu L4
inferweave deploy fish-s2-pro --provider lightning --dry-run
inferweave stop <deployment-id>
```

## Resource strategy and limits

The provider creates one uniquely named `iw-lightning-…` **container Deployment**,
using the existing `RuntimeSpec` image, setup commands, environment, port and
literal launch argv. It bypasses the image entrypoint with `/bin/sh`, runs setup
sequentially and replaces the shell with the foreground server. Built-in worker
source is packaged into the startup command from the installed InferWeave
package; it requires no new PyPI release. There are **no Studios**, snapshots,
uploaded container builds or background interactive sessions to leak.
`include_credentials=False` prevents the SDK from injecting platform secrets
into the container.

Fish S2 Pro is the required live target. FLUX uses the same generic mechanism
and the existing image client; real FLUX validation is not part of the TTS test.
Large images, package installation and weight downloads happen on cold replicas,
so readiness/cold starts may take longer than the ordinary 300-second budget.
For a first pull, register a copied model profile with a longer
`healthcheck.timeout_seconds`, or deploy with `wait_for_ready=False` and call
`wait_for_ready(timeout_secs=1800)`. Setup is repeated on a new replica.
Only public container images are supported; private registry credential plumbing
is not exposed by the verified public SDK and is not patched here.

Supported GPU names: T4, L4, L40S, A100-40GB, A100-80GB, H100, H200, B200.
Supported counts: 1/2/4/8, except B200 1/8. The shared GPU catalog resolves aliases
and validates VRAM; the provider rejects unavailable mappings rather than silently
substituting another GPU. Fish needs at least 24 GB: L4 is the live-test target;
T4 is rejected. SDK machine support does not guarantee account entitlement,
regional availability or free-tier access. Lightning has no static cost offer in
InferWeave's auto router; explicit `provider="lightning"` uses existing model GPU
recommendations. Provisioning is on-demand (`spot=False`); generic spot/price/
region options do not configure Lightning.

## Scaling, stop and failure cleanup

Defaults: `min_replicas=0`, `max_replicas=1`, GPU utilization threshold 90,
`idle_threshold_seconds=300`. Idle scale-to-zero retains an endpoint that wakes
on requests. A positive minimum keeps GPUs allocated and can consume credits
continuously. These settings are independent of `destroy_after_idle_mins`;
Modal's `scaledown_window_seconds` does not configure Lightning.

InferWeave `stop()` and watchdog destruction **delete** the owned Deployment.
The SDK's `Deployment.stop()` only scales down, and `Deployment.delete()` is an
HTTP helper, so teardown uses the official CLI shipped in the same SDK:
`python -m lightning_sdk.cli.entrypoint deployment delete … --yes`.
Deletion is asynchronous; InferWeave checks the live resource until it disappears,
with a five-minute confirmation budget. A timeout reports failure and allows retry.
Resources not owned by InferWeave are protected. No Studio reuse API is exposed.

Synchronous SDK/CLI operations run in threads. Cancellation waits for a pending
create to finish before cleanup. Provisioning, endpoint discovery and SDK deploy/
readiness failures trigger cleanup automatically. Cleanup failures preserve the
original error and log only the InferWeave ID; use that ID to retry `weave.stop()`.
Infrastructure replica presence maps to STARTING/PROVISIONING, never HEALTHY;
application health remains authoritative. Idle zero replicas stay attachable.

Records add typed nonsensitive `lightning` metadata: name, resolved teamspace,
resource ID, ownership. Existing records load without it; keys never enter this
metadata. `close()` stops local monitoring, not remote compute. InferWeave's
in-flight protection is local to the monitoring process.

## Opt-in real integration test

Do not enable this until you intend to consume credits. Configure the three
required variables, then set `INFERWEAVE_LIGHTNING_INTEGRATION=1` locally:

```bash
uv run --env-file .env --extra lightning pytest tests/test_lightning_integration.py -m integration --strict-markers -ra
```

It performs a read-only identity check, creates one `iw-lightning-test-…`
Deployment with one L4, authenticates readiness, validates WAV inference, closes
the first SDK, attaches with a fresh SDK/SQLite state, validates a second WAV,
then deletes and independently checks absence in `finally`. No Studio is
created. Missing credentials or opt-in skips the test; normal CI never enables it.
Initial readiness allows 1800 seconds and inference 900 seconds.

Credits/free-tier eligibility varies by account and machine. GPUs can consume
credits during startup, downloads, inference and idle allocation; free credits
are not a promise of permanently free usage. Review current
[Lightning pricing](https://lightning.ai/pricing) and
[container deployment documentation](https://lightning.ai/docs/overview/deploy-containers).
The verified SDK surface and teardown contract are documented in the
[official SDK repository](https://github.com/Lightning-AI/sdk) and
[official deployment guide](https://github.com/Lightning-AI/skills/blob/main/lightning-deployments/SKILL.md).
