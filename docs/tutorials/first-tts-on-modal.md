# First TTS on Modal

Deploy Fish Speech S2 Pro, synthesize speech and save `speech.wav`. Use Python
3.11 or 3.12, a Modal account with GPU access and a suitable payment method.
Review the [upstream licenses](../../README.md#third-party-licenses): S2 uses a
pinned beta runtime and separately licensed weights. Loading can take minutes.

## Install and authenticate

```bash
pip install "inferweave[modal]"
python -m modal setup
```

`modal setup` opens a browser and saves **SDK/account** credentials locally. For
a headless service, inject `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` through your
secret manager instead. These credentials manage resources; they are not HTTP
endpoint tokens. See [Modal setup](https://modal.com/docs/guide) and
[account token configuration](https://modal.com/docs/sdk/py/latest/config).

Create a separate **Proxy Token** in Modal dashboard Settings, in the same
workspace/environment as the deployment. Set `MODAL_PROXY_TOKEN_ID` and
`MODAL_PROXY_TOKEN_SECRET` in the process running InferWeave. Copy values from
the dashboard; keep them out of source, images, SQLite and shell history.
PowerShell uses `$env:NAME`; Linux/macOS uses `export NAME`. Both values are required.

InferWeave protects `modal.web_server` with `requires_proxy_auth=True` and sends
`Modal-Key`/`Modal-Secret` headers on readiness and inference requests. The current
[Proxy Token guide](https://modal.com/docs/guide/webhook-proxy-auth) also describes
CLI token creation; dashboard creation works with the tested SDK range `>=1.6,<1.7`.
Account tokens and Proxy Tokens cannot be substituted for one another.

## Deploy, synthesize and save

Save this as `tts.py` and run `python tts.py`. The same program is in
[examples/tts.py](../../examples/tts.py).

<!-- example: tts -->
```python
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
```


`deploy()` waits for readiness by default. `synthesize()` sends MessagePack to
`/v1/tts` and returns encoded WAV bytes. The `finally` block stops the remote app
and then closes local pools/watchdogs. `cleanup_on_failure=True` also stops it
if readiness times out before a handle is returned.

## Cold starts and billing

The first deployment builds/downloads its runtime and loads weights. Later
requests to an app that scaled to zero reload a GPU container. Connection errors
and 502/503 responses retry with finite backoff; configurable timeouts and typed
errors keep failure explicit. See [cold starts](../how-to/handle-cold-starts.md).

Modal bills container load time, processing and the configured warm idle interval.
Scale-to-zero ends container compute charges while retaining the app; storage,
egress or plan charges may still apply. `stop()` stops the whole app. Check
[official pricing](https://modal.com/pricing) for current rates and
[lifecycle and cost](../explanation/lifecycle-and-cost.md) for the two idle policies.
