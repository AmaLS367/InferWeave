# Runnable examples

Install `inferweave[modal]`, configure account/Proxy Token credentials and follow
[the tutorial](../docs/tutorials/first-tts-on-modal.md). From the repository root:

```bash
python examples/tts.py
python examples/image.py
python examples/reuse.py
```

These provision GPUs and incur charges. TTS/image stop apps in `finally`;
recovery demonstrates close, fresh SDK attach, inference and explicit stop.
Reuse stops the uniquely found app: use a dedicated database rather than
one serving production traffic.

## Multiple accounts

[`accounts.example.yaml`](accounts.example.yaml) registers two Modal and two Lightning
accounts plus RunPod and Vast.ai pools by environment variable name.
[`multi_account.py`](multi_account.py) deploys twice (round robin lands on both Modal
accounts), runs inference, then restores the deployments in a fresh SDK and stops them under
their owning accounts. See [use multiple accounts](../docs/how-to/use-multiple-accounts.md).

```bash
python examples/multi_account.py
```

## Credential-free validation

With only the base package installed:

```bash
python examples/simulate.py
```

The harness substitutes compute/HTTP boundaries via public injection points,
runs unchanged examples and marked doc snippets using real SDK, SQLite and
MessagePack/base64 clients, and checks outputs/stopped records in temporary
directories. No cloud or private test helpers. Other Python snippets are compiled
and local Markdown links checked. This validates contracts, not live GPU quality
or provisioning/scale-to-zero latency.
