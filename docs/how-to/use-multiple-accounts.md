# Use multiple provider accounts

InferWeave can rotate deployments across several accounts per provider with
[CredWeave](https://github.com/AmaLS367/CredWeave) 0.1: each new deployment leases an account
from that provider's pool, failures drive cooldowns and failover, and every deployment stays
bound to the account that created it until it is destroyed.

Supported pools: **Modal**, **Lightning AI**, **RunPod** and **Vast.ai**. Other SkyPilot clouds,
and any provider without a pool, keep using their native default credentials (the `ambient`
account), exactly as before.

## Credentials each provider needs

| Provider | Secret fields | Metadata | Notes |
| --- | --- | --- | --- |
| `modal` | `token_id`, `token_secret`; optional `proxy_token_id` + `proxy_token_secret` | `environment` | The API token manages apps. The proxy token pair is a separate workspace credential sent as `Modal-Key`/`Modal-Secret` to protected endpoints; it is required unless you deploy with `custom_args={"requires_proxy_auth": False}`. |
| `lightning` | `user_id`, `api_key` | `teamspace` (`owner/teamspace`) | A user API key (scoped keys are refused). The same key is the endpoint's Bearer token. A per-deploy `custom_args={"lightning": {"teamspace": ...}}` overrides the account teamspace. |
| `runpod` | `api_key` | — | Runs through SkyPilot (Linux, macOS or WSL2). |
| `vast` | `api_key` | — | Runs through SkyPilot (Linux, macOS or WSL2). |

Every account also accepts the CredWeave scheduling metadata `weight`, `priority`,
`max_concurrency` and `tags`.

## Configure accounts

The configuration names *where* secrets live (environment variable names or a protected JSON
credentials file); it never contains secret values, so it is safe to commit.

```yaml
# accounts.yaml
max_attempts: 3            # provisioning attempts per deploy, across accounts/providers
providers:
  modal:                   # two Modal workspaces
    strategy: round_robin
    accounts:
      - id: modal-team-a
        env:
          token_id: MODAL_A_TOKEN_ID
          token_secret: MODAL_A_TOKEN_SECRET
          proxy_token_id: MODAL_A_PROXY_TOKEN_ID
          proxy_token_secret: MODAL_A_PROXY_TOKEN_SECRET
      - id: modal-team-b
        environment: main
        env:
          token_id: MODAL_B_TOKEN_ID
          token_secret: MODAL_B_TOKEN_SECRET
          proxy_token_id: MODAL_B_PROXY_TOKEN_ID
          proxy_token_secret: MODAL_B_PROXY_TOKEN_SECRET
  lightning:               # two Lightning users, primary/backup
    strategy: failover
    accounts:
      - id: lightning-primary
        priority: 0
        teamspace: acme/inference
        env: {user_id: LIGHTNING_A_USER_ID, api_key: LIGHTNING_A_API_KEY}
      - id: lightning-backup
        priority: 1
        teamspace: acme-backup/inference
        env: {user_id: LIGHTNING_B_USER_ID, api_key: LIGHTNING_B_API_KEY}
  runpod:                  # three RunPod accounts, weighted 2:1:1
    strategy: weighted
    accounts:
      - {id: runpod-main, weight: 2, env: {api_key: RUNPOD_MAIN_API_KEY}}
      - {id: runpod-b, env: {api_key: RUNPOD_B_API_KEY}}
      - {id: runpod-c, env: {api_key: RUNPOD_C_API_KEY}}
  vast:                    # Vast.ai accounts from a hot-reloaded credentials file
    credentials_file: vast-accounts.json
```

`vast-accounts.json` is a CredWeave `JsonSource` file. Keep it out of version control and
readable only by you (`chmod 600`):

```json
{
  "credentials": [
    {"id": "vast-a", "secrets": {"api_key": "<key>"}},
    {"id": "vast-b", "secrets": {"api_key": "<key>"}, "metadata": {"max_concurrency": 2}}
  ]
}
```

Point InferWeave at the configuration with `InferWeave(accounts="accounts.yaml")`,
`export INFERWEAVE_ACCOUNTS_FILE=accounts.yaml`, or `inferweave --accounts accounts.yaml ...`.

The same configuration in Python:

```python
from inferweave import (
    AccountsConfig,
    InferWeave,
    ProviderAccounts,
    lightning_account,
    modal_account,
    runpod_account,
    vast_account,
)

accounts = AccountsConfig(
    {
        "modal": ProviderAccounts(
            accounts=[
                modal_account(
                    "modal-team-a",
                    token_id_env="MODAL_A_TOKEN_ID",
                    token_secret_env="MODAL_A_TOKEN_SECRET",
                    proxy_token_id_env="MODAL_A_PROXY_TOKEN_ID",
                    proxy_token_secret_env="MODAL_A_PROXY_TOKEN_SECRET",
                ),
                modal_account(
                    "modal-team-b",
                    token_id_env="MODAL_B_TOKEN_ID",
                    token_secret_env="MODAL_B_TOKEN_SECRET",
                    proxy_token_id_env="MODAL_B_PROXY_TOKEN_ID",
                    proxy_token_secret_env="MODAL_B_PROXY_TOKEN_SECRET",
                    environment="main",
                ),
            ]
        ),
        "lightning": ProviderAccounts(
            strategy="failover",
            accounts=[
                lightning_account(
                    "lightning-primary",
                    user_id_env="LIGHTNING_A_USER_ID",
                    api_key_env="LIGHTNING_A_API_KEY",
                    teamspace="acme/inference",
                    priority=0,
                ),
                lightning_account(
                    "lightning-backup",
                    user_id_env="LIGHTNING_B_USER_ID",
                    api_key_env="LIGHTNING_B_API_KEY",
                    teamspace="acme-backup/inference",
                    priority=1,
                ),
            ],
        ),
        "runpod": ProviderAccounts(
            strategy="weighted",
            accounts=[
                runpod_account("runpod-main", api_key_env="RUNPOD_MAIN_API_KEY", weight=2),
                runpod_account("runpod-b", api_key_env="RUNPOD_B_API_KEY"),
            ],
        ),
        "vast": ProviderAccounts(
            accounts=[vast_account("vast-a", api_key_env="VAST_A_API_KEY")],
            credentials_file="vast-accounts.json",
        ),
    }
)
weave = InferWeave(accounts=accounts)
```

## Deploy, infer, check status and stop

Nothing changes in the normal workflow; the account is chosen for you and reported back:

```python
import asyncio

from inferweave import InferWeave


async def main() -> None:
    weave = InferWeave(accounts="accounts.yaml")
    deployment = await weave.deploy(model="fish-s2-pro", provider="modal")
    print(deployment.id, deployment.account)  # e.g. iw-modal-fish-s2-pro-1a2b3c modal-team-b
    try:
        audio = await deployment.synthesize("Hello from the second workspace.")
        print(len(audio), "bytes")
        status = await weave.get_status(deployment.id)
        print(status.state, status.account)
    finally:
        await deployment.stop()
        await weave.close()


asyncio.run(main())
```

Pin an account when you need one explicitly (no failover happens then):
`await weave.deploy(model="fish-s2-pro", provider="modal", account="modal-team-a")`.

## Rotation, cooldown and failover

Account selection uses the pool's CredWeave strategy: `round_robin` (default), `weighted`,
`failover` (by `priority`), `least_used`, `least_recently_used` or `random`. Each provisioning
attempt reports an outcome:

| Failure | Detection | Account effect | Next account tried? |
| --- | --- | --- | --- |
| Invalid credentials | HTTP 401 / auth errors | revoked until `authorize()` | yes, if creation was rejected or safely settled |
| Permission denied | HTTP 403, missing entitlement, scoped key | parked for `permission_cooldown_seconds` (900), health unchanged | yes, if creation was rejected or safely settled |
| Quota or billing exhausted | HTTP 402, insufficient balance | parked until the provider's reset hint or `quota_cooldown_seconds` (3600) | yes, if creation was rejected or safely settled |
| Rate limited | HTTP 429 | parked for `Retry-After` | yes, if creation was rejected or safely settled |
| GPU stock-out | no capacity for the GPU | none (not a credential fault) | yes, if creation was rejected or safely settled |
| Transient | timeouts, network errors, HTTP 5xx | cooldown; unhealthy after `max_consecutive_failures` | only after original creation settled and owner-bound cleanup/absence |
| Invalid request | HTTP 400/404/422, bad GPU/model combination | none | **no** (fails the same everywhere) |

Attempts are bounded by `max_attempts` (default 3) across all accounts and, for
`provider="auto"`, across the other feasible providers in the routing ranking. When nothing is
left, `NoAccountAvailableError` lists every attempt and the shortest known cooldown in
`retry_after`. With a single account the original classified error is raised instead.

`inferweave accounts` (or `weave.account_health()`) shows each account's state, failures, usage
and cooldown. After topping up a quota call `await weave.accounts.reset(provider, account_id)`;
after repairing a revoked key call `await weave.accounts.authorize(provider, account_id)`.
CredWeave health, cooldown and concurrency state are in-memory by default and shared only
within that AccountManager, not across processes. Deployment ownership and recovery are durable.

### Rotating and removing keys

Keys are read again on every use. New deployments can use changed control-plane keys under
the same configured id. Existing pooled deployments also store a nonsecret digest of their
original control credentials: a different key is **not proof of the same cloud identity**.
Until identity verification is available in the provider adapters, changed control keys raise
`AccountUnavailableError` for those deployments. Restore the original keys to manage them,
or clean them up in the provider console. Modal endpoint proxy tokens are separate and can
rotate without changing this ownership digest.

Missing, removed or CredWeave-revoked owners fail explicitly; no other account is substituted.
Legacy and no-config `ambient` records retain native SDK credential resolution and do not have
this credential-generation binding. Keep the native cloud identity stable when managing them.

## Provisioning recovery

Before creation, records persist the owner, credential-generation digest, remote resource name,
`needs_reconciliation=True` and `creation_may_continue=True`. A per-deployment process lock
protects provisioning, stop and reconciliation in the built-in SQLite and JSON repositories.
Active provisioning stays protected even beyond `min_age_seconds` (default 30 minutes).
Custom repositories must implement a durable `operation_lock` to provide the same protection.

Definitively rejected requests can fail over. A failed create can retry only after the original
operation has settled and absence or cleanup is confirmed under its owner. A timeout, crash,
or ambiguous create response can leave work running remotely: observing absence or deleting a
currently visible resource does not prove that another resource cannot appear later.
InferWeave raises `ProvisioningUncertainError` and blocks further attempts in that case.
Status and stop cannot clear the pending-create flag just because the resource looks absent.

`await weave.reconcile()` / `inferweave reconcile` recover settled leftovers. For records with
`creation_may_continue=True`, first confirm from the provider's operation history or console
that the original create has completed or was cancelled, then explicitly acknowledge it:

```python
await weave.reconcile(confirmed_settled=("deployment-id",))
```

```bash
inferweave reconcile --confirm-settled deployment-id
```

This acknowledgement permits owner-bound inspection and cleanup; it does not override an
active local process lock. Age alone is never used as proof that remote work has finished.

## Restore and manage deployments after a restart

The SQLite record stores only nonsecret ownership (`provider`, `account`, resource name,
scope and credential-generation digest). A new process with the same accounts configuration
can restore handles and resolve their original owners:

```python
import asyncio

from inferweave import InferWeave


async def main() -> None:
    weave = InferWeave(accounts="accounts.yaml")
    for record in await weave.list_records():
        print(record.id, record.provider, record.account, record.state.value)
    deployment = await weave.find(model="fish-s2-pro", provider="modal")
    if deployment is not None:
        await deployment.refresh()  # status under its owning account
        await deployment.stop()
    await weave.reconcile()
    await weave.close()


asyncio.run(main())
```

## How credentials are isolated

* **Modal** runs in-process with an explicit `modal.Client.from_credentials(...)` per account
  (and the account's Modal environment); the process environment is never changed.
* **Lightning AI** authenticates once per process from environment variables, so every
  Lightning operation runs in a short-lived worker process that receives only its account's
  key on stdin and gets a private `HOME`; your own `~/.lightning` is never read or written.
* **RunPod and Vast.ai** go through SkyPilot, which reads keys only from files under `~` and
  caches them in its API server. Each account therefore gets a private SkyPilot home (state,
  clusters and a `0600` key file) under `~/.inferweave/accounts/skypilot/` and its own local API
  API, metrics and request-queue ports. A plugin uses SkyPilot 0.13 public queue APIs to
  avoid its shared default queue. Control-credential generations use separate homes.
* Worker processes never get secrets on their command line, never inherit other accounts' or
  your default provider credentials, and their output is never echoed into errors or logs.
* Deployment records, logs, exceptions and model containers never contain secrets. A deploy is
  refused if a platform credential (or a variable such as `MODAL_TOKEN_ID`) would be injected
  into the model's runtime environment.

## Run all four providers side by side

Lightning SDK and SkyPilot have conflicting dependencies, so install them in their own
environments and point InferWeave at those interpreters. The workers need only the provider SDK:

```bash
uv venv ~/.inferweave/envs/lightning && uv pip install --python ~/.inferweave/envs/lightning "lightning-sdk==2026.10.1"
uv venv ~/.inferweave/envs/skypilot && uv pip install --python ~/.inferweave/envs/skypilot "skypilot[runpod,vast]>=0.13.0,<0.14"
export INFERWEAVE_LIGHTNING_PYTHON=~/.inferweave/envs/lightning/bin/python
export INFERWEAVE_SKYPILOT_PYTHON=~/.inferweave/envs/skypilot/bin/python
pip install "inferweave[modal]"   # the main environment only needs Modal
```

SkyPilot (RunPod, Vast.ai) needs Linux, macOS or WSL2; native Windows is unsupported upstream.

## What still needs live verification

The account layer is covered by offline tests with mocked SDKs. WSL checks exercise real
Lightning 2026.10.1/SkyPilot 0.13.0 imports, worker protocols and concurrent local SkyPilot
servers with dummy credentials; they do not call cloud APIs or prove live deployment behavior.
Before relying on it in
production, verify with real accounts: Modal deploy/lookup/stop under explicit clients and
environments; Lightning deploy/inspect/delete from the isolated worker; per-account SkyPilot API
operations for RunPod and Vast.ai (`sky check`, launch/status/down, credential-generation isolation); and the exact
error shapes each provider returns for 402/403/429 so the failure classification above holds.
