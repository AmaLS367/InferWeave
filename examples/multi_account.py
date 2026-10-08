"""Deploy across several Modal accounts, use the endpoints, then restore and stop them.

Provisions real GPUs and incurs charges. Requires the variables named in
examples/accounts.example.yaml for the Modal accounts (other providers in that file are only
used when you deploy to them). Run from the repository root:

    python examples/multi_account.py
"""

import asyncio
from pathlib import Path

from inferweave import InferWeave

ACCOUNTS = Path(__file__).with_name("accounts.example.yaml")


async def deploy_two() -> list[str]:
    weave = InferWeave(accounts=ACCOUNTS)
    created = []
    try:
        # Round robin: consecutive deploys land on different Modal accounts.
        for _ in range(2):
            created.append(
                await weave.deploy("fish-s2-pro", provider="modal", destroy_after_idle_mins=30)
            )
        for deployment in created:
            audio = await deployment.synthesize(f"Served by account {deployment.account}.")
            print(deployment.id, deployment.account, len(audio), "bytes")
        for health in weave.account_health():
            print(health.provider, health.account_id, health.state, health.total_leases)
        return [deployment.id for deployment in created]
    except BaseException:
        for deployment in created:  # never leave paid GPUs behind on failure
            await deployment.stop()
        raise
    finally:
        await weave.close()  # local resources only; deployments keep running


async def restore_and_stop(deployment_ids: list[str]) -> None:
    # A new process: ownership comes from the persisted records, never from scheduling.
    weave = InferWeave(accounts=ACCOUNTS)
    try:
        for deployment_id in deployment_ids:
            status = await weave.get_status(deployment_id)
            print(deployment_id, status.account, status.state.value)
            await weave.stop(deployment_id)
        await weave.reconcile()
    finally:
        await weave.close()


async def main() -> None:
    deployment_ids = await deploy_two()
    await restore_and_stop(deployment_ids)


if __name__ == "__main__":
    asyncio.run(main())
