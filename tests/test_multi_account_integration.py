"""Explicitly opted-in multi-account smoke test. Provisions real GPUs on two accounts.

Enable with INFERWEAVE_MULTI_ACCOUNT_INTEGRATION=1, an accounts file in
INFERWEAVE_ACCOUNTS_FILE with at least two accounts for INFERWEAVE_MULTI_ACCOUNT_PROVIDER
(default "modal"), and the credentials those accounts reference.
"""

import os

import pytest

from inferweave import AccountsConfig, DeploymentState, InferWeave

PROVIDER = os.getenv("INFERWEAVE_MULTI_ACCOUNT_PROVIDER", "modal")


def _skip_reason() -> str | None:
    if os.getenv("INFERWEAVE_MULTI_ACCOUNT_INTEGRATION") != "1":
        return "Set INFERWEAVE_MULTI_ACCOUNT_INTEGRATION=1 to explicitly enable paid testing."
    path = os.getenv("INFERWEAVE_ACCOUNTS_FILE")
    if not path:
        return "Set INFERWEAVE_ACCOUNTS_FILE to an accounts configuration."
    pool = AccountsConfig.from_file(path).providers.get(PROVIDER)
    if pool is None or len(pool.accounts) < 2:
        return f"The accounts file needs at least two '{PROVIDER}' accounts."
    return None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or ""),
]


@pytest.mark.asyncio
async def test_live_round_robin_affinity_restart_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "multi-account.db"))
    weave1 = InferWeave()
    weave2 = None
    try:
        first = await weave1.deploy("fish-s2-pro", provider=PROVIDER, destroy_after_idle_mins=30)
        second = await weave1.deploy("fish-s2-pro", provider=PROVIDER, destroy_after_idle_mins=30)
        assert first.account != second.account
        for deployment in (first, second):
            assert deployment.state == DeploymentState.HEALTHY
            assert await deployment.synthesize("Multi-account smoke test")
        await weave1.close()

        weave2 = InferWeave()
        for original in (first, second):
            attached = await weave2.attach(original.id)
            assert attached.account == original.account
            assert (await attached.refresh()).account == original.account
            await attached.stop()
            assert attached.state == DeploymentState.STOPPED
    finally:
        cleanup = weave2 or weave1
        errors = []
        for record in await cleanup.list_records():
            if record.state != DeploymentState.STOPPED or record.needs_reconciliation:
                try:
                    await cleanup.stop(record.id, action="down")
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{type(error).__name__}: {record.id}")
        await cleanup.reconcile(min_age_seconds=0)
        if weave2 is not None:
            await weave2.close()
        await weave1.close()
        assert not errors, "Cleanup could not be verified: " + ", ".join(errors)
