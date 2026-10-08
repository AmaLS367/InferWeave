"""Explicitly opted-in Lightning L4 TTS/recovery test. Consumes real GPU credits."""

import io
import os
import wave

import httpx
import pytest

from inferweave import DeploymentState, InferenceConfig, InferWeave, ProviderAccount
from inferweave.providers.lightning_provider import LightningProvider


def _skip_reason() -> str | None:
    if os.getenv("INFERWEAVE_LIGHTNING_INTEGRATION") != "1":
        return "Set INFERWEAVE_LIGHTNING_INTEGRATION=1 to explicitly enable paid Lightning testing."
    missing = [name for name in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "LIGHTNING_TEAMSPACE")
               if not os.getenv(name)]
    return "Missing local Lightning configuration: " + ", ".join(missing) if missing else None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or ""),
]


def validate_wav(audio: bytes) -> None:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() > 0
        assert wav.getframerate() > 0
        assert wav.getnframes() > 0
        assert wav.readframes(wav.getnframes())


@pytest.mark.asyncio
async def test_live_lightning_tts_restart_and_full_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "live-lightning.db"))
    # Large image pulls/weight downloads can exceed the standard readiness budget.
    config = InferenceConfig(timeout_seconds=900)
    weave1 = InferWeave(inference_config=config)
    profile = weave1.registry.get("fish-s2-pro").model_copy(deep=True)
    profile.healthcheck.timeout_seconds = 1800
    profile.healthcheck.request_timeout_seconds = 30
    weave1.register_model(profile)
    weave2 = None
    provider = weave1.router.get("lightning")
    assert isinstance(provider, LightningProvider)
    provider.resource_prefix = "iw-lightning-test"
    try:
        # Read-only identity check precedes creation of any GPU resource.
        await provider.verify_credentials(ProviderAccount.ambient("lightning"))
        deployment = await weave1.deploy(
            model="fish-s2-pro", provider="lightning", gpu_type="L4",
            destroy_after_idle_mins=60,
            custom_args={"lightning": {"min_replicas": 0, "max_replicas": 1,
                                       "idle_threshold_seconds": 300}},
        )
        assert deployment.endpoint_url
        assert deployment.state == DeploymentState.HEALTHY
        assert (await deployment.check_health()).is_healthy
        async with httpx.AsyncClient() as client:
            denied = await client.get(deployment.endpoint_url + "/v1/health")
            assert denied.status_code in (401, 403)
        audio1 = await deployment.synthesize("InferWeave Lightning integration test")
        validate_wav(audio1)
        (tmp_path / "lightning-first.wav").write_bytes(audio1)
        deployment_id = deployment.id
        await weave1.close()

        weave2 = InferWeave(inference_config=config)
        weave2.register_model(profile)
        attached = await weave2.attach(deployment_id)
        assert attached is not deployment
        assert (await attached.refresh()).state == DeploymentState.HEALTHY
        assert await weave2.find(model="fish-s2-pro", provider="lightning") is attached
        audio2 = await attached.synthesize("Recovered Lightning deployment")
        validate_wav(audio2)
        (tmp_path / "lightning-recovered.wav").write_bytes(audio2)
        await attached.stop()
        assert attached.state == DeploymentState.STOPPED
    finally:
        # Include FAILED records: deploy/readiness may raise before returning a handle.
        # Continue cleanup for all records even if one attempt fails.
        errors = []
        for record in await weave1.list_records():
            if record.provider != "lightning" or not record.resource or not record.resource.owned:
                continue
            try:
                account = weave1.accounts.resolve(record.provider, record.account, record.id)
                await provider.stop(record, account)
                assert await provider.resource_snapshot(record, account) is None, (
                    "Live Lightning resource remains after cleanup: " + record.id
                )
            except Exception as error:  # noqa: BLE001
                errors.append(type(error).__name__ + ": " + record.id)
        try:
            if weave2 is not None:
                await weave2.close()
        finally:
            await weave1.close()
        assert not errors, "Lightning cleanup could not be verified: " + ", ".join(errors)
