"""Integration tests for VRAM-aware routing, provider-scoped instance selection, and fail-fast validation."""

import pytest

from inferweave import (
    HardwareRequirements,
    InferWeave,
    InsufficientVramError,
    ModelProfile,
    NoFeasibleProviderError,
    WorkloadType,
)


@pytest.mark.asyncio
async def test_sdk_deploy_explicit_insufficient_vram_raises_on_modal():
    weave = InferWeave()

    # wan-video/wan-2.1 requires 24GB VRAM
    # Requesting T4 (16GB) on Modal must raise InsufficientVramError before provisioning
    with pytest.raises(InsufficientVramError) as exc_info:
        await weave.deploy(
            model="wan-video/wan-2.1",
            provider="modal",
            gpu_type="T4",
            num_gpus=1,
            dry_run=True,
        )

    err = exc_info.value
    assert err.model_id == "wan-video/wan-2.1"
    assert err.required_vram_gb == 24.0
    assert err.provided_vram_gb == 16.0
    assert err.deficit_gb == 8.0
    assert err.gpu_type == "T4"
    assert any("L4" in s or "A100" in s for s in err.suggested_gpus)


@pytest.mark.asyncio
async def test_sdk_deploy_explicit_insufficient_vram_raises_on_skypilot():
    weave = InferWeave()

    # FLUX.1 requires 24GB VRAM
    # Requesting T4 (16GB) on RunPod must raise InsufficientVramError immediately
    with pytest.raises(InsufficientVramError) as exc_info:
        await weave.deploy(
            model="black-forest-labs/FLUX.1-schnell",
            provider="runpod",
            gpu_type="T4",
            num_gpus=1,
            dry_run=True,
        )

    err = exc_info.value
    assert err.model_id == "black-forest-labs/FLUX.1-schnell"
    assert err.required_vram_gb == 24.0
    assert err.provided_vram_gb == 16.0
    assert err.deficit_gb == 8.0


@pytest.mark.asyncio
async def test_sdk_deploy_explicit_insufficient_vram_raises_on_auto():
    weave = InferWeave()

    # Requesting T4 for 24GB model in auto routing mode
    with pytest.raises(InsufficientVramError):
        await weave.deploy(
            model="wan-video/wan-2.1",
            provider="auto",
            gpu_type="T4",
            num_gpus=1,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_sdk_deploy_multi_gpu_tensor_parallel_passes():
    weave = InferWeave()

    # wan-video requires 24GB. 2x T4 provides 32GB VRAM >= 24GB.
    deployment = await weave.deploy(
        model="wan-video/wan-2.1",
        provider="modal",
        gpu_type="T4",
        num_gpus=2,
        dry_run=True,
    )

    assert deployment.model == "wan-video/wan-2.1"
    assert deployment.provider == "modal"


@pytest.mark.asyncio
async def test_sdk_deploy_provider_scoped_selects_sufficient_vram():
    weave = InferWeave()

    # Deploying 24GB FLUX model on Modal without specifying gpu_type:
    # System must pick L4 (24GB, $0.80/hr), skipping T4 (16GB, $0.59/hr)
    deployment = await weave.deploy(
        model="black-forest-labs/FLUX.1-schnell",
        provider="modal",
        strategy="cheapest",
        dry_run=True,
    )

    assert deployment.provider == "modal"
    decision = weave.router.last_decision
    assert decision is not None
    assert decision.chosen_offer.gpu_spec.total_vram(1) >= 24.0
    assert decision.chosen_offer.gpu_spec.name == "L4"


@pytest.mark.asyncio
async def test_sdk_deploy_provider_scoped_impossible_vram_raises():
    weave = InferWeave()

    # Register custom massive model requiring 500GB VRAM
    weave.register_model(
        ModelProfile(
            id="custom/massive-model-500gb",
            name="Massive 500GB Model",
            workload_type=WorkloadType.LLM,
            default_runtime="vllm",
            hardware=HardwareRequirements(min_vram_gb=500.0),
        )
    )

    # Modal's maximum instance is 80GB VRAM, so routing must fail with NoFeasibleProviderError
    with pytest.raises(NoFeasibleProviderError) as exc_info:
        await weave.deploy(
            model="custom/massive-model-500gb",
            provider="modal",
            dry_run=True,
        )

    assert "No compute provider satisfies requirements" in str(exc_info.value)
