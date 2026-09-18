"""Unit tests for InferWeave core abstractions, registry, and provider routing."""

import pytest

from inferweave import InferWeave, WorkloadType
from inferweave.core.exceptions import ModelNotFoundError, ProviderNotFoundError


def test_registry_builtins():
    weave = InferWeave()
    models = weave.registry.list_models()
    assert len(models) >= 4

    audio_models = weave.registry.list_models(workload_type=WorkloadType.AUDIO)
    assert any(m.id == "fish-s2-pro" for m in audio_models)

    llm_models = weave.registry.list_models(workload_type=WorkloadType.LLM)
    assert any("llama" in m.id.lower() for m in llm_models)


def test_registry_missing_model():
    weave = InferWeave()
    with pytest.raises(ModelNotFoundError):
        weave.registry.get("non-existent-model-xyz")


def test_router_list_providers():
    weave = InferWeave()
    providers = weave.router.list_providers()
    assert "runpod" in providers
    assert "modal" in providers
    assert "aws" in providers


def test_router_unknown_provider():
    weave = InferWeave()
    with pytest.raises(ProviderNotFoundError):
        weave.router.get("super-unknown-cloud")


@pytest.mark.asyncio
async def test_modal_deployment_flow():
    weave = InferWeave()
    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="modal",
    )
    assert deployment.id.startswith("iw-modal-")
    assert deployment.provider == "modal"
    assert deployment.model == "fish-s2-pro"
    assert deployment.endpoint_url is not None
    assert deployment.is_healthy
