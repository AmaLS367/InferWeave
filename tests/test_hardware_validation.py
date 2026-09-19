"""Unit tests for GPU hardware specifications, capacity evaluation, and HardwareValidator."""

import pytest

from inferweave.adapters.catalog.static_gpu_catalog import StaticGpuCatalogAdapter
from inferweave.core.exceptions import InsufficientVramError, UnknownGpuError
from inferweave.domain.hardware_validator import HardwareValidator
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import HardwareRequirements, ModelProfile
from inferweave.models.routing import GpuSpec
from inferweave.services.hardware_service import HardwareValidationService


@pytest.fixture
def sample_profile_flux() -> ModelProfile:
    return ModelProfile(
        id="black-forest-labs/FLUX.1-schnell",
        name="FLUX.1 Schnell",
        workload_type=WorkloadType.IMAGE,
        default_runtime="flux-diffusers",
        hardware=HardwareRequirements(min_vram_gb=24.0),
    )


@pytest.fixture
def sample_profile_llama70b() -> ModelProfile:
    return ModelProfile(
        id="meta-llama/Meta-Llama-3-70B-Instruct",
        name="Llama 3 70B",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=140.0),
    )


def test_gpu_spec_total_vram_and_sufficiency():
    t4 = GpuSpec(name="T4", vram_gb=16.0)
    assert t4.total_vram(gpu_count=1) == 16.0
    assert t4.total_vram(gpu_count=2) == 32.0
    assert t4.total_vram(gpu_count=4) == 64.0

    # Single T4 does not fit 24GB
    assert not t4.is_sufficient_for(min_vram_gb=24.0, gpu_count=1)
    # 2x T4 (32GB) fits 24GB
    assert t4.is_sufficient_for(min_vram_gb=24.0, gpu_count=2)


def test_static_gpu_catalog_lookup():
    catalog = StaticGpuCatalogAdapter()

    # Exact match
    t4 = catalog.get_gpu("T4")
    assert t4 is not None
    assert t4.vram_gb == 16.0

    # Case-insensitive alias matching
    a100 = catalog.get_gpu("a100")
    assert a100 is not None
    assert a100.vram_gb == 80.0

    a100_40 = catalog.get_gpu("A100-40GB")
    assert a100_40 is not None
    assert a100_40.vram_gb == 40.0

    rtx4090 = catalog.get_gpu("4090")
    assert rtx4090 is not None
    assert rtx4090.name == "RTX4090"
    assert rtx4090.vram_gb == 24.0

    # Unknown GPU
    unknown = catalog.get_gpu("non-existent-gpu-999")
    assert unknown is None


def test_static_gpu_catalog_find_sufficient():
    catalog = StaticGpuCatalogAdapter()
    sufficient_single = catalog.find_sufficient_gpus(min_vram_gb=24.0, gpu_count=1)

    # T4 (16GB) must NOT be in sufficient list for single GPU
    assert not any(g.name == "T4" for g in sufficient_single)
    # L4 (24GB), A10G (24GB), A100 (80GB), H100 (80GB) must be present
    names = [g.name for g in sufficient_single]
    assert "L4" in names
    assert "A10G" in names
    assert "H100" in names


def test_hardware_validator_passes_when_vram_is_sufficient(sample_profile_flux):
    validator = HardwareValidator()
    l4 = GpuSpec(name="L4", vram_gb=24.0)

    # Should succeed without raising exception
    validator.validate(sample_profile_flux, l4, gpu_count=1)


def test_hardware_validator_raises_insufficient_vram(sample_profile_flux):
    validator = HardwareValidator()
    t4 = GpuSpec(name="T4", vram_gb=16.0)
    a100 = GpuSpec(name="A100-80GB", vram_gb=80.0)
    l4 = GpuSpec(name="L4", vram_gb=24.0)

    with pytest.raises(InsufficientVramError) as exc_info:
        validator.validate(
            profile=sample_profile_flux,
            gpu_spec=t4,
            gpu_count=1,
            known_gpus=[t4, l4, a100],
        )

    err = exc_info.value
    assert err.model_id == "black-forest-labs/FLUX.1-schnell"
    assert err.required_vram_gb == 24.0
    assert err.provided_vram_gb == 16.0
    assert err.deficit_gb == 8.0
    assert err.gpu_type == "T4"
    assert "L4" in err.suggested_gpus
    assert "deficit: 8.0GB" in str(err)


def test_hardware_validator_passes_with_multi_gpu(sample_profile_flux):
    validator = HardwareValidator()
    t4 = GpuSpec(name="T4", vram_gb=16.0)

    # 2x T4 gives 32GB VRAM >= 24GB required
    validator.validate(sample_profile_flux, t4, gpu_count=2)


def test_hardware_service_preflight_validation(sample_profile_flux):
    service = HardwareValidationService()

    # Incompatible GPU: T4 (16GB) for 24GB model
    bad_req = DeploymentRequest(
        model=sample_profile_flux.id,
        gpu_type="T4",
        num_gpus=1,
    )
    with pytest.raises(InsufficientVramError):
        service.validate_deployment_hardware(sample_profile_flux, bad_req)

    # Compatible GPU: L4 (24GB) for 24GB model
    good_req = DeploymentRequest(
        model=sample_profile_flux.id,
        gpu_type="L4",
        num_gpus=1,
    )
    spec = service.validate_deployment_hardware(sample_profile_flux, good_req)
    assert spec is not None
    assert spec.name == "L4"


def test_hardware_service_unknown_gpu_handling(sample_profile_flux):
    service = HardwareValidationService()
    req = DeploymentRequest(
        model=sample_profile_flux.id,
        gpu_type="Custom-Accelerator-X",
    )

    # In non-strict mode: logs warning and returns None
    assert (
        service.validate_deployment_hardware(sample_profile_flux, req, strict=False)
        is None
    )

    # In strict mode: raises UnknownGpuError
    with pytest.raises(UnknownGpuError):
        service.validate_deployment_hardware(sample_profile_flux, req, strict=True)
