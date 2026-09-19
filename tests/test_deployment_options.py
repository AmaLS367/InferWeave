"""Unit tests for DeploymentOptions parsing, validation, and CLI formatting."""

from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions, RuntimeOptions
from inferweave.models.deployment import DeploymentRequest


def test_runtime_options_to_cli_args():
    opts = RuntimeOptions(
        engine_args={
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "disable_custom_all_reduce": False,
            "quantization": None,
        },
        extra_cli_args=["--disable-log-requests", "--api-key", "secret123"],
    )

    cli_args = opts.to_cli_args()
    assert "--max-model-len" in cli_args
    assert "4096" in cli_args
    assert "--gpu-memory-utilization" in cli_args
    assert "0.9" in cli_args
    assert "--dtype" in cli_args
    assert "bfloat16" in cli_args
    assert "--enforce-eager" in cli_args
    assert "--disable-custom-all-reduce" not in cli_args
    assert "--quantization" not in cli_args
    assert "--disable-log-requests" in cli_args
    assert "secret123" in cli_args


def test_deployment_options_flat_custom_args():
    raw_custom_args = {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "allow_spot": False,
        "max_price_per_hour": 1.5,
        "preferred_regions": ["us-east-1", "us-west-2"],
        "disk_size_gb": 100,
        "scaledown_window_seconds": 600,
        "autodown": True,
        "extra_args": ["--enable-chunked-prefill"],
    }

    opts = DeploymentOptions.from_custom_args(raw_custom_args, autostop_mins=20)

    # Runtime options
    assert opts.runtime.engine_args["max_model_len"] == 8192
    assert opts.runtime.engine_args["gpu_memory_utilization"] == 0.85
    assert "--enable-chunked-prefill" in opts.runtime.extra_cli_args

    # Provider options
    assert opts.provider.allow_spot is False
    assert opts.provider.max_price_per_hour == 1.5
    assert opts.provider.preferred_regions == ["us-east-1", "us-west-2"]
    assert opts.provider.disk_size_gb == 100
    assert opts.provider.scaledown_window_seconds == 600
    assert opts.provider.autodown is True

    # Autostop policy
    assert opts.autostop.idle_minutes == 20
    assert opts.autostop.action == AutostopAction.DOWN
    assert opts.autostop.enabled is True


def test_deployment_request_auto_populates_options():
    req = DeploymentRequest(
        model="fish-s2-pro",
        autostop_mins=15,
        custom_args={
            "max_model_len": 2048,
            "allow_spot": True,
        },
    )

    assert req.options is not None
    assert req.options.autostop.idle_minutes == 15
    assert req.options.runtime.engine_args["max_model_len"] == 2048
    assert req.options.provider.allow_spot is True
