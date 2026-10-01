"""Unit tests for DeploymentOptions parsing, validation, and CLI formatting."""

import pytest

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


# --- extra_cli_args normalization (top-level legacy form and nested runtime_args form) ---

EXTRA_ARGS_CASES = [
    # (raw value, expected argv tokens)
    ('--foo "hello world"', ["--foo", "hello world"]),
    ("--foo 'it is'", ["--foo", "it is"]),
    ("--a=1   --b   2", ["--a=1", "--b", "2"]),
    ("--x ; touch /tmp/pwned", ["--x", ";", "touch", "/tmp/pwned"]),
    ("--x $(id) `id` && rm -rf / | cat", ["--x", "$(id)", "`id`", "&&", "rm", "-rf", "/", "|", "cat"]),
    ('--x "$(id) ; &&"', ["--x", "$(id) ; &&"]),
    (["--foo", "hello world"], ["--foo", "hello world"]),
    (("--foo", "hello world"), ["--foo", "hello world"]),
    (["--x", "; touch /tmp/pwned", "$(id)", "a b"], ["--x", "; touch /tmp/pwned", "$(id)", "a b"]),
    ("", []),
    ([], []),
    (None, []),
]


@pytest.mark.parametrize(("raw", "expected"), EXTRA_ARGS_CASES)
def test_normalize_extra_cli_args(raw, expected):
    from inferweave.domain.options import normalize_extra_cli_args

    assert normalize_extra_cli_args(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), EXTRA_ARGS_CASES)
def test_top_level_extra_cli_args(raw, expected):
    opts = DeploymentOptions.from_custom_args({"extra_cli_args": raw})
    assert opts.runtime.extra_cli_args == expected


@pytest.mark.parametrize(("raw", "expected"), EXTRA_ARGS_CASES)
def test_nested_runtime_args_extra_cli_args(raw, expected):
    opts = DeploymentOptions.from_custom_args({"runtime_args": {"extra_cli_args": raw}})
    assert opts.runtime.extra_cli_args == expected


def test_nested_string_is_not_split_into_characters():
    opts = DeploymentOptions.from_custom_args(
        {"runtime_args": {"extra_cli_args": '--foo "value with spaces"'}}
    )
    assert opts.runtime.extra_cli_args == ["--foo", "value with spaces"]


def test_nested_and_top_level_extra_cli_args_are_combined_in_order():
    opts = DeploymentOptions.from_custom_args(
        {
            "runtime_args": {"extra_cli_args": '--nested "a b"'},
            "extra_cli_args": ["--top", "c d"],
        }
    )
    assert opts.runtime.extra_cli_args == ["--nested", "a b", "--top", "c d"]


def test_legacy_extra_args_alias_is_consumed_not_leaked_into_engine_args():
    opts = DeploymentOptions.from_custom_args(
        {"extra_cli_args": "--a", "extra_args": '--b "c d"'}
    )
    assert opts.runtime.extra_cli_args == ["--a", "--b", "c d"]
    assert "extra_args" not in opts.runtime.engine_args


@pytest.mark.parametrize("bad", ['--foo "unterminated', 5, {"a": 1}])
def test_invalid_extra_cli_args_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        DeploymentOptions.from_custom_args({"runtime_args": {"extra_cli_args": bad}})
    with pytest.raises((ValueError, TypeError)):
        DeploymentOptions.from_custom_args({"extra_cli_args": bad})


def test_runtime_options_model_normalizes_string_directly():
    from inferweave.domain.options import RuntimeOptions

    assert RuntimeOptions(extra_cli_args='--foo "a b"').extra_cli_args == ["--foo", "a b"]


@pytest.mark.parametrize(("raw", "expected"), EXTRA_ARGS_CASES)
@pytest.mark.parametrize("nested", [False, True])
def test_extra_cli_args_reach_runtime_argv_as_literal_tokens(raw, expected, nested):
    """End to end through a real template: tokens stay literal and the shell form round-trips."""
    import shlex

    from inferweave.models.deployment import DeploymentRequest
    from inferweave.registry.base import ModelRegistry
    from inferweave.runtimes.templates import get_runtime_template

    profile = ModelRegistry().get("meta-llama/Meta-Llama-3-8B-Instruct")
    custom = {"runtime_args": {"extra_cli_args": raw}} if nested else {"extra_cli_args": raw}
    spec = get_runtime_template("vllm").render(
        profile, DeploymentRequest(model=profile.id, custom_args=custom)
    )

    if expected:
        assert spec.run_args[-len(expected):] == expected
    assert shlex.split(spec.run_command) == spec.run_args
