"""Unit tests for InferWeave CLI commands."""

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from inferweave.cli import app
from inferweave.models.deployment import DeploymentStatus
from inferweave.models.enums import DeploymentState

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.stdout
    assert "models" in result.stdout
    assert "providers" in result.stdout
    assert "status" in result.stdout
    assert "stop" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_models_list():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "fish-s2-pro" in result.stdout
    assert "Llama 3" in result.stdout


def test_cli_models_workload_filter():
    result = runner.invoke(app, ["models", "--workload", "audio"])
    assert result.exit_code == 0
    assert "fish-s2-pro" in result.stdout
    assert "Meta-Llama" not in result.stdout


def test_cli_models_invalid_workload():
    result = runner.invoke(app, ["models", "--workload", "unknown_category"])
    assert result.exit_code != 0
    assert "Invalid workload type" in result.stdout


def test_cli_providers_list():
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "runpod" in result.stdout
    assert "modal" in result.stdout
    assert "aws" in result.stdout
    assert "SkyPilot" in result.stdout


def test_cli_deploy_dry_run():
    result = runner.invoke(
        app,
        [
            "deploy",
            "fish-s2-pro",
            "--provider",
            "modal",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run completed successfully" in result.stdout
    assert "iw-modal-" in result.stdout
    assert "fish-s2-pro" in result.stdout


def test_cli_deploy_with_options_dry_run():
    result = runner.invoke(
        app,
        [
            "deploy",
            "fish-s2-pro",
            "--provider",
            "modal",
            "--gpu",
            "A100",
            "--num-gpus",
            "1",
            "--autostop",
            "45",
            "--env",
            "TEST_ENV=hello",
            "--custom-arg",
            "max_model_len=2048",
            "--custom-arg",
            "is_fast=true",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run completed successfully" in result.stdout
    assert "45" in result.stdout


def test_cli_deploy_invalid_model():
    result = runner.invoke(
        app,
        [
            "deploy",
            "non-existent-model-xyz",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "Deployment failed" in result.stdout


def test_cli_status():
    mock_status = DeploymentStatus(
        id="iw-modal-test-123456",
        model="fish-s2-pro",
        provider="modal",
        state=DeploymentState.HEALTHY,
        endpoint_url="https://test.modal.run",
    )
    with patch("inferweave.sdk.InferWeave.get_status", AsyncMock(return_value=mock_status)):
        result = runner.invoke(app, ["status", "iw-modal-test-123456"])
        assert result.exit_code == 0
        assert "iw-modal-test-123456" in result.stdout
        assert "fish-s2-pro" in result.stdout
        assert "https://test.modal.run" in result.stdout


def test_cli_stop():
    with patch("inferweave.sdk.InferWeave.stop", AsyncMock(return_value=None)):
        result = runner.invoke(app, ["stop", "iw-modal-test-123456"])
        assert result.exit_code == 0
        assert "stopped successfully" in result.stdout
