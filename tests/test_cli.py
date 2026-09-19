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
    assert "vast" in result.stdout
    assert "oci" in result.stdout
    assert "fluidstack" in result.stdout
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
    with patch("inferweave.sdk.InferWeave.stop", AsyncMock(return_value=None)) as mock_stop:
        result = runner.invoke(app, ["stop", "iw-modal-test-123456"])
        assert result.exit_code == 0
        assert "stopped successfully" in result.stdout
        mock_stop.assert_called_once()
        _, kwargs = mock_stop.call_args
        from inferweave.domain.lifecycle import AutostopAction
        assert kwargs.get("action") == AutostopAction.STOP


def test_cli_stop_action_down():
    from inferweave.domain.lifecycle import AutostopAction

    with patch("inferweave.sdk.InferWeave.stop", AsyncMock(return_value=None)) as mock_stop:
        result = runner.invoke(app, ["stop", "iw-modal-test-123456", "--action", "down"])
        assert result.exit_code == 0
        assert "stopped successfully" in result.stdout
        mock_stop.assert_called_once()
        _, kwargs = mock_stop.call_args
        assert kwargs.get("action") == AutostopAction.DOWN


def test_cli_stop_invalid_action():
    result = runner.invoke(app, ["stop", "iw-modal-test-123456", "--action", "potato"])
    assert result.exit_code != 0
    output = result.output or (result.stderr if hasattr(result, "stderr") else "")
    assert "Invalid value for '--action'" in output or "potato" in output or "Error" in output



def test_cli_cross_process_lifecycle(tmp_path):
    import os
    import re
    env_file = tmp_path / "cli_deployments.db"
    os.environ["INFERWEAVE_DEPLOYMENTS_PATH"] = str(env_file)
    try:
        # 1. Deploy dry-run
        res_deploy = runner.invoke(app, ["deploy", "fish-s2-pro", "--provider", "modal", "--dry-run"])
        assert res_deploy.exit_code == 0
        assert "iw-modal-fish-s2-pro-" in res_deploy.stdout

        # Extract deployment ID from output
        match = re.search(r"iw-modal-fish-s2-pro-[a-f0-9]+", res_deploy.stdout)
        assert match is not None
        dep_id = match.group(0)

        # 2. In a brand new command execution, run 'inferweave list'
        res_list = runner.invoke(app, ["list"], env={"COLUMNS": "200"})
        assert res_list.exit_code == 0
        assert dep_id in res_list.stdout
        assert "fish-s2-pro" in res_list.stdout

        # 3. Run 'inferweave status <dep_id>' without mocks
        res_status = runner.invoke(app, ["status", dep_id])
        assert res_status.exit_code == 0
        assert dep_id in res_status.stdout

        # 4. Run 'inferweave stop <dep_id>' without mocks
        res_stop = runner.invoke(app, ["stop", dep_id])
        assert res_stop.exit_code == 0
        assert "stopped successfully" in res_stop.stdout

        # 5. Check status is now stopped
        res_status_after = runner.invoke(app, ["status", dep_id])
        assert res_status_after.exit_code == 0
        assert "stopped" in res_status_after.stdout.lower()

    finally:
        del os.environ["INFERWEAVE_DEPLOYMENTS_PATH"]


def test_cli_status_not_found(tmp_path):
    import os
    os.environ["INFERWEAVE_DEPLOYMENTS_PATH"] = str(tmp_path / "empty.db")
    try:
        res = runner.invoke(app, ["status", "iw-nonexistent-12345"])
        assert res.exit_code != 0
        assert "not found" in res.stdout.lower()
    finally:
        del os.environ["INFERWEAVE_DEPLOYMENTS_PATH"]


def test_cli_stop_not_found(tmp_path):
    import os
    os.environ["INFERWEAVE_DEPLOYMENTS_PATH"] = str(tmp_path / "empty.db")
    try:
        res = runner.invoke(app, ["stop", "iw-nonexistent-12345"])
        assert res.exit_code != 0
        assert "not found" in res.stdout.lower()
    finally:
        del os.environ["INFERWEAVE_DEPLOYMENTS_PATH"]


def test_cli_deploy_invalid_strategy():
    result = runner.invoke(
        app,
        [
            "deploy",
            "fish-s2-pro",
            "--strategy",
            "potato",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "Validation error" in result.stdout
    assert "potato" in result.stdout


def test_cli_deploy_autostop_zero():
    result = runner.invoke(
        app,
        [
            "deploy",
            "fish-s2-pro",
            "--provider",
            "modal",
            "--autostop",
            "0",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run completed successfully" in result.stdout


def test_cli_deploy_healthcheck_timeout_warning():
    from inferweave.core.exceptions import HealthcheckTimeoutError

    timeout_err = HealthcheckTimeoutError(
        endpoint_url="https://test.modal.run/health",
        timeout_seconds=300.0,
        total_probes=10,
        deployment_id="iw-modal-test-orphan123",
    )
    with patch("inferweave.sdk.InferWeave.deploy", AsyncMock(side_effect=timeout_err)):
        result = runner.invoke(
            app,
            [
                "deploy",
                "fish-s2-pro",
                "--provider",
                "modal",
            ],
        )
        assert result.exit_code != 0
        assert "Readiness timeout" in result.stdout
        assert "inferweave stop iw-modal-test-orphan123" in result.stdout



