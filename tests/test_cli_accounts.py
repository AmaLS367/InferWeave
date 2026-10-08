"""CLI coverage for account pools: --accounts, --account, `accounts` and `reconcile`."""

import json
from unittest.mock import AsyncMock, patch

import click
from typer.testing import CliRunner

from inferweave import cli
from inferweave.cli import app
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.models.enums import DeploymentState

runner = CliRunner()
SECRET = "SENTINEL-cli-runpod-secret"


def _accounts_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CLI_RUNPOD_A", SECRET)
    monkeypatch.setenv("CLI_RUNPOD_B", SECRET + "-b")
    path = tmp_path / "accounts.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "runpod": {
                        "strategy": "round_robin",
                        "accounts": [
                            {"id": "runpod-a", "env": {"api_key": "CLI_RUNPOD_A"}},
                            {"id": "runpod-b", "env": {"api_key": "CLI_RUNPOD_B"}},
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_accounts_command_lists_pooled_accounts_without_secrets(tmp_path, monkeypatch):
    path = _accounts_file(tmp_path, monkeypatch)
    result = runner.invoke(app, ["--accounts", str(path), "accounts"])
    output = click.unstyle(result.stdout)
    assert result.exit_code == 0, output
    assert "runpod-a" in output and "runpod-b" in output and "available" in output
    assert SECRET not in output


def test_accounts_command_reads_accounts_file_from_environment(tmp_path, monkeypatch):
    path = _accounts_file(tmp_path, monkeypatch)
    monkeypatch.setenv("INFERWEAVE_ACCOUNTS_FILE", str(path))
    result = runner.invoke(app, ["accounts"])
    assert result.exit_code == 0
    assert "runpod-a" in click.unstyle(result.stdout)


def test_accounts_command_without_pools_explains_ambient(monkeypatch):
    monkeypatch.delenv("INFERWEAVE_ACCOUNTS_FILE", raising=False)
    result = runner.invoke(app, ["accounts"])
    assert result.exit_code == 0
    assert "ambient" in click.unstyle(result.stdout)


def test_invalid_accounts_file_fails_without_echoing_content(tmp_path):
    path = tmp_path / "accounts.yaml"
    path.write_text(f"providers: [{SECRET}", encoding="utf-8")
    result = runner.invoke(app, ["--accounts", str(path), "accounts"])
    assert result.exit_code != 0
    assert SECRET not in click.unstyle(result.stdout)
    assert SECRET not in str(result.exception)


def test_deploy_forwards_pinned_account(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "cli.db"))
    with patch("inferweave.sdk.InferWeave.deploy", AsyncMock(side_effect=RuntimeError("stop"))) as deploy:
        runner.invoke(
            app, ["deploy", "fish-s2-pro", "--provider", "modal", "--account", "modal-b", "--no-wait"]
        )
    assert deploy.call_args.kwargs["account"] == "modal-b"


def test_reconcile_command_reports_updated_records(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "cli.db"))
    record = DeploymentRecord(
        id="iw-leftover", model="fish-s2-pro", provider="runpod", account="runpod-a",
        state=DeploymentState.STOPPED,
    )
    with patch("inferweave.sdk.InferWeave.reconcile", AsyncMock(return_value=[record])) as reconcile:
        result = runner.invoke(app, ["reconcile", "--min-age", "0"])
    output = click.unstyle(result.stdout)
    assert result.exit_code == 0, output
    assert reconcile.call_args.kwargs["min_age_seconds"] == 0
    assert "iw-leftover" in output and "runpod/runpod-a" in output


def test_reconcile_command_with_nothing_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "cli.db"))
    with patch("inferweave.sdk.InferWeave.reconcile", AsyncMock(return_value=[])):
        result = runner.invoke(app, ["reconcile"])
    assert result.exit_code == 0
    assert "Nothing to reconcile" in click.unstyle(result.stdout)


def test_list_shows_owning_account(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "cli.db"))
    record = DeploymentRecord(
        id="iw-owned", model="fish-s2-pro", provider="modal", account="modal-team-b",
        state=DeploymentState.HEALTHY, needs_reconciliation=True,
    )
    monkeypatch.setattr(cli.console, "width", 250)  # keep the table on one line per row
    with patch("inferweave.sdk.InferWeave.list_records", AsyncMock(return_value=[record])):
        result = runner.invoke(app, ["list"])
    output = click.unstyle(result.stdout)
    assert result.exit_code == 0, output
    assert "modal-team-b" in output
    assert "needs reconcile" in output
