"""Guards that live-cloud tests stay opt-in and out of normal CI."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runpod_live_test_is_marked_integration():
    module = _load("test_runpod_integration")
    marks = {m.name for m in module.pytestmark}
    assert "integration" in marks
    assert "skipif" in marks


@pytest.mark.parametrize(
    "missing",
    [
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
    ],
)
def test_modal_live_requires_both_credential_pairs(monkeypatch, missing):
    module = _load("test_modal_integration")
    for name in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
    ):
        monkeypatch.setenv(name, "test-placeholder")
    assert module._skip_reason() is None
    monkeypatch.delenv(missing)
    assert missing in module._skip_reason()


def test_proxy_token_fixture_does_not_replace_live_credentials(monkeypatch):
    from types import SimpleNamespace

    from conftest import modal_proxy_tokens

    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "live-placeholder-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "live-placeholder-secret")
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: True))
    modal_proxy_tokens.__wrapped__(monkeypatch, request)
    import os

    assert os.environ["MODAL_PROXY_TOKEN_ID"] == "live-placeholder-id"
    assert os.environ["MODAL_PROXY_TOKEN_SECRET"] == "live-placeholder-secret"


def test_runpod_live_test_is_skipped_without_explicit_opt_in(monkeypatch):
    module = _load("test_runpod_integration")
    monkeypatch.delenv("INFERWEAVE_RUNPOD_INTEGRATION", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "present-but-not-opted-in")
    assert "opt-in" in (module._skip_reason() or "")


def test_runpod_live_test_is_skipped_without_credentials(monkeypatch, tmp_path):
    module = _load("test_runpod_integration")
    monkeypatch.setenv("INFERWEAVE_RUNPOD_INTEGRATION", "1")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert "credentials" in (module._skip_reason() or "")


def test_integration_tests_are_deselected_in_normal_runs():
    """`pytest -m "not integration"` (the CI invocation) must not select any live test."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not integration",
            "--collect-only",
            "-q",
            "--strict-markers",
            str(TESTS / "test_runpod_integration.py"),
            str(TESTS / "test_modal_integration.py"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "test_live_runpod_lifecycle" not in result.stdout
    assert "test_live_modal_deployment" not in result.stdout
    assert "deselected" in result.stdout + result.stderr
