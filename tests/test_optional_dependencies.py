"""Regression tests: the core SDK must not require optional provider SDKs (SkyPilot, Modal)."""

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from inferweave.providers.modal_provider import ModalProvider
from inferweave.providers.skypilot import SkyPilotProvider

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Installs an import hook that makes `sky` and `modal` unimportable, simulating a base
# `pip install inferweave` without any provider extras.
_BLOCK_PROVIDER_SDKS = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    BLOCKED = ("sky", "modal", "modal_proto", "lightning_sdk")

    class _Blocker(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"{name} blocked to simulate a base install")
            return None

    sys.meta_path.insert(0, _Blocker())
    """
)


def _run_without_provider_sdks(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["INFERWEAVE_DEPLOYMENTS_PATH"] = str(tmp_path / "deployments.db")
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_PROVIDER_SDKS + textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_core_import_surface_works_without_provider_sdks(tmp_path):
    result = _run_without_provider_sdks(
        """
        import inferweave
        from inferweave import InferWeave

        assert inferweave.__version__
        weave = InferWeave()
        assert "modal" in weave.router.list_providers()
        assert "lightning" in weave.router.list_providers()
        assert "runpod" in weave.router.list_providers()

        import inferweave.cli
        assert inferweave.cli.app is not None

        leaked = [m for m in ("sky", "modal", "lightning_sdk") if m in sys.modules]
        assert not leaked, f"provider SDKs imported eagerly: {leaked}"
        print("OK")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_skypilot_dry_run_works_without_skypilot_installed(tmp_path):
    result = _run_without_provider_sdks(
        """
        import asyncio
        from inferweave import InferWeave, DeploymentState

        async def main():
            weave = InferWeave()
            dep = await weave.deploy(
                model="fish-s2-pro", provider="runpod", gpu_type="L4", dry_run=True
            )
            assert dep.state == DeploymentState.PROVISIONING
            assert "sky" not in sys.modules

        asyncio.run(main())
        print("OK")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def _run_worker(module, request: dict, monkeypatch, capsys) -> tuple[int, dict]:
    """Runs a worker's ``main()`` in-process with ``request`` on stdin; returns (rc, response)."""
    import io
    import json

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    rc = module.main()
    last = [ln for ln in capsys.readouterr().out.splitlines() if "iw_worker" in ln][-1]
    return rc, json.loads(last)


@pytest.mark.parametrize("cloud", ["runpod", "fluidstack"])
def test_skypilot_worker_without_skypilot_reports_not_installed(cloud, monkeypatch, capsys):
    """The host never imports SkyPilot; a worker interpreter lacking it fails with a clear error."""
    from inferweave.isolation import skypilot_worker

    request = {"op": "probe", "payload": {"cloud": cloud, "mode": "ambient"}, "secrets": {}}
    with patch.dict(sys.modules, {"sky": None}):
        rc, response = _run_worker(skypilot_worker, request, monkeypatch, capsys)

    assert rc != 0 and response["ok"] is False
    assert response["error"]["kind"] == "invalid_request"
    assert "SkyPilot is not installed" in response["error"]["message"]
    assert response["error"]["resource_may_exist"] is False


def test_skypilot_provider_never_imports_skypilot_in_the_host_process():
    provider = SkyPilotProvider(cloud_name="runpod")
    assert not hasattr(provider, "_get_sky_module")
    from inferweave.providers import skypilot as skypilot_module

    assert "import sky\n" not in Path(skypilot_module.__file__).read_text(encoding="utf-8")


def test_missing_modal_raises_actionable_install_hint():
    provider = ModalProvider()
    with (
        patch.dict(sys.modules, {"modal": None}),
        pytest.raises(ImportError, match=r"pip install 'inferweave\[modal\]'"),
    ):
        provider._get_modal_module()


def test_lightning_worker_without_sdk_reports_not_installed(monkeypatch, capsys):
    from inferweave.isolation import lightning_worker

    request = {"op": "whoami", "payload": {}, "secrets": {}}
    with patch.dict(sys.modules, {"lightning_sdk": None, "lightning_sdk.cli.utils.auth": None}):
        rc, response = _run_worker(lightning_worker, request, monkeypatch, capsys)

    assert rc != 0 and response["ok"] is False
    assert response["error"]["kind"] == "invalid_request"
    assert "lightning-sdk is not installed" in response["error"]["message"]
    assert response["error"]["resource_may_exist"] is False


def test_lightning_provider_never_imports_the_sdk_in_the_host_process():
    from inferweave.providers import lightning_provider

    assert not hasattr(lightning_provider.LightningProvider, "_sdk")
    assert "import lightning_sdk" not in Path(lightning_provider.__file__).read_text(
        encoding="utf-8"
    )


def test_lightning_dry_run_works_without_sdk_or_credentials(tmp_path):
    result = _run_without_provider_sdks(
        """
        import asyncio
        from inferweave import InferWeave, DeploymentState
        async def main():
            weave = InferWeave()
            deployment = await weave.deploy("fish-s2-pro", provider="lightning", dry_run=True)
            assert deployment.state == DeploymentState.PROVISIONING
            assert "lightning_sdk" not in sys.modules
            await deployment.stop()
            await weave.close()
        asyncio.run(main())
        print("OK")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_pyproject_keeps_provider_sdks_out_of_base_dependencies():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    base = [dep.lower() for dep in project["dependencies"]]
    assert not any(dep.startswith(("skypilot", "modal", "lightning-sdk")) for dep in base), base

    extras = project["optional-dependencies"]
    for extra in ("runpod", "aws", "gcp", "azure", "lambda", "nebius", "kubernetes", "clouds"):
        assert any(dep.startswith("skypilot") for dep in extras[extra]), extra
    assert any(dep.startswith("modal") for dep in extras["modal"])
