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

    BLOCKED = ("sky", "modal", "modal_proto")

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
        assert "runpod" in weave.router.list_providers()

        import inferweave.cli
        assert inferweave.cli.app is not None

        leaked = [m for m in ("sky", "modal") if m in sys.modules]
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


def test_missing_skypilot_raises_actionable_install_hint():
    provider = SkyPilotProvider(cloud_name="runpod")
    with (
        patch.dict(sys.modules, {"sky": None}),
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        pytest.raises(ImportError, match=r"pip install 'inferweave\[runpod\]'"),
    ):
        provider._get_sky_module()


def test_missing_skypilot_for_uncatalogued_cloud_suggests_clouds_extra():
    provider = SkyPilotProvider(cloud_name="fluidstack")
    with (
        patch.dict(sys.modules, {"sky": None}),
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        pytest.raises(ImportError, match=r"pip install 'inferweave\[clouds\]'"),
    ):
        provider._get_sky_module()


def test_missing_modal_raises_actionable_install_hint():
    provider = ModalProvider()
    with (
        patch.dict(sys.modules, {"modal": None}),
        pytest.raises(ImportError, match=r"pip install 'inferweave\[modal\]'"),
    ):
        provider._get_modal_module()


def test_pyproject_keeps_provider_sdks_out_of_base_dependencies():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    base = [dep.lower() for dep in project["dependencies"]]
    assert not any(dep.startswith(("skypilot", "modal")) for dep in base), base

    extras = project["optional-dependencies"]
    for extra in ("runpod", "aws", "gcp", "azure", "lambda", "nebius", "kubernetes", "clouds"):
        assert any(dep.startswith("skypilot") for dep in extras[extra]), extra
    assert any(dep.startswith("modal") for dep in extras["modal"])
