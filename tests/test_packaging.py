"""Packaging regression tests: dependency bounds, PEP 561 marker, and built-wheel contents.

The built-wheel checks reuse .github/scripts/wheel_smoke.py (the same checks CI runs against
the wheel it installs into a clean environment) and need `uv` to build the wheel.
"""

import importlib.util
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SMOKE_SCRIPT = ROOT / ".github" / "scripts" / "wheel_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("wheel_smoke", SMOKE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_constrains_httpx_below_1_0():
    """httpx 1.x changes the 0.x API (AsyncClient usage); the upper bound must stay."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    (spec,) = [
        Requirement(dep) for dep in project["dependencies"] if dep.lower().startswith("httpx")
    ]
    assert spec.specifier.contains("0.28.1")
    assert not spec.specifier.contains("1.0.0")
    assert not spec.specifier.contains("1.0.0rc1")
    assert not spec.specifier.contains("1.2.0")


def test_py_typed_marker_ships_in_package_source():
    assert (ROOT / "src" / "inferweave" / "py.typed").is_file()


def test_vast_extra_is_a_real_install_path():
    """`vast` is a registered provider, so its install hint must point at a real extra."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert any(dep.startswith("skypilot[vast]") for dep in extras["vast"])
    assert any("vast" in dep for dep in extras["clouds"])


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel for packaging checks")
    out_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir), str(ROOT)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    (wheel,) = out_dir.glob("inferweave-*.whl")
    return wheel


def test_built_wheel_contains_py_typed(built_wheel):
    with zipfile.ZipFile(built_wheel) as wheel:
        assert "inferweave/py.typed" in wheel.namelist()


def test_built_wheel_metadata_passes_release_checks(built_wheel, capsys):
    """Runs the CI metadata check: httpx<1.0, no mandatory provider SDKs, extras, py.typed."""
    _load_smoke().check_metadata(str(built_wheel))
    assert "Metadata OK" in capsys.readouterr().out


def test_wheel_metadata_check_rejects_unbounded_httpx(tmp_path, built_wheel):
    """Dropping the httpx upper bound must fail the metadata check."""
    tampered = tmp_path / built_wheel.name
    with zipfile.ZipFile(built_wheel) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith(".dist-info/METADATA"):
                data = data.replace(b"httpx<1.0,>=0.28.1", b"httpx>=0.28.1")
                data = data.replace(b"httpx>=0.28.1,<1.0", b"httpx>=0.28.1")
                assert b"<1.0" not in data
            dst.writestr(item, data)

    with pytest.raises(SystemExit, match="httpx"):
        _load_smoke().check_metadata(str(tampered))


def test_wheel_metadata_check_rejects_broad_modal_bound(tmp_path, built_wheel):
    """Expanding the modal range to untested versions must fail the metadata check."""
    tampered = tmp_path / built_wheel.name
    with zipfile.ZipFile(built_wheel) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith(".dist-info/METADATA"):
                data = data.replace(b"modal<1.7,>=1.6", b"modal<2.0,>=1.3.0")
                data = data.replace(b"modal>=1.6,<1.7", b"modal>=1.3.0,<2.0")
            dst.writestr(item, data)

    with pytest.raises(SystemExit, match="Modal"):
        _load_smoke().check_metadata(str(tampered))


def test_version_is_single_sourced_from_package_module():
    """pyproject must not hardcode a version; Hatch reads it from inferweave/__init__.py."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "version" not in pyproject["project"]
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/inferweave/__init__.py"


def test_version_matches_distribution_metadata():
    """__version__ and the installed distribution metadata must agree (no literal version)."""
    import importlib.metadata

    import inferweave

    assert inferweave.__version__
    assert inferweave.__version__ == importlib.metadata.version("inferweave")


def test_built_wheel_version_matches_package_version(built_wheel):
    """The wheel METADATA Version (the artifact, not the source tree) equals __version__."""
    import inferweave

    assert _load_smoke().wheel_version(str(built_wheel)) == inferweave.__version__
    assert f"inferweave-{inferweave.__version__}-" in built_wheel.name


def test_cli_version_output_matches_package_version():
    import click
    from typer.testing import CliRunner

    import inferweave
    from inferweave.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert inferweave.__version__ in click.unstyle(result.stdout).split()


def test_wheel_smoke_does_not_hardcode_a_release_version():
    """Release checks must validate relationships, not a literal like '0.1.0'."""
    import re

    import inferweave

    source = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"""["']\d+\.\d+\.\d+["']""", source)
    assert inferweave.__version__ not in source


def test_pyproject_constrains_modal_to_tested_range():
    """pyproject.toml must declare the tested range modal>=1.6,<1.7."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    (spec,) = [
        Requirement(dep)
        for dep in project["optional-dependencies"]["modal"]
        if dep.lower().startswith("modal")
    ]
    assert spec.specifier.contains("1.6.0")
    assert not spec.specifier.contains("1.7.0")
    assert not spec.specifier.contains("2.0.0")


def test_pyproject_metadata_completeness():
    """Ensures mandatory PyPI metadata is factual, valid, and fully declared."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert project.get("license") == "Apache-2.0"
    assert "LICENSE" in project.get("license-files", [])
    assert any(a.get("name") == "AmaLS367" for a in project.get("authors", []))
    assert project.get("urls") == {
        "Homepage": "https://github.com/AmaLS367/InferWeave",
        "Repository": "https://github.com/AmaLS367/InferWeave",
        "Issues": "https://github.com/AmaLS367/InferWeave/issues",
    }, "Only factual URLs; add Changelog only together with a real changelog document"
    assert "Typing :: Typed" in project.get("classifiers", [])


def test_built_wheel_project_urls_are_factual(built_wheel):
    from email.parser import Parser

    with zipfile.ZipFile(built_wheel) as wheel:
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(name).decode("utf-8"))
    labels = {u.split(",", 1)[0].strip() for u in metadata.get_all("Project-URL") or []}
    assert labels == {"Homepage", "Repository", "Issues"}
