"""Release smoke checks for a built InferWeave wheel.

Usage:
    python wheel_smoke.py metadata <path-to-wheel>
        Verifies provider SDKs (SkyPilot, Modal) are not mandatory dependencies, the
        httpx upper bound is present, provider extras exist, and py.typed is packaged.

    python wheel_smoke.py imports [<path-to-wheel>]
        Run with the interpreter of a clean environment that has ONLY the base wheel
        installed. Verifies the public import surface and CLI entry point. With a wheel
        path, the installed version must also equal the wheel's METADATA Version.

The version is never hardcoded: inferweave.__version__, the installed distribution metadata,
the wheel METADATA Version and the `inferweave --version` output must all agree.

    python wheel_smoke.py typed
        Run with the interpreter of a clean environment that has the base wheel and mypy
        installed. Verifies the installed package is PEP 561 type-discoverable.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

OPTIONAL_PROVIDER_SDKS = ("skypilot", "modal")

_PEP440_VERSION = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def wheel_version(wheel_path: str) -> str:
    """Returns the Version field of the wheel's METADATA (the artifact, not the source tree)."""
    with zipfile.ZipFile(wheel_path) as wheel:
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(name).decode("utf-8"))
    version = metadata.get("Version")
    if not version or not _PEP440_VERSION.match(version):
        sys.exit(f"Wheel Version metadata missing or not a valid version: {version!r}")
    return version


def check_metadata(wheel_path: str) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(name).decode("utf-8"))

    # Normalize marker quoting (backends emit either extra == 'x' or extra == "x").
    requires = [r.replace('"', "'") for r in metadata.get_all("Requires-Dist") or []]
    mandatory = [r for r in requires if "extra ==" not in r]
    print("Mandatory Requires-Dist:")
    for req in mandatory:
        print(f"  {req}")

    leaked = [r for r in mandatory if r.lower().startswith(OPTIONAL_PROVIDER_SDKS)]
    if leaked:
        sys.exit(f"Provider SDKs must be optional extras, found mandatory: {leaked}")

    # httpx 1.x is a different API (0.x AsyncClient usage); the upper bound must not be dropped.
    # Metadata orders specifiers arbitrarily ("httpx<1.0,>=0.28.1"), so compare as a set.
    httpx_specs = {
        frozenset(r.replace(" ", "")[len("httpx") :].split(","))
        for r in mandatory
        if r.lower().startswith("httpx")
    }
    if httpx_specs != {frozenset({">=0.28.1", "<1.0"})}:
        sys.exit(f"httpx must be constrained to '>=0.28.1,<1.0', got: {mandatory}")

    if not any(r.lower().startswith("msgpack") for r in mandatory):
        sys.exit("MessagePack must be a mandatory runtime dependency for audio inference")

    version = wheel_version(wheel_path)
    print(f"Wheel METADATA Version: {version}")

    with zipfile.ZipFile(wheel_path) as wheel:
        if "inferweave/py.typed" not in wheel.namelist():
            sys.exit("Wheel is missing inferweave/py.typed (PEP 561 marker)")

    extras = set(metadata.get_all("Provides-Extra") or [])
    expected = {
        "runpod", "aws", "gcp", "azure", "lambda", "nebius", "kubernetes", "vast", "clouds", "modal", "workers", "all",
    }
    if missing := expected - extras:
        sys.exit(f"Missing expected extras: {sorted(missing)}")

    for extra, sdk in [(e, "skypilot") for e in sorted(expected - {"modal", "workers", "all"})] + [("modal", "modal")]:
        if not any(r.lower().startswith(sdk) and f"extra == '{extra}'" in r for r in requires):
            sys.exit(f"Extra '{extra}' does not provide '{sdk}'")

    # Modal must be optional and bounded to the verified range (>=1.6,<1.7)
    modal_reqs = [
        r for r in requires if "extra == 'modal'" in r and r.lower().startswith("modal")
    ]
    if not modal_reqs:
        sys.exit("Missing Requires-Dist for extra == 'modal'")
    modal_specs = {
        frozenset(r.replace(" ", "").split(";")[0][len("modal") :].split(","))
        for r in modal_reqs
    }
    if modal_specs != {frozenset({">=1.6", "<1.7"})}:
        sys.exit(f"Modal must be bounded to '>=1.6,<1.7', got: {modal_reqs}")

    print("Metadata OK: SkyPilot and Modal only via extras, httpx<1.0 bounded, modal range tested, py.typed packaged.")


def check_imports(wheel_path: str | None = None, *, with_modal: bool = False) -> None:
    import importlib.metadata
    import importlib.util
    from importlib.metadata import entry_points

    for sdk in (("sky",) if with_modal else ("sky", "modal")):
        if importlib.util.find_spec(sdk) is not None:
            sys.exit(f"'{sdk}' is installed; the base wheel must not pull provider SDKs")
    if with_modal and importlib.util.find_spec("modal") is None:
        sys.exit("Modal extra smoke requires the Modal SDK to be installed")

    import inferweave
    from inferweave import InferWeave

    for name in inferweave.__all__:
        if getattr(inferweave, name) is None:
            sys.exit(f"Missing public export: {name}")

    if not inferweave.__version__:
        sys.exit("inferweave.__version__ is empty")
    dist_version = importlib.metadata.version("inferweave")
    if inferweave.__version__ != dist_version:
        sys.exit(
            f"inferweave.__version__ ({inferweave.__version__}) does not match dist metadata ({dist_version})"
        )
    if "site-packages" not in (inferweave.__file__ or ""):
        sys.exit(f"inferweave imported from {inferweave.__file__}, not the installed wheel")
    if wheel_path is not None and wheel_version(wheel_path) != dist_version:
        sys.exit(
            f"Installed version {dist_version} does not match wheel METADATA "
            f"Version {wheel_version(wheel_path)}"
        )

    weave = InferWeave()
    providers = weave.router.list_providers()
    if "modal" not in providers or "runpod" not in providers:
        sys.exit(f"Default providers not registered: {providers}")

    (script,) = [ep for ep in entry_points(group="console_scripts") if ep.name == "inferweave"]
    app = script.load()
    if not callable(app):
        sys.exit("inferweave console script does not resolve to a callable")

    leaked = [m for m in ("sky", "modal") if m in sys.modules]
    if leaked:
        sys.exit(f"Provider SDKs imported eagerly: {leaked}")

    # The real console script (not the in-process app) must report the same version.
    cli = shutil.which("inferweave", path=str(Path(sys.executable).parent))
    if cli is None:
        sys.exit("inferweave console script not found next to the interpreter")
    cli_run = subprocess.run([cli, "--version"], capture_output=True, text=True, check=False)
    cli_out = _ANSI.sub("", cli_run.stdout)
    if cli_run.returncode != 0 or inferweave.__version__ not in cli_out.split():
        sys.exit(
            f"`inferweave --version` output does not report {inferweave.__version__}: "
            f"rc={cli_run.returncode} stdout={cli_out!r} stderr={cli_run.stderr!r}"
        )
    print(f"Imports OK: inferweave {inferweave.__version__} from {inferweave.__file__}")


def check_typed() -> None:
    import inferweave

    package_dir = Path(inferweave.__file__ or "").parent
    if not (package_dir / "py.typed").is_file():
        sys.exit(f"{package_dir} has no py.typed; installed package is not type-discoverable")

    snippet = """from typing import assert_type
from inferweave import InferWeave, Deployment, InferenceConfig, ReferenceAudio
reveal_type(InferWeave().list_deployments())
async def inference(deployment: Deployment) -> None:
    assert_type(await deployment.synthesize("hi", references=[ReferenceAudio(b"audio", "text")]), bytes)
    assert_type(await deployment.render("fox"), list[bytes])
    assert_type(await InferWeave(inference_config=InferenceConfig()).attach("id"), Deployment)
"""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.py"
        probe.write_text(snippet, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-incremental", "--cache-dir", tmp, str(probe)],
            capture_output=True,
            text=True,
            cwd=tmp,
            check=False,
        )
    print(result.stdout)
    if result.returncode != 0:
        sys.exit(f"Installed API type check failed: {result.stderr}")
    if "py.typed" in result.stdout or "missing library stubs" in result.stdout:
        sys.exit("mypy does not treat the installed inferweave as a typed package")
    if "list[inferweave.models.deployment.Deployment]" not in result.stdout:
        sys.exit("mypy could not resolve InferWeave.list_deployments() return type")
    print("Typed OK: mypy resolves inferweave annotations from the installed wheel.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "metadata":
        check_metadata(sys.argv[2])
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "imports":
        check_imports(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "imports-modal":
        check_imports(sys.argv[2] if len(sys.argv) == 3 else None, with_modal=True)
    elif len(sys.argv) == 2 and sys.argv[1] == "typed":
        check_typed()
    else:
        sys.exit(__doc__)
