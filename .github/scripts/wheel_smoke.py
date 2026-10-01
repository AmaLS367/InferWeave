"""Release smoke checks for a built InferWeave wheel.

Usage:
    python wheel_smoke.py metadata <path-to-wheel>
        Verifies provider SDKs (SkyPilot, Modal) are not mandatory dependencies.

    python wheel_smoke.py imports
        Run with the interpreter of a clean environment that has ONLY the base wheel
        installed. Verifies the public import surface and CLI entry point.
"""

import sys
import zipfile
from email.parser import Parser

OPTIONAL_PROVIDER_SDKS = ("skypilot", "modal")


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

    extras = set(metadata.get_all("Provides-Extra") or [])
    expected = {"runpod", "aws", "gcp", "azure", "lambda", "nebius", "kubernetes", "clouds", "modal"}
    if missing := expected - extras:
        sys.exit(f"Missing expected extras: {sorted(missing)}")

    for extra, sdk in [(e, "skypilot") for e in sorted(expected - {"modal"})] + [("modal", "modal")]:
        if not any(r.lower().startswith(sdk) and f"extra == '{extra}'" in r for r in requires):
            sys.exit(f"Extra '{extra}' does not provide '{sdk}'")
    print("Metadata OK: SkyPilot and Modal are only available via extras.")


def check_imports() -> None:
    import importlib.util
    from importlib.metadata import entry_points

    for sdk in ("sky", "modal"):
        if importlib.util.find_spec(sdk) is not None:
            sys.exit(f"'{sdk}' is installed; the base wheel must not pull provider SDKs")

    import inferweave
    from inferweave import InferWeave

    if not inferweave.__version__:
        sys.exit("inferweave.__version__ is empty")
    if "site-packages" not in (inferweave.__file__ or ""):
        sys.exit(f"inferweave imported from {inferweave.__file__}, not the installed wheel")

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
    print(f"Imports OK: inferweave {inferweave.__version__} from {inferweave.__file__}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "metadata":
        check_metadata(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "imports":
        check_imports()
    else:
        sys.exit(__doc__)
