"""Credential-free smoke harness using public SDK injection points, never real providers.

Run `python examples/simulate.py` from any directory with the base package installed.
The production examples need no simulation switches or imports from test helpers.
"""

import ast
import asyncio
import base64
import io
import json
import os
import re
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import msgpack

from inferweave import (
    CompositeEndpointAuth,
    Deployment,
    DeploymentState,
    DeploymentStatus,
    HealthcheckService,
    HttpxHealthcheckProbeAdapter,
    InferWeave,
    LifecycleService,
    ModalProxyAuth,
    ModelRegistry,
    ProviderType,
    SqliteDeploymentRepository,
    StaticHeaderAuth,
)
from inferweave.providers.base import ComputeProvider
from inferweave.providers.router import ProviderRouter

ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg=="
)


class SimulatedProvider(ComputeProvider):
    """Only the orchestration boundary is simulated; persistence and clients are real."""

    name = "modal"
    provider_type = ProviderType.MODAL

    def __init__(self, repository):
        self.repository = repository

    async def deploy(self, request, profile, runtime):
        assert runtime
        return Deployment(
            DeploymentStatus(
                id=f"sim-{uuid4().hex}",
                model=profile.id,
                provider=self.name,
                state=DeploymentState.STARTING,
                endpoint_url="https://simulation.modal.run",
            )
        )

    async def stop(self, deployment_id, action=None):
        assert await self.repository.get(deployment_id) is not None

    async def get_status(self, deployment_id):
        record = await self.repository.get(deployment_id)
        assert record
        return DeploymentStatus(**record.model_dump())


def response(request: httpx.Request) -> httpx.Response:
    assert request.url.host == "simulation.modal.run"
    assert request.headers["modal-key"] == "simulation-id"
    assert request.headers["modal-secret"] == "simulation-secret"
    if request.url.path in ("/v1/health", "/health"):
        return httpx.Response(200, json={"status": "ready"})
    if request.url.path == "/v1/tts":
        assert request.headers["content-type"] == "application/msgpack"
        payload = msgpack.unpackb(request.content, raw=False)
        assert (
            payload["text"] and payload["format"] == "wav" and not payload["streaming"]
        )
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        return httpx.Response(
            200, content=audio.getvalue(), headers={"content-type": "audio/wav"}
        )
    assert request.url.path == "/v1/images/generations"
    payload = json.loads(request.content)
    assert payload["size"] == "512x768"
    assert payload["num_inference_steps"] == 4 and payload["seed"] == 42
    assert payload["response_format"] == "b64_json" and payload["n"] == 2
    return httpx.Response(
        200,
        json={
            "data": [
                {"b64_json": base64.b64encode(PNG).decode()}
                for _ in range(payload["n"])
            ]
        },
    )


def simulated_weave() -> InferWeave:
    registry = ModelRegistry()
    for profile in registry.list_models():
        profile.healthcheck.initial_delay_seconds = 0
    repository = SqliteDeploymentRepository()
    router = ProviderRouter()
    router.register(SimulatedProvider(repository))
    lightning = SimulatedProvider(repository)
    lightning.name = "lightning"
    lightning.provider_type = ProviderType.LIGHTNING
    router.register(lightning)
    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    health = HealthcheckService(probe_port=HttpxHealthcheckProbeAdapter(client=client))
    auth = CompositeEndpointAuth(
        ModalProxyAuth(token_id="simulation-id", token_secret="simulation-secret"),
        StaticHeaderAuth({"Modal-Key": "simulation-id", "Modal-Secret": "simulation-secret"},
                         providers=("lightning",)),
    )
    lifecycle = LifecycleService(
        repository=repository,
        healthcheck_service=health,
        provider_resolver=router.get,
        endpoint_auth=auth,
    )
    return InferWeave(
        registry=registry,
        router=router,
        lifecycle_service=lifecycle,
        healthcheck_service=health,
        endpoint_auth=auth,
        inference_http_client=client,
    )


def validate_outputs(directory: Path) -> None:
    wavs = list(directory.glob("*.wav"))
    pngs = list(directory.glob("*.png"))
    assert wavs or len(pngs) == 2
    for path in wavs:
        with wave.open(str(path), "rb") as wav:
            assert wav.getnframes() == 240
    for path in pngs:
        assert path.read_bytes() == PNG


def run_source(source: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        previous = Path.cwd()
        try:
            os.chdir(tmp)
            with (
                patch.dict(
                    os.environ,
                    {"INFERWEAVE_DEPLOYMENTS_PATH": str(Path(tmp) / "state.db")},
                ),
                patch("inferweave.InferWeave", side_effect=simulated_weave),
            ):
                # Execute only explicit repository examples, under isolated fake boundaries.
                exec(compile(source, filename, "exec"), {"__name__": "__main__"})  # noqa: S102
            validate_outputs(Path(tmp))

            async def stopped():
                records = await SqliteDeploymentRepository(
                    Path(tmp) / "state.db"
                ).list_all()
                assert records and all(
                    r.state == DeploymentState.STOPPED for r in records
                )

            asyncio.run(stopped())
        finally:
            os.chdir(previous)


def main() -> None:
    for name in ("tts", "image", "reuse", "lightning_tts"):
        path = ROOT / "examples" / f"{name}.py"
        run_source(path.read_text(encoding="utf-8"), str(path))
        print(f"Example OK: {name}")
    # Explicitly marked blocks are executed, not merely compiled. Other Python snippets
    # are fragments for the API reference; compile them to catch syntax drift.
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        source = path.read_text(encoding="utf-8")
        for snippet in re.findall(r"```python\n(.*?)```", source, re.DOTALL):
            ast.parse(snippet, filename=str(path))
        for name, snippet in re.findall(
            r"<!-- example: (tts|image|reuse) -->\s*```python\n(.*?)```",
            source,
            re.DOTALL,
        ):
            example = (ROOT / "examples" / f"{name}.py").read_text(encoding="utf-8")
            assert snippet.strip() == example.strip(), (
                f"Stale executable example in {path}"
            )
            run_source(snippet, str(path))
        for link in re.findall(r"\]\(([^)]+)\)", source):
            if "://" in link or link.startswith("#"):
                continue
            assert (path.parent / link.split("#")[0]).exists(), (
                f"Broken link in {path}: {link}"
            )
    models_doc = (ROOT / "docs/reference/supported-models.md").read_text(
        encoding="utf-8"
    )
    for model in ModelRegistry().list_models():
        assert (
            f"| `{model.id}` | {model.workload_type.value} | `{model.default_runtime}` |"
            in models_doc
        )
        assert (
            f"| `{model.id}` | `{model.target_artifact}` | {model.hardware.min_vram_gb:g} GB | `{model.healthcheck.path}` (port {model.healthcheck.port}) |"
            in models_doc
        )
    print("Documentation OK: executable examples, Python syntax and local links")


if __name__ == "__main__":
    main()
