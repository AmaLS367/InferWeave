"""Base infrastructure for InferWeave runtime workers."""

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI


@dataclass
class WorkerArgs:
    """Standard parsed CLI arguments for InferWeave workers."""

    model: str
    port: int = 8000
    host: str = "0.0.0.0"
    device: str = "cuda"
    quantize_4bit: bool = False
    quantize_8bit: bool = False
    dtype: str = "bfloat16"
    mock: bool = False
    extra_args: list[str] = field(default_factory=list)


def parse_worker_args(
    args: list[str] | None = None,
    description: str = "InferWeave Model Worker",
) -> WorkerArgs:
    """Parses CLI arguments with support for custom runtime flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        dest="quantize_4bit",
        help="Enable 4-bit quantization (e.g. bitsandbytes NF4)",
    )
    parser.add_argument(
        "--quantize-8bit",
        action="store_true",
        dest="quantize_8bit",
        help="Enable 8-bit quantization",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model weights (e.g. bfloat16, float16, float32)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/simulation mode without loading weights",
    )

    known, unknown = parser.parse_known_args(args)

    return WorkerArgs(
        model=known.model,
        port=known.port,
        host=known.host,
        device=known.device,
        quantize_4bit=known.quantize_4bit,
        quantize_8bit=known.quantize_8bit,
        dtype=known.dtype,
        mock=known.mock,
        extra_args=unknown,
    )


def create_base_app(
    title: str,
    version: str = "0.1.0",
    model_id: str = "",
    lifespan: Any = None,
) -> FastAPI:
    """Instantiates a FastAPI application with standardized healthcheck routes."""
    app = FastAPI(title=title, version=version, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "model": model_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run_app(app: FastAPI, host: str, port: int) -> None:
    """Executes the Uvicorn web server."""
    uvicorn.run(app, host=host, port=port, log_level="info")
