"""FLUX model inference worker powered by Diffusers and FastAPI."""

import base64
import logging
import sys
import time
from io import BytesIO
from typing import Any

from pydantic import BaseModel, Field

from inferweave.workers.base import (
    WorkerArgs,
    create_base_app,
    parse_worker_args,
    run_app,
)

logger = logging.getLogger("inferweave.workers.flux")

# 1x1 transparent PNG base64 placeholder for simulation and test modes
_DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class GenerateImageRequest(BaseModel):
    """Direct image generation request parameters."""

    prompt: str = Field(..., description="Text prompt for image synthesis")
    width: int = Field(default=1024, description="Width of generated image in pixels")
    height: int = Field(default=1024, description="Height of generated image in pixels")
    num_inference_steps: int = Field(
        default=4, description="Number of denoising diffusion steps"
    )
    guidance_scale: float = Field(
        default=0.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    response_format: str = Field(
        default="b64_json", description="Output format: 'b64_json' or 'url'"
    )


class OpenAIImageGenerationRequest(BaseModel):
    """OpenAI API compatible image generation request."""

    prompt: str
    model: str | None = None
    n: int = 1
    quality: str = "standard"
    response_format: str = "b64_json"
    size: str = "1024x1024"
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None


class FluxWorker:
    """Encapsulates FLUX pipeline lifecycle, quantization, and generation."""

    def __init__(self, args: WorkerArgs) -> None:
        self.args = args
        self.pipe: Any = None
        self.is_loaded: bool = False
        self._check_mock_mode()

    def _check_mock_mode(self) -> None:
        """Determines whether to operate in mock mode if dependencies are unavailable."""
        if self.args.mock:
            self.is_loaded = True
            return

        try:
            import diffusers  # type: ignore # noqa: F401
            import torch  # type: ignore # noqa: F401
        except ImportError:
            logger.warning(
                "PyTorch or Diffusers not found in environment. Defaulting to mock mode."
            )
            self.args.mock = True
            self.is_loaded = True

    def load_pipeline(self) -> None:
        """Loads and prepares the FLUX diffusion pipeline on the selected device."""
        if self.args.mock:
            self.is_loaded = True
            return

        import torch  # type: ignore
        from diffusers import FluxPipeline  # type: ignore

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.args.dtype.lower(), torch.bfloat16)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }

        if self.args.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, skipping 4-bit quantization config"
                )

        logger.info(f"Loading FLUX pipeline from '{self.args.model}'...")
        self.pipe = FluxPipeline.from_pretrained(self.args.model, **load_kwargs)

        if self.args.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                self.pipe.to(self.args.device)
        else:
            self.pipe.to("cpu")

        self.is_loaded = True
        logger.info("FLUX pipeline loaded successfully.")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ) -> str:
        """Executes text-to-image synthesis and returns base64 PNG data."""
        if self.args.mock or self.pipe is None:
            return _DUMMY_PNG_B64

        import torch  # type: ignore

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.args.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


from contextlib import asynccontextmanager


def create_flux_app(worker: FluxWorker) -> Any:
    """Creates the FastAPI application for the FLUX worker."""

    @asynccontextmanager
    async def lifespan(app: Any):
        try:
            worker.load_pipeline()
        except Exception as err:  # noqa: BLE001
            logger.error(f"Failed to load pipeline on startup: {err}")
        yield

    app = create_base_app(
        title="InferWeave FLUX Worker",
        version="0.1.0",
        model_id=worker.args.model,
        lifespan=lifespan,
    )

    @app.post("/generate")
    async def generate_endpoint(req: GenerateImageRequest) -> dict[str, Any]:
        b64_data = worker.generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
        return {
            "image": b64_data,
            "format": "png",
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
        }

    @app.post("/v1/images/generations")
    async def openai_generations(req: OpenAIImageGenerationRequest) -> dict[str, Any]:
        width, height = 1024, 1024
        if req.size and "x" in req.size:
            try:
                parts = req.size.split("x")
                width, height = int(parts[0]), int(parts[1])
            except ValueError:
                pass

        steps = req.num_inference_steps or 4
        guidance = req.guidance_scale if req.guidance_scale is not None else 0.0

        images_data = []
        for _ in range(req.n):
            b64_data = worker.generate(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=req.seed,
            )
            images_data.append({"b64_json": b64_data, "revised_prompt": req.prompt})

        return {
            "created": int(time.time()),
            "data": images_data,
        }

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for FLUX worker server."""
    args = parse_worker_args(
        argv if argv is not None else sys.argv[1:], description="InferWeave FLUX Worker"
    )
    worker = FluxWorker(args)
    app = create_flux_app(worker)
    run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
