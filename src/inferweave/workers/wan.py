"""WAN 2.1 Video model inference worker powered by Diffusers and FastAPI."""

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

logger = logging.getLogger("inferweave.workers.wan")

# Dummy MP4 base64 placeholder for simulation and test modes
_DUMMY_MP4_B64 = (
    "AAAAHGZ0eXBtcDQyAAAAAG1wNDJpc29tYXZjMQAAADhtb292AAAAbG12aGQAAAAA"
    "AAAAAAAAAAAAAAACAAAAAAABAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)


class GenerateVideoRequest(BaseModel):
    """Direct video generation request parameters."""

    prompt: str = Field(..., description="Text prompt for video generation")
    width: int = Field(default=832, description="Width of video in pixels")
    height: int = Field(default=480, description="Height of video in pixels")
    num_frames: int = Field(default=16, description="Total number of video frames")
    fps: int = Field(default=16, description="Frames per second")
    num_inference_steps: int = Field(
        default=30, description="Number of diffusion denoising steps"
    )
    guidance_scale: float = Field(
        default=5.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


class OpenAIVideoGenerationRequest(BaseModel):
    """Structured video generation request payload."""

    prompt: str
    model: str | None = None
    size: str = "832x480"
    num_frames: int = 16
    fps: int = 16
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None


class WanWorker:
    """Encapsulates WAN Video pipeline lifecycle, quantization, and video synthesis."""

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
        """Loads and prepares the WAN video diffusion pipeline on the target device."""
        if self.args.mock:
            self.is_loaded = True
            return

        import torch  # type: ignore
        from diffusers import AutoPipelineForText2Video  # type: ignore

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

        logger.info(f"Loading WAN pipeline from '{self.args.model}'...")
        try:
            from diffusers import WanPipeline  # type: ignore

            self.pipe = WanPipeline.from_pretrained(self.args.model, **load_kwargs)
        except (ImportError, AttributeError):
            self.pipe = AutoPipelineForText2Video.from_pretrained(
                self.args.model, **load_kwargs
            )

        if self.args.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                self.pipe.to(self.args.device)
        else:
            self.pipe.to("cpu")

        self.is_loaded = True
        logger.info("WAN video pipeline loaded successfully.")

    def generate(
        self,
        prompt: str,
        width: int = 832,
        height: int = 480,
        num_frames: int = 16,
        fps: int = 16,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int | None = None,
    ) -> str:
        """Synthesizes video frames from text prompt and returns base64 MP4."""
        if self.args.mock or self.pipe is None:
            return _DUMMY_MP4_B64

        import torch  # type: ignore
        from diffusers.utils import export_to_video  # type: ignore

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.args.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        frames = output.frames[0]
        buffer = BytesIO()
        export_to_video(frames, buffer, fps=fps)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


from contextlib import asynccontextmanager


def create_wan_app(worker: WanWorker) -> Any:
    """Creates the FastAPI application for the WAN worker."""

    @asynccontextmanager
    async def lifespan(app: Any):
        try:
            worker.load_pipeline()
        except Exception as err:  # noqa: BLE001
            logger.error(f"Failed to load WAN pipeline on startup: {err}")
        yield

    app = create_base_app(
        title="InferWeave WAN Video Worker",
        version="0.1.0",
        model_id=worker.args.model,
        lifespan=lifespan,
    )

    @app.post("/generate")
    async def generate_endpoint(req: GenerateVideoRequest) -> dict[str, Any]:
        b64_data = worker.generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
        return {
            "video": b64_data,
            "format": "mp4",
            "prompt": req.prompt,
            "num_frames": req.num_frames,
            "fps": req.fps,
        }

    @app.post("/v1/videos/generations")
    async def openai_video_generations(
        req: OpenAIVideoGenerationRequest,
    ) -> dict[str, Any]:
        width, height = 832, 480
        if req.size and "x" in req.size:
            try:
                parts = req.size.split("x")
                width, height = int(parts[0]), int(parts[1])
            except ValueError:
                pass

        steps = req.num_inference_steps or 30
        guidance = req.guidance_scale if req.guidance_scale is not None else 5.0

        b64_data = worker.generate(
            prompt=req.prompt,
            width=width,
            height=height,
            num_frames=req.num_frames,
            fps=req.fps,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=req.seed,
        )

        return {
            "created": int(time.time()),
            "data": [
                {
                    "b64_json": b64_data,
                    "format": "mp4",
                    "revised_prompt": req.prompt,
                }
            ],
        }

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for WAN video worker server."""
    args = parse_worker_args(
        argv if argv is not None else sys.argv[1:],
        description="InferWeave WAN Video Worker",
    )
    worker = WanWorker(args)
    app = create_wan_app(worker)
    run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
