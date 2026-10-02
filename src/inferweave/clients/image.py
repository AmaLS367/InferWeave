"""OpenAI-compatible image generation client (InferWeave's FLUX worker).

Matches ``inferweave.workers.flux``: ``POST /v1/images/generations`` with a JSON body
(``prompt``, ``n``, ``size`` as ``"<w>x<h>"``, ``response_format="b64_json"``,
``num_inference_steps``, ``guidance_scale``, ``seed``) answered by
``{"created": ..., "data": [{"b64_json": "..."}]}``.
"""

import base64
import binascii
from typing import Any

from inferweave.clients.transport import InferenceTransport
from inferweave.core.exceptions import InvalidInferenceResponseError

IMAGE_GENERATIONS_PATH = "/v1/images/generations"


class ImageGenerationClient:
    """Renders images through an OpenAI-compatible ``/v1/images/generations`` endpoint."""

    def __init__(self, transport: InferenceTransport) -> None:
        self._transport = transport

    @staticmethod
    def build_payload(
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
        n: int = 1,
        guidance_scale: float | None = None,
    ) -> dict[str, Any]:
        """Builds the request body; unset options fall back to the worker's defaults."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive integers.")
        if n < 1:
            raise ValueError("n must be at least 1.")
        if steps is not None and steps < 1:
            raise ValueError("steps must be at least 1.")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n": int(n),
            "size": f"{int(width)}x{int(height)}",
            "response_format": "b64_json",
        }
        if steps is not None:
            payload["num_inference_steps"] = int(steps)
        if seed is not None:
            payload["seed"] = int(seed)
        if guidance_scale is not None:
            payload["guidance_scale"] = float(guidance_scale)
        return payload

    async def render(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
        n: int = 1,
        *,
        guidance_scale: float | None = None,
        timeout: float | None = None,
    ) -> list[bytes]:
        """Returns ``n`` decoded images (PNG bytes for the FLUX worker)."""
        payload = self.build_payload(
            prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            n=n,
            guidance_scale=guidance_scale,
        )
        response = await self._transport.post(
            IMAGE_GENERATIONS_PATH, json=payload, timeout=timeout
        )
        return self._decode(response, n)

    def _decode(self, response: Any, expected: int) -> list[bytes]:
        dep_id = self._transport.deployment_id

        def malformed(reason: str) -> InvalidInferenceResponseError:
            return InvalidInferenceResponseError(
                f"Image endpoint returned a malformed response: {reason}.",
                deployment_id=dep_id,
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError:
            raise malformed("body is not valid JSON") from None
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            raise malformed("missing 'data' list")
        if len(data) != expected:
            raise malformed(f"expected {expected} image(s) but received {len(data)}")
        images: list[bytes] = []
        for item in data:
            encoded = item.get("b64_json") if isinstance(item, dict) else None
            if not isinstance(encoded, str) or not encoded:
                raise malformed("image entry has no 'b64_json' payload")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                raise malformed("'b64_json' is not valid base64") from None
            if not decoded:
                raise malformed("'b64_json' decoded to zero bytes")
            images.append(decoded)
        return images
