"""Fish Speech text-to-speech client.

Implements the ``POST /v1/tts`` contract of the Fish Speech release pinned in
``inferweave.runtimes.manifest`` (``v2.0.0-beta``; unchanged from v1.x). Verified against
upstream ``tools/server/views.py``, ``tools/server/api_utils.py`` and
``fish_speech/utils/schema.py``:

* The request body is a ``ServeTTSRequest`` serialized as **MessagePack**
  (``Content-Type: application/msgpack``; the server also accepts JSON, but reference audio is a
  ``bytes`` field that only MessagePack carries natively, and upstream's own client uses it).
* Numeric fields are validated with ``strict=True``: ``top_p``/``temperature``/
  ``repetition_penalty`` must be floats and ``chunk_length`` an int, so values are coerced here.
* A successful non-streaming response body is the raw encoded audio (``audio/wav`` etc.).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import msgpack  # type: ignore[import-untyped]

from inferweave.clients.transport import InferenceTransport
from inferweave.core.exceptions import InvalidInferenceResponseError

TTS_PATH = "/v1/tts"
SUPPORTED_AUDIO_FORMATS = ("wav", "pcm", "mp3")


@dataclass(frozen=True)
class ReferenceAudio:
    """An in-context voice reference: raw audio bytes plus their transcript."""

    audio: bytes
    text: str


def _is_valid_audio(data: bytes, audio_format: str) -> bool:
    if not data:
        return False
    if audio_format == "wav":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if audio_format == "mp3":
        return data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
    return True  # pcm is headerless


class FishSpeechClient:
    """Synthesizes speech through a Fish Speech ``/v1/tts`` endpoint."""

    def __init__(self, transport: InferenceTransport) -> None:
        self._transport = transport

    @staticmethod
    def build_payload(
        text: str,
        reference_id: str | None = None,
        format: str = "wav",
        references: Sequence[ReferenceAudio] = (),
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        chunk_length: int | None = None,
        max_new_tokens: int | None = None,
        normalize: bool | None = None,
    ) -> dict[str, Any]:
        """Builds the ``ServeTTSRequest`` mapping; unset options fall back to server defaults."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        if format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{format}'. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}."
            )
        payload: dict[str, Any] = {"text": text, "format": format, "streaming": False}
        if reference_id is not None:
            payload["reference_id"] = reference_id
        if references:
            payload["references"] = [
                {"audio": bytes(ref.audio), "text": ref.text} for ref in references
            ]
        if seed is not None:
            payload["seed"] = int(seed)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = float(repetition_penalty)
        if chunk_length is not None:
            payload["chunk_length"] = int(chunk_length)
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        if normalize is not None:
            payload["normalize"] = bool(normalize)
        return payload

    async def synthesize(
        self,
        text: str,
        reference_id: str | None = None,
        format: str = "wav",
        *,
        references: Sequence[ReferenceAudio] = (),
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        chunk_length: int | None = None,
        max_new_tokens: int | None = None,
        normalize: bool | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Returns the synthesized audio as raw bytes in the requested ``format``."""
        payload = self.build_payload(
            text,
            reference_id=reference_id,
            format=format,
            references=references,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_length=chunk_length,
            max_new_tokens=max_new_tokens,
            normalize=normalize,
        )
        response = await self._transport.post(
            TTS_PATH,
            content=msgpack.packb(payload, use_bin_type=True),
            headers={"Content-Type": "application/msgpack"},
            timeout=timeout,
        )
        audio = response.content
        content_type = response.headers.get("content-type", "").lower()
        if (
            "json" in content_type
            or "msgpack" in content_type
            or "text/html" in content_type
            or not _is_valid_audio(audio, format)
        ):
            raise InvalidInferenceResponseError(
                f"Fish Speech endpoint returned an invalid '{format}' audio payload "
                f"(content-type '{content_type or 'unknown'}', {len(audio)} bytes).",
                deployment_id=self._transport.deployment_id,
                status_code=response.status_code,
            )
        return audio
