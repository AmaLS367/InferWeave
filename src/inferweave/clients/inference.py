"""Unified inference client: workload validation, activity tracking, and client dispatch."""

import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from inferweave.clients.audio import FishSpeechClient, ReferenceAudio
from inferweave.clients.image import ImageGenerationClient
from inferweave.clients.transport import InferenceTransport
from inferweave.core.exceptions import UnsupportedWorkloadError
from inferweave.models.enums import WorkloadType

logger = logging.getLogger(__name__)

ActivityCallback = Callable[[], Awaitable[None] | None]


class InferenceClient:
    """Runs audio and image inference against one deployment.

    Activity semantics (the contract used by the idle/destroy policy):

    * A call that passes workload validation counts as deployment activity **when it starts**
      and again **when it finishes, whether it succeeded or failed**. A request that merely
      hits a cold or failing endpoint is still real demand, and a long render must not leave
      the deployment looking idle for its whole duration.
    * Calls rejected up front (``UnsupportedWorkloadError``) are not activity.
    * Individual retry attempts are not counted separately; they sit inside the call.
    * Activity reporting is best-effort: a failing callback never fails an inference call.
    """

    def __init__(
        self,
        deployment_id: str,
        workload_type: WorkloadType | None,
        transport: InferenceTransport,
        on_activity: ActivityCallback | None = None,
    ) -> None:
        self.deployment_id = deployment_id
        self.workload_type = workload_type
        self._transport = transport
        self._on_activity = on_activity
        self._audio = FishSpeechClient(transport)
        self._image = ImageGenerationClient(transport)

    def _require(self, expected: WorkloadType, operation: str) -> None:
        if self.workload_type != expected:
            actual = self.workload_type.value if self.workload_type else "unknown"
            raise UnsupportedWorkloadError(
                f"{operation}() requires a '{expected.value}' deployment, but "
                f"'{self.deployment_id}' has workload type '{actual}'.",
                deployment_id=self.deployment_id,
                workload_type=actual,
                operation=operation,
            )

    async def _report_activity(self) -> None:
        if self._on_activity is None:
            return
        try:
            result = self._on_activity()
            if inspect.isawaitable(result):
                await result
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Failed to record activity for deployment '%s': %s", self.deployment_id, err
            )

    @asynccontextmanager
    async def _activity(self) -> AsyncIterator[None]:
        await self._report_activity()
        try:
            yield
        finally:
            await self._report_activity()

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
        """Synthesizes speech on an audio deployment and returns the raw audio bytes."""
        self._require(WorkloadType.AUDIO, "synthesize")
        async with self._activity():
            return await self._audio.synthesize(
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
                timeout=timeout,
            )

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
        """Renders ``n`` images on an image deployment and returns their raw bytes."""
        self._require(WorkloadType.IMAGE, "render")
        async with self._activity():
            return await self._image.render(
                prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                n=n,
                guidance_scale=guidance_scale,
                timeout=timeout,
            )

    async def aclose(self) -> None:
        """Releases the underlying HTTP connection pool."""
        await self._transport.aclose()
