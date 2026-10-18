"""Unified inference client: workload validation, activity tracking, and client dispatch."""

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx

from inferweave.clients.audio import FishSpeechClient, ReferenceAudio
from inferweave.clients.image import ImageGenerationClient
from inferweave.clients.metering import (
    TransferMeasurement,
    UsageDelta,
    current_transfer,
)
from inferweave.clients.transport import InferenceTransport
from inferweave.core.exceptions import UnsupportedWorkloadError
from inferweave.models.enums import WorkloadType

logger = logging.getLogger(__name__)

ActivityCallback = Callable[[], Awaitable[None] | None]


class InferenceClient:
    """Runs generic HTTP, audio and image inference against one deployment.

    Activity semantics (the contract used by the idle/destroy policy):

    * A call that passes workload validation counts as deployment activity **when it starts**
      and again **when it finishes, whether it succeeded or failed**. A request that merely
      hits a cold or failing endpoint is still real demand, and a long render must not leave
      the deployment looking idle for its whole duration.
    * While a call is running it is also tracked as *in flight*, and a deployment with an
      in-flight request is never idle, however long the request takes. The idle timer
      restarts when the call ends.
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
        on_request_start: Callable[[], None] | None = None,
        on_request_end: Callable[[], None] | None = None,
        on_usage: Callable[[UsageDelta], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.deployment_id = deployment_id
        self.workload_type = workload_type
        self._transport = transport
        self._on_activity = on_activity
        self._on_request_start = on_request_start
        self._on_request_end = on_request_end
        self._on_usage = on_usage
        self._clock = clock
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
        measurement = TransferMeasurement()
        token = current_transfer.set(measurement)
        started_at = self._clock()
        recorded_start = False
        outcome = "failures"
        self._notify(self._on_request_start)
        try:
            recorded_start, interrupted = await self._persist_usage(UsageDelta(requests=1))
            if interrupted:
                raise asyncio.CancelledError
            await self._report_activity()
            yield
            outcome = "successes"
        except asyncio.CancelledError:
            outcome = "cancellations"
            raise
        finally:
            current_transfer.reset(token)
            self._notify(self._on_request_end)
            if recorded_start:
                delta = UsageDelta(
                    **{outcome: 1},
                    total_latency_seconds=max(0, self._clock() - started_at),
                    request_body_bytes=measurement.request_body_bytes,
                    response_body_bytes=measurement.response_body_bytes,
                    incomplete_transfers=int(measurement.incomplete),
                )
                _, interrupted = await self._persist_usage(delta)
                if interrupted:
                    raise asyncio.CancelledError
            await self._report_activity()

    async def _persist_usage(self, delta: UsageDelta) -> tuple[bool, bool]:
        """Drain storage even under repeated cancellation; never retry a committed increment."""
        task = asyncio.create_task(self._record_usage(delta))
        interrupted = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                interrupted = True
        return task.result(), interrupted

    async def _record_usage(self, delta: UsageDelta) -> bool:
        if self._on_usage is None:
            return False
        try:
            await self._on_usage(delta)
            return True
        except Exception as err:  # noqa: BLE001 - metering must preserve inference behavior
            logger.warning("Usage persistence failed (%s).", type(err).__name__)
            return False

    def _notify(self, callback: Callable[[], None] | None) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "In-flight tracking failed for deployment '%s': %s", self.deployment_id, err
            )

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

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        json: object | None = None,
        content: bytes | str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Sends a workload-independent HTTP request with lifecycle activity tracking."""
        async with self._activity():
            return await self._transport.request(
                method, path, params=params, json=json, content=content,
                headers=headers, timeout=timeout,
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
