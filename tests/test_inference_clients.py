"""Unified inference client: Fish Speech audio, FLUX image, retries, errors and activity."""

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import msgpack
import pytest
from support import (
    AUDIO_MODEL,
    ENDPOINT,
    IMAGE_MODEL,
    deploy_on_modal,
    make_weave,
    modal_pool,
)

from inferweave.clients import (
    FishSpeechClient,
    ImageGenerationClient,
    InferenceClient,
    InferenceConfig,
    InferenceTransport,
    ReferenceAudio,
)
from inferweave.core.exceptions import (
    EndpointNotReadyError,
    InferenceError,
    InferenceTimeoutError,
    InvalidInferenceResponseError,
    UnsupportedWorkloadError,
)
from inferweave.models.enums import DeploymentState, WorkloadType

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16
PNG_A = b"\x89PNG\r\n\x1a\n-image-a"
PNG_B = b"\x89PNG\r\n\x1a\n-image-b"


def wav_response(_: httpx.Request | None = None) -> httpx.Response:
    return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})


def b64_response(*images: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "created": 1,
            "data": [{"b64_json": base64.b64encode(i).decode()} for i in images],
        },
    )


class Recorder:
    """Scripted MockTransport handler: replays ``script`` entries and records requests."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step(request) if callable(step) else step


def make_transport(
    handler,
    *,
    config: InferenceConfig | None = None,
    auth: dict[str, str] | None = None,
    sleeps: list[float] | None = None,
    endpoint: str | None = ENDPOINT,
    random_value: float = 0.0,
) -> InferenceTransport:
    async def fake_sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return InferenceTransport(
        deployment_id="dep-1",
        endpoint_fn=lambda: endpoint,
        auth_headers_fn=(lambda: auth) if auth is not None else None,
        config=config
        or InferenceConfig(
            max_retries=3, backoff_base_seconds=1.0, backoff_max_seconds=4.0, jitter_ratio=0.0
        ),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=fake_sleep,
        random_fn=lambda: random_value,
    )


# --------------------------------------------------------------------------- audio


@pytest.mark.asyncio
async def test_fish_speech_request_uses_msgpack_contract_and_returns_raw_bytes():
    rec = Recorder(wav_response)
    client = FishSpeechClient(make_transport(rec, auth={"Modal-Key": "k-123456"}))

    audio = await client.synthesize(
        "Hello world",
        reference_id="voice-1",
        format="wav",
        references=[ReferenceAudio(audio=b"\x00\x01ref", text="reference text")],
        seed=7,
        temperature=1,
        top_p=1,
        repetition_penalty=1,
        chunk_length=250,
        max_new_tokens=512,
        normalize=False,
    )

    assert audio == WAV
    (request,) = rec.requests
    assert request.method == "POST"
    assert str(request.url) == f"{ENDPOINT}/v1/tts"
    assert request.headers["content-type"] == "application/msgpack"
    assert request.headers["modal-key"] == "k-123456"
    body = msgpack.unpackb(request.content, raw=False)
    assert body == {
        "text": "Hello world",
        "format": "wav",
        "streaming": False,
        "reference_id": "voice-1",
        "references": [{"audio": b"\x00\x01ref", "text": "reference text"}],
        "seed": 7,
        # upstream validates these with pydantic strict=True, so ints must be sent as floats
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "chunk_length": 250,
        "max_new_tokens": 512,
        "normalize": False,
    }
    assert isinstance(body["temperature"], float) and isinstance(body["top_p"], float)
    assert isinstance(body["references"][0]["audio"], bytes)


@pytest.mark.asyncio
async def test_fish_speech_minimal_request_leaves_defaults_to_server():
    rec = Recorder(wav_response)
    await FishSpeechClient(make_transport(rec)).synthesize("hi")
    body = msgpack.unpackb(rec.requests[0].content, raw=False)
    assert body == {"text": "hi", "format": "wav", "streaming": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["mp3", "pcm"])
async def test_fish_speech_other_formats(audio_format: str):
    payload = b"ID3\x04\x00" if audio_format == "mp3" else b"\x01\x02\x03\x04"
    rec = Recorder(httpx.Response(200, content=payload, headers={"content-type": "audio/mpeg"}))
    audio = await FishSpeechClient(make_transport(rec)).synthesize("hi", format=audio_format)
    assert audio == payload
    assert msgpack.unpackb(rec.requests[0].content, raw=False)["format"] == audio_format


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text": ""}, "non-empty"),
        ({"text": "   "}, "non-empty"),
        ({"text": "x", "format": "flac"}, "Unsupported audio format"),
    ],
)
async def test_fish_speech_rejects_invalid_arguments_without_http(kwargs, match):
    rec = Recorder(wav_response)
    with pytest.raises(ValueError, match=match):
        await FishSpeechClient(make_transport(rec)).synthesize(**kwargs)
    assert rec.requests == []


@pytest.mark.asyncio
async def test_retries_connection_failures_then_succeeds():
    sleeps: list[float] = []
    rec = Recorder(
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow"),
        wav_response,
    )
    audio = await FishSpeechClient(make_transport(rec, sleeps=sleeps)).synthesize("hi")
    assert audio == WAV
    assert len(rec.requests) == 3
    assert sleeps == [1.0, 2.0]  # exponential, jitter disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [502, 503])
async def test_retries_cold_start_statuses_then_succeeds(status: int):
    sleeps: list[float] = []
    rec = Recorder(httpx.Response(status, text="warming up"), wav_response)
    audio = await FishSpeechClient(make_transport(rec, sleeps=sleeps)).synthesize("hi")
    assert audio == WAV
    assert len(rec.requests) == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_retries_are_bounded_and_end_in_endpoint_not_ready():
    sleeps: list[float] = []
    rec = Recorder(httpx.Response(503, text="still cold"))
    with pytest.raises(EndpointNotReadyError) as excinfo:
        await FishSpeechClient(make_transport(rec, sleeps=sleeps)).synthesize("hi")
    assert len(rec.requests) == 4  # 1 attempt + max_retries(3)
    assert sleeps == [1.0, 2.0, 4.0]  # capped by backoff_max_seconds
    err = excinfo.value
    assert err.status_code == 503
    assert err.deployment_id == "dep-1"
    assert err.endpoint == f"{ENDPOINT}/v1/tts"
    assert err.response_body == "still cold"


@pytest.mark.asyncio
async def test_connection_failure_exhaustion_is_endpoint_not_ready():
    rec = Recorder(httpx.ConnectError("refused"))
    with pytest.raises(EndpointNotReadyError) as excinfo:
        await FishSpeechClient(make_transport(rec)).synthesize("hi")
    assert len(rec.requests) == 4
    assert excinfo.value.status_code is None


@pytest.mark.asyncio
async def test_backoff_uses_jitter_and_retry_after_within_cap():
    sleeps: list[float] = []
    config = InferenceConfig(
        max_retries=2, backoff_base_seconds=2.0, backoff_max_seconds=10.0, jitter_ratio=0.5
    )
    rec = Recorder(
        httpx.Response(503, headers={"retry-after": "7"}),
        httpx.Response(503),
        wav_response,
    )
    await FishSpeechClient(
        make_transport(rec, config=config, sleeps=sleeps, random_value=1.0)
    ).synthesize("hi")
    # attempt 0: 2.0*(1-0.5)=1.0 but Retry-After=7 wins; attempt 1: 4.0*(1-0.5)=2.0
    assert sleeps == [7.0, 2.0]
    assert all(s <= config.backoff_max_seconds for s in sleeps)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 422, 500])
async def test_permanent_errors_are_not_retried(status: int):
    rec = Recorder(httpx.Response(status, text='{"detail":"bad request"}'))
    with pytest.raises(InferenceError) as excinfo:
        await FishSpeechClient(make_transport(rec)).synthesize("hi")
    assert len(rec.requests) == 1
    assert not isinstance(excinfo.value, EndpointNotReadyError)
    assert excinfo.value.status_code == status
    assert "bad request" in (excinfo.value.response_body or "")


@pytest.mark.asyncio
async def test_read_timeout_is_not_retried_and_raises_timeout_error():
    rec = Recorder(httpx.ReadTimeout("took too long"))
    with pytest.raises(InferenceTimeoutError) as excinfo:
        await FishSpeechClient(make_transport(rec)).synthesize("hi", timeout=1.5)
    assert len(rec.requests) == 1
    assert excinfo.value.deployment_id == "dep-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"", headers={"content-type": "audio/wav"}),
        httpx.Response(200, content=b"<html>proxy error</html>", headers={"content-type": "text/html"}),
        httpx.Response(200, content=b"proxy error", headers={"content-type": "text/plain"}),
        httpx.Response(200, json={"error": "nope"}),
        httpx.Response(200, content=b"not a wav file at all", headers={"content-type": "audio/wav"}),
    ],
)
async def test_malformed_audio_response_raises(response: httpx.Response):
    rec = Recorder(response)
    with pytest.raises(InvalidInferenceResponseError):
        await FishSpeechClient(make_transport(rec)).synthesize("hi")


@pytest.mark.asyncio
async def test_missing_endpoint_raises_endpoint_not_ready_without_http():
    rec = Recorder(wav_response)
    with pytest.raises(EndpointNotReadyError, match="no endpoint"):
        await FishSpeechClient(make_transport(rec, endpoint=None)).synthesize("hi")
    assert rec.requests == []


@pytest.mark.asyncio
async def test_endpoint_without_scheme_or_port_uses_http_and_default_port():
    rec = Recorder(wav_response)
    transport = InferenceTransport(
        deployment_id="dep-1",
        endpoint_fn=lambda: "203.0.113.5",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec)),
        default_port=8080,
    )
    await FishSpeechClient(transport).synthesize("hi")
    assert str(rec.requests[0].url) == "http://203.0.113.5:8080/v1/tts"


@pytest.mark.asyncio
async def test_same_origin_303_redirect_is_followed_with_get_and_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(303, headers={"location": "/__result/abc"})
        assert request.url.path == "/__result/abc"
        return wav_response()

    rec = Recorder(handler)
    audio = await FishSpeechClient(
        make_transport(rec, auth={"Modal-Key": "k-123456"})
    ).synthesize("hi")
    assert audio == WAV
    assert [r.method for r in rec.requests] == ["POST", "GET"]
    assert rec.requests[1].headers["modal-key"] == "k-123456"


@pytest.mark.asyncio
async def test_cross_origin_redirect_is_refused_and_credentials_not_forwarded():
    rec = Recorder(httpx.Response(307, headers={"location": "https://evil.example/steal"}))
    with pytest.raises(InferenceError, match="cross-origin"):
        await FishSpeechClient(
            make_transport(rec, auth={"Modal-Secret": "s-supersecret"})
        ).synthesize("hi")
    assert len(rec.requests) == 1


# --------------------------------------------------------------------------- image


@pytest.mark.asyncio
async def test_flux_render_maps_parameters_and_decodes_base64():
    rec = Recorder(b64_response(PNG_A))
    images = await ImageGenerationClient(make_transport(rec)).render(
        "a red fox", width=512, height=768, steps=8, seed=42, n=1, guidance_scale=3.5
    )
    assert images == [PNG_A]
    (request,) = rec.requests
    assert request.method == "POST"
    assert str(request.url) == f"{ENDPOINT}/v1/images/generations"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "prompt": "a red fox",
        "n": 1,
        "size": "512x768",
        "response_format": "b64_json",
        "num_inference_steps": 8,
        "seed": 42,
        "guidance_scale": 3.5,
    }


@pytest.mark.asyncio
async def test_flux_render_defaults_omit_optional_fields_and_support_multiple_images():
    rec = Recorder(b64_response(PNG_A, PNG_B))
    images = await ImageGenerationClient(make_transport(rec)).render("two foxes", n=2)
    assert images == [PNG_A, PNG_B]
    assert json.loads(rec.requests[0].content) == {
        "prompt": "two foxes",
        "n": 2,
        "size": "1024x1024",
        "response_format": "b64_json",
    }


@pytest.mark.asyncio
async def test_flux_render_retries_cold_start():
    sleeps: list[float] = []
    rec = Recorder(httpx.ConnectError("refused"), httpx.Response(503), b64_response(PNG_A))
    images = await ImageGenerationClient(make_transport(rec, sleeps=sleeps)).render("fox")
    assert images == [PNG_A]
    assert len(rec.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["list"]),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"url": "https://x/y.png"}]}),
        httpx.Response(200, json={"data": [{"b64_json": "!!!not-base64!!!"}]}),
        httpx.Response(200, json={"data": [{"b64_json": ""}]}),
        b64_response(PNG_A, PNG_B),  # two returned for n=1
    ],
)
async def test_flux_malformed_payloads_raise(response: httpx.Response):
    rec = Recorder(response)
    with pytest.raises(InvalidInferenceResponseError):
        await ImageGenerationClient(make_transport(rec)).render("fox", n=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"prompt": ""}, {"prompt": "x", "width": 0}, {"prompt": "x", "n": 0}, {"prompt": "x", "steps": 0}],
)
async def test_flux_rejects_invalid_arguments_without_http(kwargs):
    rec = Recorder(b64_response(PNG_A))
    with pytest.raises(ValueError):
        await ImageGenerationClient(make_transport(rec)).render(**kwargs)
    assert rec.requests == []


# --------------------------------------------------------------- workload safety


def _client(workload: WorkloadType, rec: Recorder, **kwargs) -> InferenceClient:
    return InferenceClient("dep-1", workload, make_transport(rec), **kwargs)


@pytest.mark.asyncio
async def test_synthesize_on_image_workload_is_unsupported_and_sends_nothing():
    rec = Recorder(wav_response)
    activity: list[str] = []
    client = _client(WorkloadType.IMAGE, rec, on_activity=lambda: activity.append("x"))
    with pytest.raises(UnsupportedWorkloadError) as excinfo:
        await client.synthesize("hi")
    assert excinfo.value.operation == "synthesize"
    assert excinfo.value.workload_type == "image"
    assert excinfo.value.deployment_id == "dep-1"
    assert rec.requests == [] and activity == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workload", [WorkloadType.AUDIO, WorkloadType.LLM, None])
async def test_render_on_non_image_workload_is_unsupported(workload):
    rec = Recorder(b64_response(PNG_A))
    with pytest.raises(UnsupportedWorkloadError):
        await _client(workload, rec).render("fox")
    assert rec.requests == []


@pytest.mark.asyncio
async def test_activity_is_reported_at_start_and_end_even_on_failure():
    events: list[str] = []
    rec = Recorder(lambda _: events.append("http") or httpx.Response(422, text="bad"))
    client = _client(WorkloadType.AUDIO, rec, on_activity=lambda: events.append("activity"))
    with pytest.raises(InferenceError):
        await client.synthesize("hi")
    assert events == ["activity", "http", "activity"]


@pytest.mark.asyncio
async def test_failing_activity_callback_never_breaks_inference():
    async def broken() -> None:
        raise RuntimeError("repository offline")

    client = _client(WorkloadType.AUDIO, Recorder(wav_response), on_activity=broken)
    assert await client.synthesize("hi") == WAV


# ------------------------------------------- end-to-end via Deployment handles


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


def _router(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/tts":
        return wav_response()
    if request.url.path == "/v1/images/generations":
        return b64_response(PNG_A)
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_deployment_methods_dispatch_by_workload_after_attach(db_path: Path):
    weave_a = make_weave(db_path)
    audio_dep = await deploy_on_modal(weave_a, model=AUDIO_MODEL)
    image_dep = await deploy_on_modal(weave_a, model=IMAGE_MODEL)

    weave_b = make_weave(db_path, handler=_router)
    audio = await weave_b.attach(audio_dep.id)
    image = await weave_b.attach(image_dep.id)

    assert await audio.synthesize("hello") == WAV
    assert await image.render("fox", width=256, height=256) == [PNG_A]
    with pytest.raises(UnsupportedWorkloadError):
        await image.synthesize("hello")
    with pytest.raises(UnsupportedWorkloadError):
        await audio.render("fox")
    await weave_b.close()


@pytest.mark.asyncio
async def test_inference_uses_the_owning_accounts_proxy_tokens(db_path: Path):
    """Pooled deployments send their own account's proxy tokens, never the env/ambient ones."""
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return wav_response()

    weave_a = make_weave(db_path, accounts=modal_pool("a", "b"))
    pooled = await deploy_on_modal(weave_a, account="b")
    ambient = await deploy_on_modal(make_weave(db_path))  # same DB, ambient account

    weave_b = make_weave(db_path, handler=handler, accounts=modal_pool("a", "b"))
    assert await (await weave_b.attach(pooled.id)).synthesize("hello") == WAV
    assert seen[-1]["Modal-Key"] == "SENTINEL-modal-b-proxy-id"
    assert seen[-1]["Modal-Secret"] == "SENTINEL-modal-b-proxy-secret"

    assert await (await weave_b.attach(ambient.id)).synthesize("hello") == WAV
    assert seen[-1]["Modal-Key"] == "wk-test-proxy-id-0000"  # conftest ambient env tokens
    await weave_b.close()


@pytest.mark.asyncio
async def test_inference_counts_as_activity_and_persists_across_restarts(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=60)
    stale = datetime.now(UTC) - timedelta(minutes=59)
    await weave_a.lifecycle_service.touch_activity(dep.id, now=stale)

    weave_b = make_weave(db_path, handler=_router)
    attached = await weave_b.attach(dep.id)
    assert attached.last_activity_at == stale
    before = datetime.now(UTC)
    await attached.synthesize("hello")
    assert attached.last_activity_at is not None
    assert attached.last_activity_at >= before
    assert attached.is_idle() is False

    # a third process still sees the reset timer: restart does not lose the activity
    weave_c = make_weave(db_path)
    recovered = await weave_c.attach(dep.id)
    assert recovered.last_activity_at is not None
    assert recovered.last_activity_at >= before


@pytest.mark.asyncio
async def test_inference_on_stopped_handle_raises_endpoint_not_ready(db_path: Path):
    weave = make_weave(db_path, handler=_router)
    dep = await deploy_on_modal(weave)
    dep._status.state = DeploymentState.STOPPED
    with pytest.raises(EndpointNotReadyError, match="stopped"):
        await dep.synthesize("hello")


@pytest.mark.asyncio
async def test_dry_run_deployment_has_no_inference_client(db_path: Path):
    weave = make_weave(db_path, handler=_router)
    dep = await weave.deploy(model=AUDIO_MODEL, provider="modal", dry_run=True)
    with pytest.raises(InferenceError, match="dry-run"):
        await dep.synthesize("hello")


@pytest.mark.asyncio
async def test_cold_start_through_deployment_handle_retries_with_configured_policy(
    db_path: Path,
):
    attempts = {"n": 0}

    def cold_then_warm(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="booting") if attempts["n"] < 3 else wav_response()

    weave = make_weave(db_path, handler=cold_then_warm)
    dep = await deploy_on_modal(weave)
    assert await dep.synthesize("hello") == WAV
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_single_byte_mp3_response_is_invalid_not_index_error():
    rec = Recorder(httpx.Response(200, content=b"\xff", headers={"content-type": "audio/mpeg"}))
    with pytest.raises(InvalidInferenceResponseError):
        await FishSpeechClient(make_transport(rec)).synthesize("hi", format="mp3")


@pytest.mark.asyncio
async def test_in_flight_request_is_never_idle_then_timer_restarts_after_it(db_path: Path):
    seen: dict[str, object] = {}
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=10)
    lifecycle = weave_a.lifecycle_service

    async def slow_render(request: httpx.Request) -> httpx.Response:
        # The request outlives the whole 10 minute destroy window while still running.
        later = datetime.now(UTC) + timedelta(hours=1)
        seen["idle_during"] = lifecycle.is_idle(dep.id, now=later)
        seen["stopped_during"] = await lifecycle.check_and_autostop(dep.id, now=later)
        return wav_response()

    weave = make_weave(db_path, handler=slow_render)
    handle = await weave.attach(dep.id)
    lifecycle = weave.lifecycle_service
    assert await handle.synthesize("hello") == WAV

    assert seen == {"idle_during": False, "stopped_during": False}
    state = lifecycle.get_state(dep.id)
    assert state is not None and state.in_flight_requests == 0
    # Once the call has ended the destroy timer runs again, counted from the call's end.
    assert lifecycle.is_idle(dep.id, now=datetime.now(UTC) + timedelta(hours=1)) is True
    await weave.close()


@pytest.mark.asyncio
async def test_failed_request_releases_in_flight_marker(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=10)

    weave = make_weave(db_path, handler=lambda _: httpx.Response(422, text="bad"))
    handle = await weave.attach(dep.id)
    with pytest.raises(InferenceError):
        await handle.synthesize("hello")
    state = weave.lifecycle_service.get_state(dep.id)
    assert state is not None and state.in_flight_requests == 0
    await weave.close()
