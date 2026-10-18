"""Generic HTTP inference: real SDK wiring/recovery with all HTTP/cloud edges mocked."""

import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from support import ENDPOINT, deploy_on_modal, make_weave, modal_pool, patched_modal

from inferweave.clients import InferenceConfig, InferenceTransport
from inferweave.core.exceptions import (
    AccountUnavailableError,
    EndpointNotReadyError,
    InferenceError,
    InferenceTimeoutError,
    ModelNotFoundError,
    UnsupportedWorkloadError,
)
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import ModelProfile


@pytest_asyncio.fixture
async def weave_factory(tmp_path):
    weaves = []

    def make(**kwargs):
        weave = make_weave(tmp_path / "deployments.db", **kwargs)
        weaves.append(weave)
        return weave

    yield make
    for weave in weaves:
        await weave.close()
        if weave._inference_http_client is not None:
            await weave._inference_http_client.aclose()


async def custom_deployment(weave, **kwargs):
    profile = ModelProfile.for_container(
        id="custom-http", image="example.test/custom:v1", args=["/app/serve"], port=8090,
    )
    weave.register_model(profile)
    return await deploy_on_modal(weave, model=profile.id, **kwargs)


@pytest_asyncio.fixture
async def transport_factory():
    clients = []

    def make(handler, *, endpoint=ENDPOINT, auth=None, client_auth=None, **client_options):
        requests, sleeps = [], []

        async def record(request):
            requests.append(request)
            result = handler(request)
            return await result if inspect.isawaitable(result) else result

        async def sleep(delay):
            sleeps.append(delay)

        client = httpx.AsyncClient(transport=httpx.MockTransport(record), auth=client_auth, **client_options)
        clients.append(client)
        transport = InferenceTransport(
            "dep-http", endpoint_fn=lambda: endpoint,
            auth_headers_fn=auth if callable(auth) else lambda: auth or {},
            config=InferenceConfig(
                max_retries=2, backoff_base_seconds=1, jitter_ratio=0,
            ),
            http_client=client, default_port=8090, sleep=sleep,
        )
        return transport, requests, sleeps

    yield make
    for client in clients:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["deploy", "attach", "find"])
async def test_custom_container_get_and_post_through_sdk_and_recovery(weave_factory, recovery):
    seen = []

    def handler(request):
        seen.append(request)
        state = weave.lifecycle_service.get_state(handle.id)
        assert state.in_flight_requests == 1
        future = datetime.now(UTC) + timedelta(hours=1)
        assert not weave.lifecycle_service.is_idle(handle.id, now=future)
        if request.method == "GET":
            return httpx.Response(200, json={"available": True}, headers={"x-version": "v1"})
        return httpx.Response(201, content=b"\x00\xff\x80result", headers={
            "content-type": "application/octet-stream", "x-job": "123",
        })

    original = weave_factory(handler=handler, accounts=modal_pool("a", "b"))
    handle = await custom_deployment(original, account="b", destroy_after_idle_mins=10)
    weave = original
    if recovery != "deploy":
        deployment_id = handle.id
        await original.close()
        weave = weave_factory(handler=handler, accounts=modal_pool("a", "b"))
        with pytest.raises(ModelNotFoundError):
            weave.registry.get("custom-http")
        with patched_modal() as cloud:
            handle = (
                await weave.attach(deployment_id) if recovery == "attach"
                else await weave.find(model="custom-http", provider="modal")
            )
            cloud.lookup_app.assert_not_called()
            cloud.stop_app.assert_not_called()
            cloud.from_credentials.assert_not_called()
    assert handle is not None and handle.workload_type == WorkloadType.CUSTOM
    await weave.lifecycle_service.touch_activity(
        handle.id, now=datetime.now(UTC) - timedelta(minutes=9),
    )
    before = datetime.now(UTC)
    response = await handle.request(
        "get", "api/status", params=[("tag", "a"), ("tag", "b"), ("limit", 2)],
        headers={"X-Request-ID": "custom", "mOdAl-kEy": "wrong", "MODAL-SECRET": "wrong"},
    )
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200 and response.json() == {"available": True}
    assert response.headers["x-version"] == "v1"
    assert response.request.url.path == "/api/status"
    assert response.request.url.params.multi_items() == [("tag", "a"), ("tag", "b"), ("limit", "2")]
    result = await handle.request("POST", "/api/predict", json={"inputs": [1, 2]}, timeout=12.5)
    assert result.status_code == 201 and result.content == b"\x00\xff\x80result"
    assert result.headers["x-job"] == "123"
    assert json.loads(seen[-1].content) == {"inputs": [1, 2]}
    assert seen[-1].extensions["timeout"] == {"connect": 10.0, "read": 12.5, "write": 12.5, "pool": 12.5}
    for request in seen:
        assert request.headers.get_list("modal-key") == ["SENTINEL-modal-b-proxy-id"]
        assert request.headers.get_list("modal-secret") == ["SENTINEL-modal-b-proxy-secret"]
    assert seen[0].headers["x-request-id"] == "custom"
    assert handle.last_activity_at >= before
    assert weave.lifecycle_service.get_state(handle.id).in_flight_requests == 0
    restarted = weave_factory(accounts=modal_pool("a", "b"))
    assert (await restarted.attach(handle.id)).last_activity_at >= before
    with pytest.raises(UnsupportedWorkloadError):
        await handle.synthesize("unsupported")
    with pytest.raises(UnsupportedWorkloadError):
        await handle.render("unsupported")


@pytest.mark.asyncio
async def test_generic_requests_keep_accounts_separate_and_refresh_proxy_tokens(weave_factory):
    seen, env = [], {}
    accounts = modal_pool("a", "b", environ=env)
    weave = weave_factory(
        accounts=accounts, handler=lambda r: seen.append(r) or httpx.Response(204),
    )
    first = await custom_deployment(weave, account="a")
    second = await custom_deployment(weave, account="b")
    await first.request("GET", "/")
    await second.request("GET", "/")
    env["T_B_PSECRET"] = "SENTINEL-rotated-proxy-secret"
    await second.request("GET", "/")
    assert [r.headers["modal-secret"] for r in seen] == [
        "SENTINEL-modal-a-proxy-secret", "SENTINEL-modal-b-proxy-secret",
        "SENTINEL-rotated-proxy-secret",
    ]
    del env["T_B_ID"]
    with pytest.raises(AccountUnavailableError):
        await second.request("GET", "/")
    assert len(seen) == 3
    assert weave.lifecycle_service.get_state(second.id).in_flight_requests == 0
    removed = weave_factory(accounts=modal_pool("a"))
    with pytest.raises(AccountUnavailableError):
        await removed.attach(second.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"])
async def test_standard_methods_and_raw_bodies(transport_factory, method):
    transport, requests, _ = transport_factory(lambda _: httpx.Response(204))
    response = await transport.request(
        method, "/v1/raw?kind=binary", content=b"\x00\xffpayload",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 204
    assert requests[0].method == method
    assert requests[0].content == b"\x00\xffpayload"
    assert requests[0].url.params["kind"] == "binary"
    assert requests[0].headers["content-type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_text_body_and_endpoint_prefix_and_default_port(transport_factory):
    transport, requests, _ = transport_factory(
        lambda _: httpx.Response(200), endpoint="203.0.113.5/prefix/",
    )
    await transport.request("POST", "echo", content="hello λ")
    assert str(requests[0].url) == "http://203.0.113.5:8090/prefix/echo"
    assert requests[0].content.decode() == "hello λ"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "https://evil.example/steal", f"{ENDPOINT}/same-origin", "//evil.example/path",
    "///evil.example/path", "https:evil.example", "ftp://evil.example/path",
    "\\\\evil.example\\path", "/\\evil.example/path", " https://evil.example/path",
    "\n//evil.example", "/safe\r\nHost: evil.example", "/api#fragment", "/api#",
])
async def test_unsafe_api_paths_never_send_auth(transport_factory, path):
    transport, requests, _ = transport_factory(
        lambda _: httpx.Response(200), auth={"Modal-Secret": "SENTINEL-provider-secret"},
    )
    with pytest.raises(ValueError):
        await transport.request("GET", path)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("location", [
    "https://evil.example/steal", "//evil.example/steal", f"{ENDPOINT}:444/steal",
    ENDPOINT.replace("https:", "http:") + "/steal",
    ENDPOINT.replace("https://", "https://user:password@") + "/steal",
    "/\\evil.example/steal", "ftp://evil.example/steal", "https://[invalid",
    "/result\t/unsafe",
])
async def test_unsafe_redirects_with_redirect_enabled_client_never_forward_auth(transport_factory, location, method):
    transport, requests, _ = transport_factory(
        lambda _: httpx.Response(307, headers={"location": location}),
        auth={"Modal-Secret": "SENTINEL-provider-secret"}, follow_redirects=True,
    )
    with pytest.raises(InferenceError) as error:
        await transport.request(method, "/start", json={"input": 1})
    assert len(requests) == 1
    assert "SENTINEL-provider-secret" not in str(error.value)
    assert error.value.endpoint == f"{ENDPOINT}/start"
    # httpx rejects control characters before returning the response/status to us.
    assert error.value.status_code == (None if "\t" in location else 307)


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "status", "redirect_method"), [
    ("POST", 301, "GET"), ("POST", 302, "GET"), ("POST", 303, "GET"),
    ("POST", 307, "POST"), ("PATCH", 308, "PATCH"), ("PUT", 302, "PUT"),
    ("HEAD", 303, "HEAD"), ("DELETE", 303, "GET"),
])
async def test_same_origin_redirect_method_body_query_and_auth(
    transport_factory, method, status, redirect_method,
):
    def handler(request):
        return (
            httpx.Response(status, headers={"location": "/result?job=2"})
            if request.url.path == "/start" else httpx.Response(200, content=b"done")
        )

    transport, requests, _ = transport_factory(handler, auth={"X-Provider-Key": "SENTINEL-key"})
    assert (await transport.request(
        method, "/start", params={"initial": "1"}, content=b"body",
        headers={"content-type": "application/octet-stream"},
    )).content == b"done"
    assert [r.method for r in requests] == [method, redirect_method]
    assert dict(requests[1].url.params) == {"job": "2"}
    assert requests[1].headers["x-provider-key"] == "SENTINEL-key"
    if method != redirect_method:
        assert requests[1].content == b""
        assert "content-type" not in requests[1].headers
        assert "content-length" not in requests[1].headers
    else:
        assert requests[1].content == b"body"


@pytest.mark.asyncio
async def test_same_origin_redirect_uses_canonical_origin_and_is_bounded(transport_factory):
    transport, requests, _ = transport_factory(
        lambda _: httpx.Response(302, headers={"location": ENDPOINT.upper() + ":443/loop"}),
    )
    with pytest.raises(InferenceError, match="Too many redirects"):
        await transport.request("GET", "/start")
    assert len(requests) == 6


@pytest.mark.asyncio
async def test_case_insensitive_mandatory_auth_wins_over_headers_and_client_auth(transport_factory):
    transport, requests, _ = transport_factory(
        lambda _: httpx.Response(200),
        auth={"Authorization": "Bearer SENTINEL-owner", "Modal-Key": "SENTINEL-owner-key"},
        client_auth=httpx.BasicAuth("other", "account"),
        headers={"Host": "evil.example", "Authorization": "Bearer SENTINEL-client"},
    )
    await transport.request("GET", "/", headers={
        "authorization": "Bearer SENTINEL-wrong", "MODAL-KEY": "wrong",
        "Modal-Key": "also-wrong", "x-extra": "kept",
    })
    assert requests[0].headers.get_list("authorization") == ["Bearer SENTINEL-owner"]
    assert requests[0].headers.get_list("modal-key") == ["SENTINEL-owner-key"]
    assert requests[0].headers["x-extra"] == "kept"
    assert requests[0].headers["host"] == httpx.URL(ENDPOINT).host


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE", "POST", "PATCH"])
@pytest.mark.parametrize("failure", ["connect", "connect-timeout", "protocol", 502, 503])
async def test_retries_are_method_appropriate_and_bounded(transport_factory, method, failure):
    def handler(_):
        if failure == "connect":
            raise httpx.ConnectError("SENTINEL-internal")
        if failure == "connect-timeout":
            raise httpx.ConnectTimeout("SENTINEL-internal")
        if failure == "protocol":
            raise httpx.RemoteProtocolError("SENTINEL-internal")
        return httpx.Response(failure, text="cold")

    transport, requests, sleeps = transport_factory(handler)
    with pytest.raises(EndpointNotReadyError) as error:
        await transport.request(method, "/infer", json={"input": 1})
    idempotent = method not in {"POST", "PATCH"}
    assert len(requests) == (3 if idempotent else 1)
    assert sleeps == ([1.0, 2.0] if idempotent else [])
    assert "SENTINEL-internal" not in str(error.value)
    assert error.value.status_code == (failure if isinstance(failure, int) else None)


@pytest.mark.asyncio
async def test_get_retries_with_refreshed_auth_and_preserved_query(transport_factory):
    auth = {"X-Provider-Key": "SENTINEL-first"}

    def handler(request):
        if request.headers["x-provider-key"] == "SENTINEL-first":
            auth["X-Provider-Key"] = "SENTINEL-rotated"
            return httpx.Response(503, headers={"retry-after": "3"})
        return httpx.Response(200, json=[1, 2])

    transport, requests, sleeps = transport_factory(handler, auth=lambda: auth)
    assert (await transport.request("GET", "/api", params={"q": "hello"})).json() == [1, 2]
    assert len(requests) == 2 and sleeps == [3.0]
    assert requests[0].url == requests[1].url


@pytest.mark.asyncio
async def test_post_redirect_result_failure_does_not_replay_original_post(transport_factory):
    def handler(request):
        return (
            httpx.Response(303, headers={"location": "/result"})
            if request.method == "POST" else httpx.Response(503)
        )

    transport, requests, sleeps = transport_factory(handler)
    with pytest.raises(EndpointNotReadyError):
        await transport.request("POST", "/generate", json={"input": 1})
    assert [r.method for r in requests] == ["POST", "GET"]
    assert sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_error", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_response_timeouts_never_retry(transport_factory, timeout_error, method):
    def handler(_):
        raise timeout_error("SENTINEL-sensitive-error")

    transport, requests, sleeps = transport_factory(handler)
    with pytest.raises(InferenceTimeoutError) as error:
        await transport.request(method, "/slow?token=SENTINEL-query", timeout=1.5)
    assert len(requests) == 1 and sleeps == []
    assert error.value.deployment_id == "dep-http"
    assert "SENTINEL" not in str(error.value)
    assert error.value.endpoint == f"{ENDPOINT}/slow"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 429, 500])
async def test_http_errors_are_typed_and_redact_caller_and_provider_headers(transport_factory, status, caplog):
    provider_secret, caller_secret = "SENTINEL-provider", "SENTINEL-caller"
    transport, requests, sleeps = transport_factory(
        lambda _: httpx.Response(status, text=f"{provider_secret} {caller_secret}"),
        auth={"Modal-Secret": provider_secret},
    )
    with pytest.raises(InferenceError) as error:
        await transport.request("GET", "/api?key=SENTINEL-query", headers={"X-API-Key": caller_secret})
    assert len(requests) == 1 and sleeps == []
    assert error.value.status_code == status
    visible = f"{error.value} {error.value.response_body} {error.value.endpoint} {caplog.text}"
    assert "SENTINEL" not in visible
    assert error.value.response_body == "[REDACTED] [REDACTED]"


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["SENTINEL-bearer", "SENTINEL-" + "x" * 1500])
async def test_error_redaction_covers_bare_bearer_and_tokens_crossing_snippet_boundary(
    transport_factory, token,
):
    transport, _, _ = transport_factory(
        lambda _: httpx.Response(400, text=f"Bearer {token}; bare={token}"),
        auth={"Authorization": f"Bearer {token}"},
    )
    with pytest.raises(InferenceError) as error:
        await transport.request("GET", "/api")
    assert error.value.response_body == "[REDACTED]; bare=[REDACTED]"


@pytest.mark.asyncio
async def test_overlapping_header_secrets_are_fully_redacted(transport_factory):
    transport, _, _ = transport_factory(
        lambda _: httpx.Response(400, text="SENTINEL-full-secret"),
        auth={"X-Token-Prefix": "SENTINEL", "X-API-Key": "SENTINEL-full-secret"},
    )
    with pytest.raises(InferenceError) as error:
        await transport.request("GET", "/api")
    assert error.value.response_body == "[REDACTED]"


@pytest.mark.asyncio
async def test_legacy_post_keeps_json_precedence(transport_factory):
    transport, requests, _ = transport_factory(lambda _: httpx.Response(200))
    await transport.post("/api", json={"input": 1}, content=b"ignored")
    assert json.loads(requests[0].content) == {"input": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"method": "CONNECT"}, {"method": "GET\r\nPOST"},
    {"json": {}, "content": b"body"}, {"headers": {"hOsT": "evil.example"}},
    {"timeout": 0}, {"timeout": -1}, {"timeout": float("nan")}, {"timeout": float("inf")},
])
async def test_invalid_arguments_send_nothing(transport_factory, kwargs):
    transport, requests, _ = transport_factory(lambda _: httpx.Response(200))
    arguments = {"method": "POST", "path": "/api", **kwargs}
    with pytest.raises(ValueError):
        await transport.request(**arguments)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["stopped", "failed", "pending", "provisioning", "missing", "dry-run"])
async def test_unavailable_deployments_never_send_requests(weave_factory, unavailable):
    seen = []
    weave = weave_factory(handler=lambda r: seen.append(r) or httpx.Response(200))
    handle = await custom_deployment(weave, dry_run=unavailable == "dry-run")
    if unavailable == "missing":
        handle._status.endpoint_url = None
    elif unavailable != "dry-run":
        handle._status.state = DeploymentState(unavailable)
    with pytest.raises(InferenceError):
        await handle.request("GET", "/api")
    assert seen == []
    state = weave.lifecycle_service.get_state(handle.id)
    assert state is None or state.in_flight_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "timeout", "cancel"])
async def test_failed_and_cancelled_requests_update_activity_and_release_in_flight(weave_factory, failure):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_):
        entered.set()
        await release.wait()
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(422)

    weave = weave_factory(handler=handler)
    handle = await custom_deployment(weave, destroy_after_idle_mins=10)
    task = asyncio.create_task(handle.request("POST", "/api", json={"input": 1}))
    await asyncio.wait_for(entered.wait(), timeout=2)
    state = weave.lifecycle_service.get_state(handle.id)
    assert state.in_flight_requests == 1
    future = datetime.now(UTC) + timedelta(hours=1)
    assert not await weave.lifecycle_service.check_and_autostop(handle.id, now=future)
    before_end = datetime.now(UTC)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        expected = InferenceTimeoutError if failure == "timeout" else InferenceError
        with pytest.raises(expected):
            await task
    assert state.in_flight_requests == 0
    assert handle.last_activity_at >= before_end
    assert weave.lifecycle_service.is_idle(handle.id, now=future)
    restarted = weave_factory()
    assert (await restarted.attach(handle.id)).last_activity_at >= before_end


@pytest.mark.asyncio
async def test_concurrent_requests_and_cancellation_keep_other_request_in_flight(weave_factory):
    entered = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def handler(_):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
        await release.wait()
        return httpx.Response(200)

    weave = weave_factory(handler=handler)
    handle = await custom_deployment(weave)
    first = asyncio.create_task(handle.request("GET", "/first"))
    second = asyncio.create_task(handle.request("GET", "/second"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    state = weave.lifecycle_service.get_state(handle.id)
    assert state.in_flight_requests == 2
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert state.in_flight_requests == 1
    release.set()
    assert (await second).status_code == 200
    assert state.in_flight_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 307])
async def test_deployment_stopped_during_retry_or_redirect_is_not_sent_another_request(weave_factory, status):
    seen = []

    async def handler(request):
        seen.append(request)
        with patched_modal():
            await handle.stop()
        return httpx.Response(status, headers={"location": "/result"})

    weave = weave_factory(handler=handler)
    handle = await custom_deployment(weave)
    with pytest.raises(EndpointNotReadyError, match="no endpoint"):
        await handle.request("GET", "/api")
    assert handle.state == DeploymentState.STOPPED
    assert len(seen) == 1
    assert weave.lifecycle_service.get_state(handle.id).in_flight_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["activity", "backoff"])
async def test_cancellation_during_activity_or_backoff_releases_counter(weave_factory, phase):
    entered = asyncio.Event()
    release = asyncio.Event()
    weave = weave_factory(handler=lambda _: httpx.Response(503))
    handle = await custom_deployment(weave)
    transport = handle._inference_client._transport
    original_activity = weave.lifecycle_service.touch_activity

    async def blocked_activity(deployment_id):
        if not entered.is_set():
            entered.set()
            await release.wait()
        await original_activity(deployment_id)

    async def blocked_backoff(_):
        entered.set()
        await release.wait()

    if phase == "activity":
        weave.lifecycle_service.touch_activity = blocked_activity
    else:
        transport._sleep = blocked_backoff
    task = asyncio.create_task(handle.request("GET", "/api"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    state = weave.lifecycle_service.get_state(handle.id)
    assert state.in_flight_requests == 1
    before_end = datetime.now(UTC)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.in_flight_requests == 0
    assert handle.last_activity_at >= before_end


@pytest.mark.asyncio
async def test_transport_close_respects_client_ownership(transport_factory):
    transport, _, _ = transport_factory(lambda _: httpx.Response(200))
    await transport.request("GET", "/")
    borrowed = transport._client
    await transport.aclose()
    assert not borrowed.is_closed
    owned = InferenceTransport("owned", lambda: ENDPOINT)
    client = owned._get_client()
    await owned.aclose()
    assert client.is_closed and owned._client is None


@pytest.mark.asyncio
async def test_concurrent_multi_account_shared_client_isolation():
    recorded = []
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        async with lock:
            recorded.append(request)
        dep = "A" if "dep-a.test" in str(request.url) else "B"
        return httpx.Response(
            200,
            json={"status": "ok", "dep": dep},
            headers=[("set-cookie", f"stolen_from_{dep}=token_{dep}; Path=/; HttpOnly")],
        )

    shared_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={
            "Authorization": "Bearer hostile-client-token",
            "X-Hostile-Header": "hostile-secret",
            "Host": "evil.example",
        },
        params={"client_default_param": "hostile_param_val"},
        cookies={"hostile_client_cookie": "hostile_cookie_val"},
        auth=httpx.BasicAuth("hostile_user", "hostile_pass"),
    )

    t_a = InferenceTransport(
        "dep-a",
        endpoint_fn=lambda: "https://dep-a.test",
        auth_headers_fn=lambda: {"Authorization": "Bearer dep-A-token", "X-Deploy-A": "secret-A"},
        http_client=shared_client,
    )
    t_b = InferenceTransport(
        "dep-b",
        endpoint_fn=lambda: "https://dep-b.test",
        auth_headers_fn=lambda: {"Authorization": "Bearer dep-B-token", "X-Deploy-B": "secret-B"},
        http_client=shared_client,
    )

    # Concurrently run requests across A and B using both generic request() and legacy post()
    coros = [
        t_a.request("GET", "/infer_a", params={"caller_a": "1"}, headers={"X-Caller-A": "yes", "Cookie": "caller_cookie=a"}),
        t_b.request("GET", "/infer_b", params={"caller_b": "2"}, headers={"X-Caller-B": "yes"}),
        t_a.post("/post_a", json={"msg": "a"}, headers={"X-Caller-Post-A": "yes"}),
        t_b.post("/post_b", content=b"msg_b", headers={"X-Caller-Post-B": "yes"}),
        t_a.request("PUT", "/put_a", json={"put": "a"}),
        t_b.request("DELETE", "/del_b"),
    ]
    responses = await asyncio.gather(*coros)
    assert len(responses) == 6
    assert all(r.is_success for r in responses)

    for req in recorded:
        is_a = "dep-a.test" in str(req.url)
        headers = dict(req.headers)

        # 1. No hostile client defaults leak into requests
        assert "x-hostile-header" not in headers
        assert "hostile-client-token" not in str(headers)
        assert "Basic" not in headers.get("authorization", "")
        assert "hostile_client_cookie" not in headers.get("cookie", "")
        assert "client_default_param" not in req.url.params

        # 2. Host header matches target origin, not hostile client default
        expected_host = "dep-a.test" if is_a else "dep-b.test"
        assert headers.get("host") == expected_host

        # 3. Deployment isolation: account credentials and cookies do not leak across accounts
        if is_a:
            assert headers.get("authorization") == "Bearer dep-A-token"
            assert headers.get("x-deploy-a") == "secret-A"
            assert "x-deploy-b" not in headers
            assert "dep-b-token" not in str(headers)
            assert "stolen_from_b" not in headers.get("cookie", "")
            if req.url.path == "/infer_a":
                assert req.url.params.get("caller_a") == "1"
                assert headers.get("x-caller-a") == "yes"
                assert headers.get("cookie") == "caller_cookie=a"
        else:
            assert headers.get("authorization") == "Bearer dep-B-token"
            assert headers.get("x-deploy-b") == "secret-B"
            assert "x-deploy-a" not in headers
            assert "dep-a-token" not in str(headers)
            assert "stolen_from_a" not in headers.get("cookie", "")
            assert "cookie" not in headers

    # 4. Response Set-Cookie did not contaminate shared client state
    assert "stolen_from_a" not in shared_client.cookies
    assert "stolen_from_b" not in shared_client.cookies
    assert shared_client.cookies.get("hostile_client_cookie") == "hostile_cookie_val"
    assert shared_client.headers["x-hostile-header"] == "hostile-secret"
    assert shared_client.params["client_default_param"] == "hostile_param_val"

    # 5. Subsequent requests from either transport remain completely clean
    resp_subsequent = await t_a.request("GET", "/infer_a_subsequent")
    assert resp_subsequent.is_success
    req_subsequent = recorded[-1]
    assert "cookie" not in req_subsequent.headers
    assert "stolen_from_a" not in req_subsequent.headers.get("cookie", "")
    assert "stolen_from_b" not in req_subsequent.headers.get("cookie", "")

    # Cleanup
    await shared_client.aclose()


@pytest.mark.asyncio
async def test_sdk_multi_deployment_shared_client_isolation(weave_factory):
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(
            200,
            json={"status": "ok"},
            headers=[("set-cookie", "leaked_sid=stolen; Path=/")],
        )

    shared_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"X-Shared-Default": "hostile", "Authorization": "Bearer hostile-token"},
        cookies={"shared_cookie": "hostile_cookie"},
        params={"shared_query": "hostile_query"},
    )

    weave = weave_factory(accounts=modal_pool("a", "b"))
    weave._inference_http_client = shared_client
    dep_a = await custom_deployment(weave, account="a")
    dep_b = await custom_deployment(weave, account="b")

    resp_a, resp_b = await asyncio.gather(
        dep_a.request("GET", "/status", params={"account": "a"}),
        dep_b.request("POST", "/infer", json={"input": "b"}),
    )
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    for req in recorded:
        assert "x-shared-default" not in req.headers
        assert "hostile-token" not in str(req.headers)
        assert "shared_cookie" not in req.headers.get("cookie", "")
        assert "shared_query" not in req.url.params
        assert "leaked_sid" not in req.headers.get("cookie", "")

    assert "leaked_sid" not in shared_client.cookies
    assert shared_client.cookies.get("shared_cookie") == "hostile_cookie"
    await shared_client.aclose()
