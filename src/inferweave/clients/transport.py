"""HTTP transport for deployment inference calls: auth, timeouts, bounded cold-start retries.

The transport knows nothing about workloads or providers. It is handed the endpoint and a
callable producing auth headers, so the same code serves Modal proxy auth, a static API key or
an unauthenticated endpoint. It is deliberately usable on its own with ``httpx.MockTransport``.
"""

import asyncio
import datetime
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from inferweave.core.exceptions import (
    EndpointNotReadyError,
    InferenceError,
    InferenceTimeoutError,
)

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 500
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"})
_HTTP_METHODS = _IDEMPOTENT_METHODS | {"POST", "PATCH"}


@dataclass(frozen=True)
class InferenceConfig:
    """Timeout and bounded-retry policy for inference requests.

    Retries cover only cold-start-style failures: connection errors/timeouts, dropped
    connections and HTTP 502/503 (Modal wakes a scaled-to-zero app while answering these).
    Read timeouts and every other status (including 4xx) are never retried, because replaying a
    slow or rejected generation would only duplicate cost. The defaults allow roughly a minute
    of waiting for a container to start, then give up. Generic ``request()`` additionally
    disables retries for non-idempotent methods; ``post()`` retains the audio/image policy.
    """

    timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 6
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    jitter_ratio: float = 0.25
    retry_statuses: tuple[int, ...] = (502, 503)

    def __post_init__(self) -> None:
        durations = (self.timeout_seconds, self.connect_timeout_seconds,
                     self.backoff_base_seconds, self.backoff_max_seconds)
        if not all(math.isfinite(value) for value in durations):
            raise ValueError("Inference timeouts and backoff durations must be finite.")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("Inference timeouts must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("Backoff durations must be >= 0.")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be between 0 and 1.")


def redact(text: str, secrets: Mapping[str, str] | list[str] | tuple[str, ...]) -> str:
    """Replaces every known secret value in ``text`` with a placeholder."""
    values = secrets.values() if isinstance(secrets, Mapping) else secrets
    for value in sorted(values, key=len, reverse=True):
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def safe_endpoint(url: str) -> str:
    """Strips userinfo, query and fragment so an endpoint is safe to put in messages."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class InferenceTransport:
    """Sends deployment-bound HTTP requests with authentication and bounded retries."""

    def __init__(
        self,
        deployment_id: str,
        endpoint_fn: Callable[[], str | None],
        auth_headers_fn: Callable[[], Mapping[str, str]] | None = None,
        config: InferenceConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        default_port: int | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.deployment_id = deployment_id
        self.config = config or InferenceConfig()
        self._endpoint_fn = endpoint_fn
        self._auth_headers_fn = auth_headers_fn
        self._client = http_client
        self._owns_client = http_client is None
        self._default_port = default_port
        self._sleep = sleep
        self._random = random_fn

    def __repr__(self) -> str:
        return (
            f"InferenceTransport(deployment_id={self.deployment_id!r}, "
            f"endpoint={safe_endpoint(endpoint) if (endpoint := self._endpoint_fn()) else None!r})"
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or (self._owns_client and self._client.is_closed):
            self._client = httpx.AsyncClient(follow_redirects=False)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Closes the underlying HTTP client if this transport created it."""
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()
        if self._owns_client:
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return dict(self._auth_headers_fn()) if self._auth_headers_fn else {}

    def build_url(self, path: str) -> str:
        """Joins the current deployment endpoint and an API path."""
        if (
            path != path.lstrip()
            or "\\" in path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
        ):
            raise ValueError("Inference path must be a relative API path.")
        api = urlsplit(path)
        if api.scheme or api.netloc or path.startswith("//") or "#" in path:
            raise ValueError("Inference path must be relative, without an origin or fragment.")
        endpoint = (self._endpoint_fn() or "").strip()
        if not endpoint:
            raise EndpointNotReadyError(
                f"Deployment '{self.deployment_id}' has no endpoint URL yet.",
                deployment_id=self.deployment_id,
            )
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
            raise ValueError("Inference endpoint must be an HTTP(S) URL without userinfo.")
        netloc = parts.netloc
        if parts.scheme == "http" and parts.port is None and self._default_port:
            netloc = f"{netloc}:{self._default_port}"
        base_path = parts.path.rstrip("/")
        api_path = api.path if api.path.startswith("/") else f"/{api.path}"
        return urlunsplit((parts.scheme, netloc, f"{base_path}{api_path}", api.query, ""))

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        cfg = self.config
        delay = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * (2**attempt))
        delay *= 1.0 - cfg.jitter_ratio * self._random()
        if retry_after is not None:
            delay = max(delay, min(retry_after, cfg.backoff_max_seconds))
        return delay

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _body_snippet(self, response: httpx.Response, secrets: Mapping[str, str]) -> str:
        text = response.content.decode("utf-8", errors="replace")
        return redact(text, secrets)[:_MAX_BODY_CHARS]

    @staticmethod
    async def _send_single_request(
        client: httpx.AsyncClient,
        request: httpx.Request,
    ) -> httpx.Response:
        """Dispatches an explicit Request through client's transport without mutating client state."""
        if client.is_closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        transport = client._transport_for_url(request.url)
        start = time.perf_counter()
        try:
            response = await transport.handle_async_request(request)
        except httpx.RequestError as exc:
            exc.request = request
            raise
        response.request = request
        response.default_encoding = getattr(
            client,
            "default_encoding",
            getattr(client, "_default_encoding", "utf-8"),
        )
        try:
            await response.aread()
        except BaseException:
            await response.aclose()
            raise
        response.elapsed = datetime.timedelta(seconds=time.perf_counter() - start)
        return response

    async def _send_following_redirects(
        self,
        method: str,
        url: str,
        content: bytes | str | None,
        json_body: object,
        headers: httpx.Headers,
        timeout: httpx.Timeout,
        params: httpx.QueryParams | None,
        diagnostic_endpoint: str,
    ) -> httpx.Response:
        """Sends one logical request, following only same-origin redirects.

        Modal answers long-running requests with a 303 to a result URL on the same host. Custom
        auth headers are not stripped by httpx on cross-origin redirects, so redirects to another
        origin are refused instead of followed.
        """
        client = self._get_client()
        initial = httpx.URL(url)
        origin = (initial.scheme, initial.host, initial.port)
        for _ in range(_MAX_REDIRECTS + 1):
            if not self._endpoint_fn():
                raise EndpointNotReadyError(
                    f"Deployment '{self.deployment_id}' has no endpoint URL yet.",
                    deployment_id=self.deployment_id,
                )
            request = httpx.Request(
                method,
                url,
                content=content,
                json=json_body,
                headers=headers,
                params=params,
                extensions={"timeout": timeout.as_dict()},
            )
            response = await self._send_single_request(client, request)
            params = None  # Redirect queries belong to Location, not the original request.
            location = response.headers.get("location")
            if response.status_code not in _REDIRECT_STATUSES or not location:
                return response
            await response.aclose()
            if any(ord(char) < 32 or ord(char) == 127 for char in location):
                raise InferenceError(
                    "Invalid redirect from the inference endpoint.",
                    deployment_id=self.deployment_id,
                    endpoint=diagnostic_endpoint,
                )
            try:
                target = httpx.URL(urljoin(str(response.url), location))
            except (ValueError, httpx.InvalidURL):
                raise InferenceError(
                    "Invalid redirect from the inference endpoint.",
                    deployment_id=self.deployment_id,
                    status_code=response.status_code,
                    endpoint=diagnostic_endpoint,
                ) from None
            if (
                (target.scheme, target.host, target.port) != origin
                or target.userinfo
                or "\\" in location
            ):
                raise InferenceError(
                    "Refusing to follow a cross-origin redirect from the inference endpoint.",
                    deployment_id=self.deployment_id,
                    status_code=response.status_code,
                    endpoint=diagnostic_endpoint,
                )
            if (response.status_code == 303 and method != "HEAD") or (
                response.status_code in {301, 302} and method == "POST"
            ):
                method = "GET"
                content, json_body = None, None
                headers = httpx.Headers(headers)
                for key in ("content-type", "content-length", "transfer-encoding"):
                    headers.pop(key, None)
            url = str(target)
        raise InferenceError(
            "Too many redirects from the inference endpoint.",
            deployment_id=self.deployment_id,
            endpoint=diagnostic_endpoint,
        )

    async def post(
        self,
        path: str,
        *,
        content: bytes | None = None,
        json: object | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """POSTs to ``path`` and returns the successful (2xx) response.

        Raises:
            EndpointNotReadyError: no endpoint, or still failing with a retryable condition
                after ``max_retries`` retries.
            InferenceTimeoutError: the request timed out while waiting for a response.
            InferenceError: any other failure (including 4xx and refused redirects).
        """
        return await self._request(
            "POST", path, content=content if json is None else None, json=json,
            headers=headers, timeout=timeout,
            retry=True,
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
        """Returns a buffered 2xx response; HTTP failures use the inference error types.

        Only idempotent methods retry cold-start failures. POST/PATCH are attempted once.
        Read/write timeouts are never retried. JSON and raw content are mutually exclusive.
        """
        method = method.upper()
        if method not in _HTTP_METHODS:
            raise ValueError("Unsupported inference HTTP method.")
        if content is not None and json is not None:
            raise ValueError("Pass either json or content, not both.")
        return await self._request(
            method, path, params=params, content=content, json=json, headers=headers,
            timeout=timeout, retry=method in _IDEMPOTENT_METHODS,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        content: bytes | str | None = None,
        json: object | None = None,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        retry: bool,
    ) -> httpx.Response:
        cfg = self.config
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Inference timeout must be positive and finite.")
        request_timeout = httpx.Timeout(
            timeout if timeout is not None else cfg.timeout_seconds,
            connect=cfg.connect_timeout_seconds,
        )
        request_params = (
            httpx.QueryParams(params if isinstance(params, Mapping) else list(params))
            if params is not None else None
        )
        last_failure = "no attempt made"
        last_status: int | None = None
        last_body: str | None = None
        url = ""
        secrets: dict[str, str] = {}
        max_retries = cfg.max_retries if retry else 0

        for attempt in range(max_retries + 1):
            url = self.build_url(path)
            auth = self._auth_headers()
            request_headers = httpx.Headers(headers)
            if "host" in request_headers:
                raise ValueError("Overriding the inference Host header is not allowed.")
            request_headers.update(auth)  # Case-insensitive; remove caller duplicates.
            secrets = dict(request_headers)
            for key in ("authorization", "proxy-authorization"):
                if value := request_headers.get(key):
                    # A server may echo the token without the Bearer/Basic scheme.
                    secrets[f"{key}-credential"] = value.partition(" ")[2]
            request_headers["Host"] = httpx.URL(url).netloc.decode("ascii")
            diagnostic_endpoint = redact(safe_endpoint(url), secrets)
            retry_after: float | None = None
            try:
                response = await self._send_following_redirects(
                    method, url, content, json, request_headers, request_timeout,
                    request_params, diagnostic_endpoint,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as err:
                last_failure = f"connection failed ({type(err).__name__})"
                last_status, last_body = None, None
            except httpx.TimeoutException as err:
                raise InferenceTimeoutError(
                    f"Inference request to {diagnostic_endpoint} timed out "
                    f"({type(err).__name__}).",
                    deployment_id=self.deployment_id,
                    endpoint=diagnostic_endpoint,
                ) from None
            except httpx.HTTPError as err:
                raise InferenceError(
                    f"Inference request to {diagnostic_endpoint} failed "
                    f"({type(err).__name__}).",
                    deployment_id=self.deployment_id,
                    endpoint=diagnostic_endpoint,
                ) from None
            else:
                if response.is_success:
                    return response
                body = self._body_snippet(response, secrets)
                await response.aclose()
                if response.status_code in cfg.retry_statuses:
                    last_failure = f"HTTP {response.status_code}"
                    last_status, last_body = response.status_code, body
                    retry_after = self._parse_retry_after(response)
                else:
                    hint = (
                        " The endpoint rejected the credentials; check the provider auth "
                        "configuration (for Modal: MODAL_PROXY_TOKEN_ID/MODAL_PROXY_TOKEN_SECRET)."
                        if response.status_code in {401, 403}
                        else ""
                    )
                    raise InferenceError(
                        f"Inference request to {diagnostic_endpoint} failed with "
                        f"HTTP {response.status_code}.{hint}",
                        deployment_id=self.deployment_id,
                        status_code=response.status_code,
                        endpoint=diagnostic_endpoint,
                        response_body=body,
                    )

            if attempt >= max_retries:
                break
            delay = self._backoff_delay(attempt, retry_after)
            logger.info(
                "Inference to '%s' not ready (%s); retry %d/%d in %.1fs",
                self.deployment_id,
                last_failure,
                attempt + 1,
                max_retries,
                delay,
            )
            await self._sleep(delay)

        raise EndpointNotReadyError(
            f"Endpoint {diagnostic_endpoint} of deployment '{self.deployment_id}' was not "
            f"ready after {max_retries} retries (last failure: {last_failure}).",
            deployment_id=self.deployment_id,
            status_code=last_status,
            endpoint=diagnostic_endpoint,
            response_body=last_body,
        )
