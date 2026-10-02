"""HTTP transport for deployment inference calls: auth, timeouts, bounded cold-start retries.

The transport knows nothing about workloads or providers. It is handed the endpoint and a
callable producing auth headers, so the same code serves Modal proxy auth, a static API key or
an unauthenticated endpoint. It is deliberately usable on its own with ``httpx.MockTransport``.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class InferenceConfig:
    """Timeout and bounded-retry policy for inference requests.

    Retries cover only cold-start-style failures: connection errors/timeouts, dropped
    connections and HTTP 502/503 (Modal wakes a scaled-to-zero app while answering these).
    Read timeouts and every other status (including 4xx) are never retried, because replaying a
    slow or rejected generation would only duplicate cost. The defaults allow roughly a minute
    of waiting for a container to start, then give up.
    """

    timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 6
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    jitter_ratio: float = 0.25
    retry_statuses: tuple[int, ...] = (502, 503)

    def __post_init__(self) -> None:
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
    for value in values:
        if value and len(value) >= 4:
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
    """Sends authenticated POST requests to a deployment endpoint with bounded retries."""

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
            f"endpoint={self._endpoint_fn()!r})"
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
        endpoint = (self._endpoint_fn() or "").strip()
        if not endpoint:
            raise EndpointNotReadyError(
                f"Deployment '{self.deployment_id}' has no endpoint URL yet.",
                deployment_id=self.deployment_id,
            )
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"
        parts = urlsplit(endpoint)
        netloc = parts.netloc
        if parts.scheme == "http" and parts.port is None and self._default_port:
            netloc = f"{netloc}:{self._default_port}"
        base_path = parts.path.rstrip("/")
        api_path = path if path.startswith("/") else f"/{path}"
        return urlunsplit((parts.scheme, netloc, f"{base_path}{api_path}", "", ""))

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
        text = response.content[:_MAX_BODY_CHARS * 2].decode("utf-8", errors="replace")
        return redact(text, secrets)[:_MAX_BODY_CHARS]

    async def _send_following_redirects(
        self,
        url: str,
        content: bytes | None,
        json_body: object,
        headers: dict[str, str],
        timeout: httpx.Timeout,
    ) -> httpx.Response:
        """Sends one logical request, following only same-origin redirects.

        Modal answers long-running requests with a 303 to a result URL on the same host. Custom
        auth headers are not stripped by httpx on cross-origin redirects, so redirects to another
        origin are refused instead of followed.
        """
        client = self._get_client()
        method = "POST"
        origin = urlsplit(url)[:2]
        for _ in range(_MAX_REDIRECTS + 1):
            kwargs: dict[str, object] = {"headers": headers, "timeout": timeout}
            if method == "POST":
                if json_body is not None:
                    kwargs["json"] = json_body
                elif content is not None:
                    kwargs["content"] = content
            response = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
            location = response.headers.get("location")
            if response.status_code not in _REDIRECT_STATUSES or not location:
                return response
            target = urljoin(url, location)
            if urlsplit(target)[:2] != origin:
                raise InferenceError(
                    "Refusing to follow a cross-origin redirect from the inference endpoint.",
                    deployment_id=self.deployment_id,
                    status_code=response.status_code,
                    endpoint=safe_endpoint(url),
                )
            if response.status_code == 303 or (
                response.status_code in {301, 302} and method == "POST"
            ):
                method = "GET"
            url = target
        raise InferenceError(
            "Too many redirects from the inference endpoint.",
            deployment_id=self.deployment_id,
            endpoint=safe_endpoint(url),
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
        cfg = self.config
        request_timeout = httpx.Timeout(
            timeout if timeout is not None else cfg.timeout_seconds,
            connect=cfg.connect_timeout_seconds,
        )
        last_failure = "no attempt made"
        last_status: int | None = None
        last_body: str | None = None
        url = ""
        secrets: dict[str, str] = {}

        for attempt in range(cfg.max_retries + 1):
            url = self.build_url(path)
            auth = self._auth_headers()
            secrets = auth
            request_headers = {**(headers or {}), **auth}
            retry_after: float | None = None
            try:
                response = await self._send_following_redirects(
                    url, content, json, request_headers, request_timeout
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as err:
                last_failure = f"connection failed ({type(err).__name__})"
                last_status, last_body = None, None
            except httpx.TimeoutException as err:
                raise InferenceTimeoutError(
                    f"Inference request to {safe_endpoint(url)} timed out "
                    f"({type(err).__name__}).",
                    deployment_id=self.deployment_id,
                    endpoint=safe_endpoint(url),
                ) from None
            except httpx.HTTPError as err:
                raise InferenceError(
                    f"Inference request to {safe_endpoint(url)} failed "
                    f"({type(err).__name__}).",
                    deployment_id=self.deployment_id,
                    endpoint=safe_endpoint(url),
                ) from None
            else:
                if response.is_success:
                    return response
                body = self._body_snippet(response, secrets)
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
                        f"Inference request to {safe_endpoint(url)} failed with "
                        f"HTTP {response.status_code}.{hint}",
                        deployment_id=self.deployment_id,
                        status_code=response.status_code,
                        endpoint=safe_endpoint(url),
                        response_body=body,
                    )

            if attempt >= cfg.max_retries:
                break
            delay = self._backoff_delay(attempt, retry_after)
            logger.info(
                "Inference to '%s' not ready (%s); retry %d/%d in %.1fs",
                self.deployment_id,
                last_failure,
                attempt + 1,
                cfg.max_retries,
                delay,
            )
            await self._sleep(delay)

        raise EndpointNotReadyError(
            f"Endpoint {safe_endpoint(url)} of deployment '{self.deployment_id}' was not "
            f"ready after {cfg.max_retries} retries (last failure: {last_failure}).",
            deployment_id=self.deployment_id,
            status_code=last_status,
            endpoint=safe_endpoint(url),
            response_body=last_body,
        )
