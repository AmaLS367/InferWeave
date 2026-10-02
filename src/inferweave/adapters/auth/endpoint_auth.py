"""Concrete EndpointAuthPort implementations.

Secrets live only in process memory / the environment. Every class here hides secret values
from ``repr``/``str`` so accidental logging cannot leak them.
"""

import os
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

from inferweave.ports.auth import EndpointAuthPort

MODAL_PROXY_TOKEN_ID_ENV = "MODAL_PROXY_TOKEN_ID"
MODAL_PROXY_TOKEN_SECRET_ENV = "MODAL_PROXY_TOKEN_SECRET"
_MODAL_HOST_SUFFIXES = (".modal.run", ".modal.host")


def _host_matches(endpoint_url: str | None, suffixes: tuple[str, ...]) -> bool:
    if not endpoint_url:
        return True
    raw = endpoint_url.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    host = (urlsplit(raw).hostname or "").lower()
    return host.endswith(suffixes)


class NoEndpointAuth(EndpointAuthPort):
    """Sends no credentials (public endpoints)."""

    def headers_for(self, provider: str, endpoint_url: str | None) -> dict[str, str]:
        return {}


class StaticHeaderAuth(EndpointAuthPort):
    """Applies a fixed header set to every endpoint of the given providers (all if empty)."""

    def __init__(
        self, headers: Mapping[str, str], providers: tuple[str, ...] = ()
    ) -> None:
        self._headers = dict(headers)
        self._providers = tuple(p.lower() for p in providers)

    def headers_for(self, provider: str, endpoint_url: str | None) -> dict[str, str]:
        if self._providers and provider.lower() not in self._providers:
            return {}
        return dict(self._headers)

    def __repr__(self) -> str:
        return f"StaticHeaderAuth(headers={sorted(self._headers)}, providers={self._providers})"


class ModalProxyAuth(EndpointAuthPort):
    """Modal proxy-auth tokens (``Modal-Key`` / ``Modal-Secret`` headers).

    Pairs with ``@modal.web_server(..., requires_proxy_auth=True)``. Tokens are created in the
    Modal workspace settings and read from ``MODAL_PROXY_TOKEN_ID`` / ``MODAL_PROXY_TOKEN_SECRET``
    at request time unless explicit values or a ``token_getter`` are supplied. Headers are only
    produced for ``*.modal.run`` / ``*.modal.host`` endpoints so the token cannot be sent to a
    foreign host by mistake.
    """

    provider_name = "modal"

    def __init__(
        self,
        token_id: str | None = None,
        token_secret: str | None = None,
        token_getter: Callable[[], tuple[str | None, str | None]] | None = None,
    ) -> None:
        self._token_id = token_id
        self._token_secret = token_secret
        self._token_getter = token_getter

    def _tokens(self) -> tuple[str | None, str | None]:
        if self._token_getter is not None:
            return self._token_getter()
        return (
            self._token_id or os.environ.get(MODAL_PROXY_TOKEN_ID_ENV),
            self._token_secret or os.environ.get(MODAL_PROXY_TOKEN_SECRET_ENV),
        )

    def headers_for(self, provider: str, endpoint_url: str | None) -> dict[str, str]:
        if provider.lower() != self.provider_name:
            return {}
        if not _host_matches(endpoint_url, _MODAL_HOST_SUFFIXES):
            return {}
        token_id, token_secret = self._tokens()
        if not token_id or not token_secret:
            return {}
        return {"Modal-Key": token_id, "Modal-Secret": token_secret}

    def __repr__(self) -> str:
        return f"ModalProxyAuth(configured={all(self._tokens())})"


class CompositeEndpointAuth(EndpointAuthPort):
    """Merges the headers produced by several auth providers (later ones win)."""

    def __init__(self, *auths: EndpointAuthPort) -> None:
        self._auths = auths

    def headers_for(self, provider: str, endpoint_url: str | None) -> dict[str, str]:
        merged: dict[str, str] = {}
        for auth in self._auths:
            merged.update(auth.headers_for(provider, endpoint_url))
        return merged

    def __repr__(self) -> str:
        return f"CompositeEndpointAuth({', '.join(repr(a) for a in self._auths)})"


def default_endpoint_auth() -> EndpointAuthPort:
    """Environment-driven auth used when the SDK is not given an explicit resolver."""
    return CompositeEndpointAuth(ModalProxyAuth())
