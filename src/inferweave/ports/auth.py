"""Port contract for resolving runtime credentials needed to call a deployment endpoint."""

from abc import ABC, abstractmethod


class EndpointAuthPort(ABC):
    """Resolves authentication headers for requests sent to a deployment endpoint.

    Credentials are resolved at request time from runtime configuration (environment, secret
    stores, ...) and are never persisted in deployment records. Both the inference clients and
    the healthcheck/readiness probes use the same port so that a protected endpoint stays
    reachable for every kind of request.
    """

    @abstractmethod
    def headers_for(self, provider: str, endpoint_url: str | None) -> dict[str, str]:
        """Returns auth headers for ``endpoint_url`` hosted on ``provider`` (empty if none apply)."""

    def is_configured_for(self, provider: str) -> bool:
        """Returns True when credentials for ``provider`` are available at this moment."""
        return bool(self.headers_for(provider, None))
