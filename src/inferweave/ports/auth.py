"""Port contract for resolving runtime credentials needed to call a deployment endpoint."""

from abc import ABC, abstractmethod

from inferweave.accounts.models import ProviderAccount


class EndpointAuthPort(ABC):
    """Resolves authentication headers for requests sent to a deployment endpoint.

    Credentials are resolved at request time from the account that owns the deployment (or from
    runtime configuration for the ambient account) and are never persisted in deployment
    records. Both the inference clients and the healthcheck/readiness probes use the same port
    so that a protected endpoint stays reachable for every kind of request.
    """

    @abstractmethod
    def headers_for(
        self,
        provider: str,
        endpoint_url: str | None,
        account: ProviderAccount | None = None,
    ) -> dict[str, str]:
        """Returns auth headers for ``endpoint_url`` hosted on ``provider`` (empty if none apply).

        ``account`` is the provider account owning the deployment. Resolvers must only use that
        account's endpoint credentials, never another account's.
        """

    def is_configured_for(self, provider: str, account: ProviderAccount | None = None) -> bool:
        """Returns True when endpoint credentials for ``provider``/``account`` are available."""
        return bool(self.headers_for(provider, None, account))
