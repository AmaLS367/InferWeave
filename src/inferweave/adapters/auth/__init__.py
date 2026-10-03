"""Endpoint authentication adapters."""

from inferweave.adapters.auth.endpoint_auth import (
    CompositeEndpointAuth,
    LightningEndpointAuth,
    ModalProxyAuth,
    NoEndpointAuth,
    StaticHeaderAuth,
    default_endpoint_auth,
)

__all__ = [
    "CompositeEndpointAuth",
    "LightningEndpointAuth",
    "ModalProxyAuth",
    "NoEndpointAuth",
    "StaticHeaderAuth",
    "default_endpoint_auth",
]
