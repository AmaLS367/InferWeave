"""Shared pytest fixtures."""

import pytest

TEST_PROXY_TOKEN_ID = "wk-test-proxy-id-0000"
TEST_PROXY_TOKEN_SECRET = "ws-test-proxy-secret-1111"


@pytest.fixture(autouse=True)
def modal_proxy_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gives every test deterministic Modal proxy-auth tokens.

    Modal endpoints are protected by default, so deploying through the Modal provider requires
    tokens to be configured. Tests of the missing-token path delete these explicitly.
    """
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", TEST_PROXY_TOKEN_ID)
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", TEST_PROXY_TOKEN_SECRET)
