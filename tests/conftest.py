from __future__ import annotations

import socket
from typing import Any

import pytest


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_live_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require tests to use mocks instead of live API endpoints."""

    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def assert_mocked(address: Any) -> None:
        host = address[0] if isinstance(address, tuple) and address else None
        if str(host).lower() not in _LOCAL_HOSTS:
            raise AssertionError(
                "Tests must mock external API calls; attempted live connection "
                f"to {address!r}."
            )

    def guarded_connect(self: socket.socket, address: Any) -> None:
        assert_mocked(address)
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        assert_mocked(address)
        return original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
