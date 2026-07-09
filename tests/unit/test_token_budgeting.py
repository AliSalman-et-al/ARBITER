from __future__ import annotations

from arbiter import token_budgeting


def test_count_tokens_uses_packaged_tiktoken_encoding() -> None:
    token_budgeting._encoding.cache_clear()

    encoding = token_budgeting._encoding()

    assert encoding.__class__.__name__ != "_WhitespaceEncoding"
