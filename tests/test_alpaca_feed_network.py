"""Tests for transient network error detection and bar batching in alpaca_feed."""

from daytrading.data.alpaca_feed import _chunk_symbols, _is_transient_network_error


def test_dns_error_message_is_transient() -> None:
    exc = Exception(
        "HTTPSConnectionPool(host='data.alpaca.markets', port=443): "
        "Failed to resolve 'data.alpaca.markets' "
        "([Errno 8] nodename nor servname provided, or not known)"
    )
    assert _is_transient_network_error(exc) is True


def test_auth_error_not_transient() -> None:
    exc = Exception("403 Forbidden — invalid API key")
    assert _is_transient_network_error(exc) is False


def test_chained_dns_error_is_transient() -> None:
    inner = Exception("Failed to resolve 'data.alpaca.markets'")
    outer = Exception("Max retries exceeded")
    outer.__cause__ = inner
    assert _is_transient_network_error(outer) is True


def test_chunk_symbols_splits_batches() -> None:
    assert _chunk_symbols([], 10) == []
    assert _chunk_symbols(["A", "B", "C"], 2) == [["A", "B"], ["C"]]
    assert _chunk_symbols(["A", "B", "C"], 10) == [["A", "B", "C"]]
    assert _chunk_symbols(["A", "B", "C"], 1) == [["A"], ["B"], ["C"]]
