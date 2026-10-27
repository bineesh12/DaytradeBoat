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


def _fake_stream():
    from queue import Queue
    from unittest.mock import AsyncMock
    from daytrading.data.alpaca_feed import _ResilientDataStream
    stream = _ResilientDataStream.__new__(_ResilientDataStream)
    stream._handlers = {'trades': {'SPY': object()}}
    stream._stop_stream_queue = Queue()
    stream._start_ws = AsyncMock()
    stream._send_subscribe_msg = AsyncMock()
    stream.close = AsyncMock()
    return stream


def test_dns_reconnect_backs_off_and_releases_every_failed_socket():
    import asyncio
    import socket
    stream = _fake_stream()
    waits = []
    stream._start_ws.side_effect = socket.gaierror('DNS unavailable')
    async def pause(delay):
        waits.append(delay)
        if len(waits) == 8:
            stream._should_run = False
    stream._reconnect_pause = pause
    asyncio.run(stream._run_forever())
    assert waits == [1, 2, 4, 8, 16, 30, 30, 30]
    assert stream.close.await_count == 8
    assert not stream._running


def test_connection_limit_preserves_longer_backoff_until_data_recovers():
    import asyncio
    stream = _fake_stream()
    waits = []
    stream._start_ws.side_effect = ValueError('connection limit exceeded')
    async def pause(delay):
        waits.append(delay)
        if len(waits) == 3:
            stream._should_run = False
    stream._reconnect_pause = pause
    asyncio.run(stream._run_forever())
    assert waits == [30, 30, 60]
    assert stream.close.await_count == 3


def test_mute_socket_retries_grow_and_actual_data_resets_backoff():
    import asyncio
    stream = _fake_stream()
    waits = []
    async def consume():
        stream._data_received = len(waits) == 2
        return True
    async def pause(delay):
        waits.append(delay)
        if len(waits) == 4:
            stream._should_run = False
    stream._consume = consume
    stream._reconnect_pause = pause
    asyncio.run(stream._run_forever())
    assert waits == [1, 2, 1, 2]
    assert stream._send_subscribe_msg.await_count == 4
    assert set(stream._handlers['trades']) == {'SPY'}


def test_fatal_subscription_failure_closes_and_does_not_retry():
    import asyncio
    from unittest.mock import AsyncMock
    stream = _fake_stream()
    stream._start_ws.side_effect = ValueError('insufficient subscription')
    stream._reconnect_pause = AsyncMock()
    asyncio.run(stream._run_forever())
    stream.close.assert_awaited_once()
    stream._reconnect_pause.assert_not_awaited()
    assert not stream._running


def test_stop_interrupts_backoff_without_another_connection():
    import asyncio
    stream = _fake_stream()
    stream._start_ws.side_effect = OSError('network unavailable')
    async def check():
        task = asyncio.create_task(stream._run_forever())
        while stream.close.await_count == 0:
            await asyncio.sleep(0)
        await stream.stop_ws()
        await asyncio.wait_for(task, timeout=.2)
    asyncio.run(check())
    stream._start_ws.assert_awaited_once()


def test_connected_property_distinguishes_retrying_thread_from_socket():
    from types import SimpleNamespace
    from daytrading.data.alpaca_feed import AlpacaStreamFeed
    feed = AlpacaStreamFeed.__new__(AlpacaStreamFeed)
    feed._thread = SimpleNamespace(is_alive=lambda: True)
    feed._running = True
    feed._stream = SimpleNamespace(_running=False)
    assert feed.is_running and not feed.is_connected
    feed._stream._running = True
    assert feed.is_connected
    feed._running = False
    assert not feed.is_connected
