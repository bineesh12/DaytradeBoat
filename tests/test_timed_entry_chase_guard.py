from __future__ import annotations

from datetime import datetime, timezone

from daytrading.models import Bar, Quote, ScanResult, SignalAction, Timeframe, TradeSignal
from daytrading.runner import AlpacaRunner
from daytrading.strategy.entry_policy import EntryPolicy


class _Agg:
    def __init__(self, bars):
        self._bars = bars

    def get_latest_10s(self, symbol: str, count: int = 1):
        return self._bars[-count:]


def _bar(symbol: str = "DXST", close: float = 5.00) -> Bar:
    return Bar(
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        ts=datetime.now(timezone.utc),
    )


def _signal(symbol: str = "DXST", price: float = 5.00) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        reason="ABC continuation",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="abc_continuation",
            ts=datetime.now(timezone.utc),
            score=8.0,
            criteria={
                "pattern": "abc_continuation",
                "close": price,
            },
        ),
    )


def _opening_range_signal(symbol: str = "BGMS", price: float = 3.26) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=416,
        entry_price=price,
        stop_loss=3.14,
        reason="Opening Range Breakout",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="opening_range_breakout",
            ts=datetime.now(timezone.utc),
            score=31.0,
            criteria={
                "pattern": "opening_range_breakout",
                "close": price,
                "orb_high": 3.16,
                "stop_price": 3.14,
            },
        ),
    )


def _level_breakout_signal(symbol: str = "DAIC", price: float = 4.32) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.ENTER_LONG,
        quantity=100,
        entry_price=price,
        stop_loss=3.98,
        reason="Level Breakout Reclaim",
        scan_result=ScanResult(
            symbol=symbol,
            scanner_name="level_breakout_reclaim",
            ts=datetime.now(timezone.utc),
            score=44.0,
            criteria={
                "pattern": "level_breakout_reclaim",
                "close": price,
                "breakout_level": 4.12,
                "base_high": 4.12,
                "stop_price": 3.98,
            },
        ),
    )


def _runner(live_price: float) -> AlpacaRunner:
    runner = AlpacaRunner.__new__(AlpacaRunner)
    runner._live_prices = lambda symbols: {symbols[0]: live_price}
    runner._latest_price = lambda symbol: live_price
    runner._quote_buffer = {}
    runner._bar_aggregator = None
    return runner


class _Journal:
    def __init__(self) -> None:
        self.records = []

    def record(self, event_type, payload, ts=None):
        self.records.append((event_type, payload, ts))


class _FloatChecker:
    def __init__(self, cached=None, fetched=None) -> None:
        self.cached = cached
        self.fetched = fetched
        self.cached_calls = 0
        self.fetch_calls = 0
        self._avg_vol_cache = {}

    def get_float_cached(self, symbol: str):
        self.cached_calls += 1
        return self.cached

    def get_float(self, symbol: str):
        self.fetch_calls += 1
        return self.fetched


def _policy_runner() -> AlpacaRunner:
    runner = _runner(5.0)
    runner._entry_policy = EntryPolicy()
    runner._journal = _Journal()
    runner._market_phase = lambda: "OPEN"
    runner._tick_buffer = {}
    runner._quote_buffer = {}
    runner._float_checker = None
    return runner


def test_timed_entry_chase_guard_rejects_extended_hot_signal() -> None:
    runner = _runner(5.20)

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.20))

    assert reason is not None
    assert "ran 4.0%" in reason


def test_timed_entry_chase_guard_allows_near_signal_price() -> None:
    runner = _runner(5.08)

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.08))

    assert reason is None


def test_shared_entry_quality_records_structured_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        "daytrading.strategy.entry_policy.check_entry_quality",
        lambda *args, **kwargs: "entry score too low (75/100, need 80+)",
    )
    runner = _policy_runner()

    reason = runner._shared_entry_quality_reject(
        "DXST",
        [_bar(close=5.0), _bar(close=5.05), _bar(close=5.10)],
        signal=_signal(price=5.10),
        stage="timed_entry_final_guard",
        source="timed_entry",
    )

    assert "entry score too low" in reason
    assert runner._journal.records
    event_type, payload, _ = runner._journal.records[-1]
    assert event_type == "entry_decision"
    assert payload["source"] == "timed_entry"
    assert payload["stage"] == "timed_entry_final_guard"
    assert payload["blocked_layer"] == "entry_guard"
    assert payload["passed"] is False


def test_shared_entry_quality_uses_cached_or_stored_float_without_network(monkeypatch) -> None:
    monkeypatch.setattr("daytrading.strategy.entry_guard.ENTRY_MAX_FLOAT_SHARES", 20_000_000)
    monkeypatch.setattr("daytrading.strategy.entry_guard._xgb_model", None)
    runner = _policy_runner()
    runner._float_checker = _FloatChecker(cached=120_000_000, fetched=5_000_000)
    now = datetime.now(timezone.utc)
    bars = [
        Bar(
            symbol="SPCE",
            ts=now,
            open=4.8 + i * 0.01,
            high=5.05,
            low=4.75 + i * 0.01,
            close=4.9 + i * 0.01,
            volume=100_000,
            timeframe=Timeframe.MIN_1,
        )
        for i in range(25)
    ]
    bars[-1] = Bar(
        symbol="SPCE",
        ts=now,
        open=5.00,
        high=5.04,
        low=4.98,
        close=5.02,
        volume=250_000,
        timeframe=Timeframe.MIN_1,
    )
    runner._quote_buffer = {
        "SPCE": [
            Quote(symbol="SPCE", ts=now, bid=5.01, ask=5.02, bid_size=2000, ask_size=2000)
            for _ in range(3)
        ]
    }
    signal = _signal(symbol="SPCE", price=5.02)
    signal.scan_result.criteria.update({
        "pattern": "first_pullback_reclaim",
        "setup_tier": "A+ setup",
    })

    reason = runner._shared_entry_quality_reject(
        "SPCE",
        bars,
        signal=signal,
        stage="timed_entry_final_guard",
        source="timed_entry",
    )

    assert runner._float_checker.cached_calls == 1
    assert runner._float_checker.fetch_calls == 0
    assert reason is not None
    assert "float too large" in reason.lower()


def test_timed_entry_chase_guard_allows_one_tick_sub_two_spread() -> None:
    runner = _runner(1.66)
    now = datetime.now(timezone.utc)
    runner._quote_buffer = {
        "BATL": [
            Quote(symbol="BATL", ts=now, bid=1.655, ask=1.665, bid_size=1500, ask_size=1500)
            for _ in range(3)
        ]
    }

    reason = runner._timed_entry_chase_reject(_signal(symbol="BATL", price=1.66), _bar(symbol="BATL", close=1.66))

    assert reason is None


def test_timed_entry_chase_guard_uses_opportunity_scaled_spread_policy() -> None:
    runner = _runner(2.13)
    now = datetime.now(timezone.utc)
    runner._quote_buffer = {
        "RGNT": [
            Quote(symbol="RGNT", ts=now, bid=2.121, ask=2.139, bid_size=1500, ask_size=1400)
            for _ in range(3)
        ]
    }
    runner._bar_buffer = {
        "RGNT": [
            Bar(
                symbol="RGNT",
                ts=now,
                open=1.95 + i * 0.01,
                high=2.14,
                low=1.93 + i * 0.01,
                close=2.00 + i * 0.0068,
                volume=90_000,
                timeframe=Timeframe.MIN_1,
            )
            for i in range(20)
        ]
    }
    signal = _signal(symbol="RGNT", price=2.13)
    signal.scan_result.criteria.update({
        "pattern": "first_pullback_reclaim",
        "setup_tier": "A+ setup",
        "entry_tier": "a_plus_reclaim_scout",
    })

    reason = runner._timed_entry_chase_reject(signal, _bar(symbol="RGNT", close=2.13))

    assert reason is None
    assert signal.scan_result.criteria["spread_exception"] == "opportunity_scaled"
    assert signal.scan_result.criteria["spread_size_factor"] < 1.0


def test_timed_entry_chase_guard_uses_queued_base_not_late_signal_price() -> None:
    runner = _runner(2.34)
    signal = _signal(symbol="BATL", price=2.30)
    signal.scan_result.criteria["queued_entry_price"] = 1.66

    reason = runner._timed_entry_chase_reject(signal, _bar(symbol="BATL", close=2.34))

    assert reason is not None
    assert "ran" in reason
    assert "1.6600" in reason


def test_timed_entry_chase_anchor_persists_across_requeues() -> None:
    # A grinding name re-defers higher each scan. The anchor must stay pinned to
    # where the setup FIRST deferred so the chase ceiling can't crawl up.
    runner = _runner(2.34)
    sym = "BATL"

    # First defer near the base pins the anchor at 1.66.
    first = _signal(symbol=sym, price=1.66)
    assert runner._timed_entry_chase_anchor(first) == 1.66

    # A later re-defer is a brand-new signal at a higher price with NO queued
    # base of its own — the persisted anchor must still win.
    later = _signal(symbol=sym, price=2.30)
    assert runner._timed_entry_chase_anchor(later) == 1.66
    # And it is stamped into criteria so the execution timer sees it too.
    assert later.scan_result.criteria.get("setup_anchor") == 1.66

    # The late entry is rejected, citing the original base — not 2.30.
    reason = runner._timed_entry_chase_reject(later, _bar(symbol=sym, close=2.34))
    assert reason is not None
    assert "1.6600" in reason


def test_timed_entry_chase_anchor_reanchors_after_stale_ttl() -> None:
    # Once a setup goes stale (TTL elapsed) the same symbol re-anchors fresh.
    runner = _runner(3.00)
    runner._timed_entry_anchor_ttl_sec = 0.0  # everything is immediately stale
    sym = "BATL"

    first = _signal(symbol=sym, price=1.66)
    assert runner._timed_entry_chase_anchor(first) == 1.66

    later = _signal(symbol=sym, price=3.00)
    # With a 0s TTL the prior anchor is stale, so a genuinely new setup re-anchors.
    assert runner._timed_entry_chase_anchor(later) == 3.00


def test_timed_entry_chase_guard_rejects_red_10s_release() -> None:
    runner = _runner(5.05)
    red_10s = Bar(
        symbol="DXST",
        open=5.08,
        high=5.09,
        low=5.04,
        close=5.05,
        volume=1000,
        ts=datetime.now(timezone.utc),
        timeframe=Timeframe.SEC_10,
    )
    runner._bar_aggregator = _Agg([red_10s])

    reason = runner._timed_entry_chase_reject(_signal(price=5.00), _bar(close=5.05))

    assert reason == "latest 10s candle turned red during entry wait"


def test_timed_entry_chase_guard_rejects_failed_opening_range_reclaim() -> None:
    runner = _runner(3.11)

    reason = runner._timed_entry_chase_reject(
        _opening_range_signal(),
        _bar(symbol="BGMS", close=3.11),
    )

    assert reason == "live price 3.1100 pulled back too far from breakout signal 3.2600"


def test_timed_entry_chase_guard_allows_opening_range_reclaim() -> None:
    runner = _runner(3.18)

    reason = runner._timed_entry_chase_reject(
        _opening_range_signal(),
        _bar(symbol="BGMS", close=3.18),
    )

    assert reason is None


def test_timed_entry_chase_guard_rejects_lost_level_breakout() -> None:
    runner = _runner(4.08)

    reason = runner._timed_entry_chase_reject(
        _level_breakout_signal(),
        _bar(symbol="DAIC", close=4.08),
    )

    assert reason == "live price 4.0800 lost breakout level 4.1200"


def test_timed_entry_chase_guard_allows_held_level_breakout() -> None:
    runner = _runner(4.20)

    reason = runner._timed_entry_chase_reject(
        _level_breakout_signal(),
        _bar(symbol="DAIC", close=4.20),
    )

    assert reason is None


def test_timed_entry_chase_guard_rejects_extended_level_breakout() -> None:
    runner = _runner(4.30)

    reason = runner._timed_entry_chase_reject(
        _level_breakout_signal(),
        _bar(symbol="DAIC", close=4.30),
    )

    assert reason == "live price 4.3000 too extended from breakout level 4.1200 (max 2.5%)"
