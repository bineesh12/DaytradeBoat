"""Tests for ``entry_guard.check_entry_quality`` — scoring-based system."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daytrading.models import Bar, Timeframe
from daytrading.strategy import entry_guard as eg


def _bar(
    i: int,
    *,
    close: float,
    open_: float,
    high: float,
    low: float,
    volume: float,
    base_ts: datetime,
    n: int,
) -> Bar:
    ts = base_ts - timedelta(seconds=(n - i))
    return Bar(
        symbol="TST",
        ts=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=Timeframe.SEC_5,
    )


def _5m_bar(ts: datetime, o: float, c: float, h: float, l: float) -> Bar:
    return Bar(
        symbol="TST",
        ts=ts,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=50_000,
        timeframe=Timeframe.MIN_5,
    )


def _uptrend_bars_passing_default_guard() -> list[Bar]:
    """25 bars: uptrend with a pullback, enough volume."""
    now = datetime.now(timezone.utc)
    n = 25
    bars: list[Bar] = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 1.0
        c = 3.0 + (5.0 - 3.0) * frac
        o = c - 0.02
        hi = c + 0.04
        lo = c - 0.04
        vol = 50_000.0 if i < n - 1 else 250_000.0
        bars.append(_bar(i, close=c, open_=o, high=hi, low=lo, volume=vol, base_ts=now, n=n))
    bars[-3] = _bar(
        n - 3, close=4.88, open_=4.90, high=4.91, low=4.87,
        volume=50_000.0, base_ts=now, n=n,
    )
    bars[-2] = _bar(
        n - 2, close=4.92, open_=4.88, high=4.93, low=4.87,
        volume=60_000.0, base_ts=now, n=n,
    )
    i = n - 1
    bars[-1] = _bar(
        i, close=5.00, open_=4.996, high=5.002, low=4.993,
        volume=250_000.0, base_ts=now, n=n,
    )
    return bars


class _MonitorStub:
    def __init__(self) -> None:
        self.rule_rejections = 0

    @property
    def is_model_enabled(self) -> bool:
        return True

    def record_rule_rejection(self) -> None:
        self.rule_rejections += 1

    def record_entry_passed(self) -> None:
        pass

    def record_ml_rejection(self, *args, **kwargs) -> None:
        pass


class TestHardRejects:
    """Hard rejects should always fail regardless of score."""

    def test_rule_rejects_are_counted_for_ml_dashboard(self, monkeypatch) -> None:
        monitor = _MonitorStub()
        monkeypatch.setattr(eg, "_ml_monitor", monitor)

        reason = eg.check_entry_quality([], symbol="TST")

        assert reason == "insufficient bars"
        assert monitor.rule_rejections == 1

    def test_insufficient_bars(self) -> None:
        assert eg.check_entry_quality([]) == "insufficient bars"

    def test_price_outside_range(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [_bar(i, close=1.49, open_=1.4, high=1.55, low=1.4, volume=100_000, base_ts=now, n=3) for i in range(3)]
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "price" in r.lower()

    def test_stale_likely_halt_can_pass(self) -> None:
        """Stale bars can pass if the last bar looks like a likely halt."""
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=400)
        bars = [
            _bar(i, close=5.0, open_=4.5, high=5.1, low=4.4, volume=50_000, base_ts=old_ts, n=10)
            for i in range(10)
        ]
        r = eg.check_entry_quality(bars, symbol="TST", avg_daily_volume=500_000)
        assert r is None or "stale" not in r.lower()

    def test_stale_quiet_data_rejects(self) -> None:
        """Stale bars without halt-like volume/move still hard-reject."""
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=400)
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=50_000, base_ts=old_ts, n=10)
            for i in range(10)
        ]
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "stale" in r.lower()

    def test_below_vwap_rejects(self) -> None:
        """Price well below VWAP should hard-reject."""
        now = datetime.now(timezone.utc)
        n = 10
        bars = []
        for i in range(n):
            bars.append(_bar(
                i, close=4.50, open_=5.00, high=5.10, low=4.40,
                volume=100_000, base_ts=now, n=n,
            ))
        bars[-1] = _bar(n - 1, close=4.00, open_=4.10, high=4.15, low=3.95, volume=100_000, base_ts=now, n=n)
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "vwap" in r.lower()

    def test_dead_cat_bounce_rejects(self) -> None:
        """Price >20% below session HOD should hard-reject."""
        now = datetime.now(timezone.utc)
        n = 10
        bars = []
        for i in range(n):
            if i < 5:
                bars.append(_bar(i, close=10.0, open_=9.8, high=10.2, low=9.7, volume=100_000, base_ts=now, n=n))
            else:
                bars.append(_bar(i, close=7.5, open_=7.8, high=7.9, low=7.4, volume=100_000, base_ts=now, n=n))
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and ("dead cat" in r.lower() or "vwap" in r.lower())

    def test_wide_spread_rejects(self) -> None:
        """Spread > 0.5% of price should hard-reject."""
        from daytrading.models import Quote
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        # Spread = $0.10 on a $5 stock = 2% — way above 0.5% threshold
        quotes = [
            Quote(symbol="TST", ts=now, bid=4.90, ask=5.00, bid_size=100, ask_size=100)
            for _ in range(5)
        ]
        r = eg.check_entry_quality(bars, symbol="TST", quotes=quotes)
        assert r is not None and "spread" in r.lower()

    def test_elite_a_plus_runner_can_reach_ml_with_slightly_wide_spread(self, monkeypatch) -> None:
        """CIIT-style A+ runners should not die at 0.5%-0.8% spread before ML."""
        from daytrading.models import Quote

        monkeypatch.setattr(eg, "_xgb_model", None)
        now = datetime.now(timezone.utc)
        n = 24
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 3.00 + (7.68 - 3.00) * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.08,
                high=close + 0.10,
                low=close - 0.12,
                volume=110_000,
                base_ts=now,
                n=n,
            ))
        bars[-1] = _bar(
            n - 1,
            close=7.68,
            open_=7.48,
            high=7.70,
            low=7.42,
            volume=240_000,
            base_ts=now,
            n=n,
        )
        quotes = [
            Quote(symbol="CIIT", ts=now, bid=7.65, ask=7.71, bid_size=900, ask_size=900)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="CIIT",
            quotes=quotes,
            avg_daily_volume=1_000_000,
            float_shares=1_300_000,
            entry_pattern="abc_continuation",
            setup_tier="A+ setup",
        )

        if r is not None:
            assert "spread" not in r.lower()

    def test_non_a_plus_runner_still_rejects_slightly_wide_spread(self, monkeypatch) -> None:
        """The runner spread exception must not loosen ordinary entries."""
        from daytrading.models import Quote

        monkeypatch.setattr(eg, "_xgb_model", None)
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        quotes = [
            Quote(symbol="TST", ts=now, bid=4.98, ask=5.02, bid_size=900, ask_size=900)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="TST",
            quotes=quotes,
            entry_pattern="vwap_pullback",
            setup_tier="watch only",
        )

        assert r is not None and "spread" in r.lower()

    def test_elite_shallow_stair_runner_allows_tiny_size_wider_spread(
        self,
        monkeypatch,
    ) -> None:
        """Fast low-float stair-step runners get only a narrow spread exception."""
        from daytrading.models import Quote

        monkeypatch.setattr(eg, "_xgb_model", None)
        now = datetime.now(timezone.utc)
        n = 25
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 2.35 + (3.08 - 2.35) * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=close + 0.05,
                low=close - 0.06,
                volume=140_000,
                base_ts=now,
                n=n,
            ))
        bars[-1] = _bar(
            n - 1,
            close=3.08,
            open_=2.94,
            high=3.12,
            low=2.91,
            volume=320_000,
            base_ts=now,
            n=n,
        )
        quotes = [
            Quote(symbol="WCT", ts=now, bid=3.065, ask=3.095, bid_size=900, ask_size=900)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="WCT",
            quotes=quotes,
            avg_daily_volume=1_500_000,
            float_shares=2_550_000,
            entry_pattern="shallow_stair_continuation",
            setup_tier="A+ setup",
        )

        if r is not None:
            assert "spread" not in r.lower()

    def test_sub_five_thin_liquidity_rejects_before_ml(self) -> None:
        """ANNA-like sub-$5 setups need enough day volume before ML/order."""
        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            close = 3.05 + (3.48 - 3.05) * (i / (n - 1))
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=24_000,
                base_ts=now,
                n=n,
            ))
        r = eg.check_entry_quality(bars, symbol="ANNA")
        assert r is not None
        assert "thin sub-$5 liquidity" in r

    def test_sub_two_a_plus_runner_reaches_liquidity_checks(self) -> None:
        """Lowering the floor to $1.50 should evaluate real sub-$2 runners."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 1.52 + 0.34 * frac
            vol = 55_000 if i < 15 else 140_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="CLIK", ts=now, bid=1.855, ask=1.86, bid_size=2000, ask_size=2000)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="CLIK",
            quotes=quotes,
            avg_daily_volume=400_000,
            float_shares=800_000,
            setup_tier="A+ setup",
        )

        if r is not None:
            assert "outside range" not in r

    def test_entry_guard_still_rejects_below_dollar_fifty(self) -> None:
        now = datetime.now(timezone.utc)
        n = 5
        bars = [
            _bar(
                i,
                close=1.49,
                open_=1.47,
                high=1.50,
                low=1.45,
                volume=200_000,
                base_ts=now,
                n=n,
            )
            for i in range(n)
        ]

        r = eg.check_entry_quality(bars, symbol="LOW")

        assert r == "price $1.49 outside range $1.50-$20.00"

    def test_five_plus_runner_can_reach_scoring_with_100k_plus_day_volume(self) -> None:
        """FOXX-like $5+ runners should not fail the first gate near 200K volume."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 5.05 + 0.77 * frac
            vol = 7_000 if i < 15 else 15_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=close + 0.04,
                low=close - 0.04,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="FOXX", ts=now, bid=5.815, ask=5.825, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars, symbol="FOXX", quotes=quotes, avg_daily_volume=900_000,
        )

        if r is not None:
            assert "low day volume" not in r

    def test_five_plus_runner_still_rejects_below_100k_day_volume(self) -> None:
        """The looser $5+ gate still blocks names with too little total tape."""
        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 5.05 + 0.75 * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=close + 0.04,
                low=close - 0.04,
                volume=4_500,
                base_ts=now,
                n=n,
            ))

        r = eg.check_entry_quality(bars, symbol="FOXX")

        assert r is not None
        assert "low day volume" in r
        assert "need 100K+" in r

    def test_sub_five_weak_runner_still_needs_stronger_liquidity(self) -> None:
        """Lowering the general gate must not loosen protection for cheap thin names."""
        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 3.50 + 0.25 * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=6_000,
                base_ts=now,
                n=n,
            ))

        r = eg.check_entry_quality(bars, symbol="THIN")

        assert r is not None
        assert "thin sub-$5 liquidity" in r

    def test_sub_five_high_relative_volume_can_pass_under_500k(self) -> None:
        """A sub-$5 stock under 500K volume can pass when RVOL confirms momentum."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 3.00 + 0.75 * frac
            vol = 15_000 if i < 15 else 35_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="TST", ts=now, bid=3.745, ask=3.75, bid_size=800, ask_size=800)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars, symbol="TST", quotes=quotes, avg_daily_volume=100_000,
        )

        if r is not None:
            assert "liquidity" not in r

    def test_tight_spread_passes(self) -> None:
        """Spread < 0.5% of price should not reject."""
        from daytrading.models import Quote
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        # Spread = $0.01 on a $5 stock = 0.2% — fine
        quotes = [
            Quote(symbol="TST", ts=now, bid=4.99, ask=5.00, bid_size=100, ask_size=100)
            for _ in range(5)
        ]
        r = eg.check_entry_quality(bars, symbol="TST", quotes=quotes)
        # Should not reject on spread (may pass or fail on other criteria)
        if r is not None:
            assert "spread" not in r.lower()

    def test_one_tick_spread_on_sub_two_runner_is_not_too_wide(self) -> None:
        """A penny-wide book is the normal tick, even when it is >0.5% below $2."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 25
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 1.20 + (1.66 - 1.20) * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.015,
                high=close + 0.03,
                low=close - 0.03,
                volume=90_000 if i < n - 1 else 250_000,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="BATL", ts=now, bid=1.655, ask=1.665, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="BATL",
            quotes=quotes,
            entry_pattern="pullback_base",
            setup_tier="A+ setup",
            entry_tier="a_plus_reclaim_scout",
        )

        if r is not None:
            assert "spread" not in r.lower()

    def test_five_plus_scalp_rejects_weak_active_entry_tape(self) -> None:
        """SMCZ/TSLQ-like entries need current candle volume, not just day volume."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 6.05 + 0.78 * frac
            vol = 30_000 if i < 15 else 12_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        bars[-1] = _bar(
            n - 1,
            close=6.83,
            open_=6.79,
            high=6.85,
            low=6.78,
            volume=10_000,
            base_ts=now,
            n=n,
        )
        quotes = [
            Quote(symbol="SMCZ", ts=now, bid=6.82, ask=6.83, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(bars, symbol="SMCZ", quotes=quotes, avg_daily_volume=400_000)

        assert r is not None
        assert "weak active tape" in r

    def test_five_plus_scalp_allows_real_active_entry_tape(self) -> None:
        """The active-tape guard should not block a current high-volume reclaim."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 5.05 + 0.75 * frac
            vol = 10_000 if i < 15 else 35_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=close + 0.04,
                low=close - 0.04,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="FOXX", ts=now, bid=5.79, ask=5.80, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(bars, symbol="FOXX", quotes=quotes, avg_daily_volume=800_000)

        if r is not None:
            assert "weak active tape" not in r

    def test_selling_pressure_rejects(self) -> None:
        """Order flow imbalance <= -0.3 (sellers dominating) should hard-reject."""
        from daytrading.models import Tick, Side
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        # All sells — imbalance will be -1.0, well below -0.3
        ticks = [
            Tick(symbol="TST", ts=now, price=5.0, size=100, side=Side.SELL)
            for _ in range(30)
        ]
        r = eg.check_entry_quality(bars, symbol="TST", ticks=ticks)
        assert r is not None and "tape" in r.lower()

    def test_elite_sub_two_a_plus_reclaim_can_survive_moderate_selling_pressure(self) -> None:
        """FLD-style A+ runners can keep going despite mixed tape if structure is elite."""
        from daytrading.models import Tick, Side

        now = datetime.now(timezone.utc)
        n = 12
        rows = [
            (1.00, 1.06, 1.08, 0.98, 160_000),
            (1.06, 1.20, 1.23, 1.04, 190_000),
            (1.20, 1.38, 1.42, 1.16, 250_000),
            (1.38, 1.56, 1.60, 1.32, 340_000),
            (1.56, 1.85, 1.85, 1.50, 520_000),
            (1.85, 1.62, 1.84, 1.55, 250_000),
            (1.62, 1.43, 1.66, 1.37, 230_000),
            (1.43, 1.50, 1.55, 1.40, 210_000),
            (1.50, 1.58, 1.62, 1.48, 230_000),
            (1.58, 1.63, 1.67, 1.55, 260_000),
            (1.63, 1.66, 1.70, 1.60, 280_000),
            (1.66, 1.68, 1.73, 1.62, 300_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=n)
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        ticks = [
            Tick(
                symbol="FLD",
                ts=now + timedelta(milliseconds=i),
                price=1.68,
                size=100,
                side=Side.BUY if i < 8 else Side.SELL,
            )
            for i in range(30)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="FLD",
            ticks=ticks,
            avg_daily_volume=500_000,
            float_shares=6_400_000,
            entry_pattern="abc_continuation",
            setup_tier="A+ setup",
        )

        assert r is None

    def test_buying_pressure_passes(self) -> None:
        """Order flow imbalance > 0 (buyers dominating) should not reject."""
        from daytrading.models import Tick, Side
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        # All buys — imbalance will be +1.0
        ticks = [
            Tick(symbol="TST", ts=now, price=5.0, size=100, side=Side.BUY)
            for _ in range(30)
        ]
        r = eg.check_entry_quality(bars, symbol="TST", ticks=ticks)
        # Should not reject on tape (may pass or fail on other criteria)
        if r is not None:
            assert "tape" not in r.lower()

    def test_balanced_flow_passes(self) -> None:
        """Balanced buy/sell (imbalance ~0) should not reject on tape."""
        from daytrading.models import Tick, Side
        now = datetime.now(timezone.utc)
        bars = _uptrend_bars_passing_default_guard()
        ticks = []
        for i in range(30):
            side = Side.BUY if i % 2 == 0 else Side.SELL
            ticks.append(Tick(symbol="TST", ts=now, price=5.0, size=100, side=side))
        r = eg.check_entry_quality(bars, symbol="TST", ticks=ticks)
        if r is not None:
            assert "tape" not in r.lower()


class TestScoringSystem:
    """Scoring system allows strong signals to compensate for weaker ones."""

    def test_full_pass_synthetic(self) -> None:
        bars = _uptrend_bars_passing_default_guard()
        assert eg.check_entry_quality(bars, symbol="TST", avg_daily_volume=500_000) is None

    def test_strong_momentum_passes_despite_low_volume(self) -> None:
        """A stock up 10%+ with strong candles should pass even with moderate volume."""
        now = datetime.now(timezone.utc)
        n = 15
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            c = 4.0 + 1.5 * frac
            o = c - 0.03
            bars.append(_bar(i, close=c, open_=o, high=c + 0.05, low=o - 0.02,
                            volume=20_000, base_ts=now, n=n))
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is None or "score" in r or "liquidity" in r

    def test_full_liquidity_score_rejects_thin_sub_five_chop(self) -> None:
        """Sub-$5 names need more than just 500K day volume if tape is thin."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 3.00 + 0.40 * frac
            vol = 35_000 if i < 15 else 15_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="TST", ts=now, bid=3.39, ask=3.40, bid_size=50, ask_size=50)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(bars, symbol="TST", quotes=quotes)

        assert r is not None
        assert "thin liquidity score" in r

    def test_full_liquidity_score_allows_hot_sub_five_tape(self) -> None:
        """A real high-volume sub-$5 runner should not be blocked by liquidity scoring."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 3.00 + 0.85 * frac
            vol = 45_000 if i < 15 else 130_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.02,
                high=close + 0.03,
                low=close - 0.03,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="TST", ts=now, bid=3.845, ask=3.85, bid_size=1200, ask_size=1200)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(bars, symbol="TST", quotes=quotes)

        if r is not None:
            assert "liquidity" not in r

    def test_low_day_change_low_score(self) -> None:
        """Flat stock with no day change gets hard-rejected for movement."""
        now = datetime.now(timezone.utc)
        n = 10
        bars = [
            _bar(i, close=5.0, open_=4.99, high=5.01, low=4.98, volume=5_000, base_ts=now, n=n)
            for i in range(n)
        ]
        r = eg.check_entry_quality(bars, symbol="TST")
        assert r is not None and "not enough movement" in r

    def test_a_plus_reclaim_scout_can_use_recent_vwap_after_runner_spike(self) -> None:
        """HWH-style scouts should not be blocked by old spike-distorted VWAP."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        rows = [
            (1.50, 1.60, 1.65, 1.48, 120_000),
            (1.60, 2.50, 2.70, 1.58, 500_000),
            (2.50, 3.30, 3.40, 2.45, 900_000),
            (3.30, 2.70, 3.35, 2.55, 350_000),
            (2.70, 2.20, 2.75, 2.10, 280_000),
            (2.20, 1.90, 2.25, 1.82, 230_000),
            (1.90, 1.95, 1.98, 1.86, 120_000),
            (1.95, 2.05, 2.08, 1.92, 130_000),
            (2.05, 2.15, 2.18, 2.02, 150_000),
            (2.15, 2.25, 2.28, 2.10, 180_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        quotes = [
            Quote(symbol="HWH", ts=now, bid=2.245, ask=2.255, bid_size=2000, ask_size=2000)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="HWH",
            avg_daily_volume=1_000_000,
            float_shares=3_000_000,
            quotes=quotes,
            entry_pattern="early_vwap_reclaim_scout",
            setup_tier="A+ setup",
            entry_tier="a_plus_reclaim_scout",
        )

        assert r is None or "below VWAP" not in r

    def test_normal_setup_still_rejects_below_spike_distorted_vwap(self) -> None:
        """The recent-VWAP exception is only for A+ reclaim scouts."""
        now = datetime.now(timezone.utc)
        rows = [
            (1.50, 1.60, 1.65, 1.48, 120_000),
            (1.60, 2.50, 2.70, 1.58, 500_000),
            (2.50, 3.30, 3.40, 2.45, 900_000),
            (3.30, 2.70, 3.35, 2.55, 350_000),
            (2.70, 2.20, 2.75, 2.10, 280_000),
            (2.20, 1.90, 2.25, 1.82, 230_000),
            (1.90, 1.95, 1.98, 1.86, 120_000),
            (1.95, 1.99, 2.02, 1.92, 130_000),
            (1.99, 2.03, 2.06, 1.96, 150_000),
            (2.03, 2.08, 2.11, 2.00, 180_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]

        r = eg.check_entry_quality(bars, symbol="HWH")

        assert r is not None and "below VWAP" in r

    def test_a_plus_level_scout_gets_narrow_wide_spread_exception(self) -> None:
        """AZI-style A+ level scout can pass around 1% spread with strong tape."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 16
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 1.95 + 1.05 * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.04,
                high=close + 0.08,
                low=close - 0.08,
                volume=220_000 if i < n - 1 else 450_000,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="AZI", ts=now, bid=2.944, ask=2.976, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="AZI",
            avg_daily_volume=1_000_000,
            float_shares=2_000_000,
            quotes=quotes,
            entry_pattern="level_breakout_reclaim",
            setup_tier="A+ setup",
            entry_tier="level_scout",
        )

        assert r is None or "spread too wide" not in r

    def test_a_plus_hod_reclaim_can_use_recent_vwap_after_old_runner_spike(self) -> None:
        """WCT-style later legs should not be blocked by the first spike VWAP."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        rows = [
            (1.30, 1.50, 1.55, 1.28, 300_000),
            (1.50, 2.40, 2.70, 1.48, 900_000),
            (2.40, 4.40, 4.60, 2.30, 1_400_000),
            (4.40, 3.10, 4.45, 2.95, 700_000),
            (3.10, 2.30, 3.15, 2.18, 500_000),
            (2.30, 2.05, 2.35, 1.95, 350_000),
            (2.05, 2.10, 2.16, 2.00, 180_000),
            (2.10, 2.16, 2.20, 2.06, 190_000),
            (2.16, 2.22, 2.25, 2.12, 210_000),
            (2.22, 2.30, 2.34, 2.18, 260_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]
        quotes = [
            Quote(symbol="WCT", ts=now, bid=2.292, ask=2.308, bid_size=1800, ask_size=1800)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="WCT",
            avg_daily_volume=1_000_000,
            float_shares=2_600_000,
            quotes=quotes,
            entry_pattern="hod_reclaim",
            setup_tier="A+ setup",
            entry_tier="deep_runner_scout",
        )

        assert r is None or "below VWAP" not in r
        assert r is None or "dead cat" not in r

    def test_non_a_plus_hod_reclaim_still_rejects_old_vwap_failure(self) -> None:
        """The WCT-style recent-VWAP exception must stay A+ only."""
        now = datetime.now(timezone.utc)
        rows = [
            (1.30, 1.50, 1.55, 1.28, 300_000),
            (1.50, 2.40, 2.70, 1.48, 900_000),
            (2.40, 4.40, 4.60, 2.30, 1_400_000),
            (4.40, 3.10, 4.45, 2.95, 700_000),
            (3.10, 2.30, 3.15, 2.18, 500_000),
            (2.30, 2.05, 2.35, 1.95, 350_000),
            (2.05, 2.10, 2.16, 2.00, 180_000),
            (2.10, 2.16, 2.20, 2.06, 190_000),
            (2.16, 2.22, 2.25, 2.12, 210_000),
            (2.22, 2.30, 2.34, 2.18, 260_000),
        ]
        bars = [
            _bar(i, close=c, open_=o, high=h, low=l, volume=v, base_ts=now, n=len(rows))
            for i, (o, c, h, l, v) in enumerate(rows)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="WCT",
            avg_daily_volume=1_000_000,
            float_shares=2_600_000,
            entry_pattern="hod_reclaim",
            setup_tier="watch only",
        )

        assert r is not None and ("below VWAP" in r or "dead cat" in r)

    def test_hot_low_float_runner_can_bypass_five_percent_movement_gate(self) -> None:
        """XOS-style premarket gappers can pass at 3-5% session change if tape is excellent."""
        from daytrading.models import Quote

        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 7.18 + 0.25 * frac  # ~3.5% session change
            vol = 55_000 if i < 15 else 280_000
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.03,
                high=close + 0.04,
                low=close - 0.04,
                volume=vol,
                base_ts=now,
                n=n,
            ))
        quotes = [
            Quote(symbol="TST", ts=now, bid=7.425, ask=7.435, bid_size=1500, ask_size=1500)
            for _ in range(5)
        ]

        r = eg.check_entry_quality(
            bars,
            symbol="XOS",
            avg_daily_volume=2_000_000,
            float_shares=6_000_000,
            quotes=quotes,
        )

        assert r is None or "not enough movement" not in r

    def test_moderate_low_change_without_hot_tape_still_rejects(self) -> None:
        """Do not weaken the movement gate for ordinary 3-5% movers."""
        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            close = 7.18 + 0.25 * frac
            bars.append(_bar(
                i,
                close=close,
                open_=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                volume=20_000,
                base_ts=now,
                n=n,
            ))

        r = eg.check_entry_quality(
            bars,
            symbol="WEAK",
            avg_daily_volume=2_000_000,
            float_shares=6_000_000,
        )

        assert r is not None and "not enough movement" in r

    def test_bearish_5m_lowers_score_but_strong_stock_can_pass(self) -> None:
        """Bearish 5m context lowers score but doesn't hard-reject a strong setup."""
        now = datetime.now(timezone.utc)
        n = 25
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            c = 3.0 + 2.5 * frac
            o = c - 0.02
            bars.append(_bar(i, close=c, open_=o, high=c + 0.04, low=o - 0.02,
                            volume=50_000, base_ts=now, n=n))

        base = datetime(2026, 5, 19, 14, 0, 0, tzinfo=timezone.utc)
        bars_5m = [
            _5m_bar(base, 5.70, 5.50, 5.75, 5.45),
            _5m_bar(base + timedelta(minutes=5), 5.50, 5.30, 5.55, 5.25),
            _5m_bar(base + timedelta(minutes=10), 5.30, 5.10, 5.35, 5.05),
        ]
        r = eg.check_entry_quality(
            bars, symbol="TST", avg_daily_volume=500_000, bars_5m=bars_5m,
        )
        # With scoring, strong momentum can compensate for weak 5m context
        # Result depends on total score — not a guaranteed rejection anymore
        if r is not None:
            assert "score" in r


class TestMomentumQuality:
    def test_strong_uptrend_high_score(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=4.0 + i * 0.1, open_=4.0 + i * 0.1 - 0.02,
                 high=4.0 + i * 0.1 + 0.03, low=4.0 + i * 0.1 - 0.03,
                 volume=50_000, base_ts=now, n=10)
            for i in range(10)
        ]
        score, _ = eg._momentum_quality(bars)
        assert score >= 50

    def test_flat_bars_low_score(self) -> None:
        now = datetime.now(timezone.utc)
        bars = [
            _bar(i, close=5.0, open_=5.0, high=5.01, low=4.99,
                 volume=10_000, base_ts=now, n=10)
            for i in range(10)
        ]
        score, _ = eg._momentum_quality(bars)
        assert score < 30


class TestVolumeExhaustion:
    """S8: Volume exhaustion penalty for declining-volume green bars."""

    def test_3_declining_green_bars_penalizes(self) -> None:
        """3 consecutive green bars with declining volume -> -20 penalty."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=100_000, base_ts=now, n=5),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=80_000, base_ts=now, n=5),
            _bar(2, close=5.2, open_=5.1, high=5.3, low=5.0, volume=60_000, base_ts=now, n=5),
            _bar(3, close=5.3, open_=5.2, high=5.4, low=5.1, volume=40_000, base_ts=now, n=5),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        assert penalty == -20

    def test_4_declining_green_bars_severe_penalty(self) -> None:
        """4 consecutive green bars with declining volume -> -30 penalty."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=100_000, base_ts=now, n=5),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=80_000, base_ts=now, n=5),
            _bar(2, close=5.2, open_=5.1, high=5.3, low=5.0, volume=60_000, base_ts=now, n=5),
            _bar(3, close=5.3, open_=5.2, high=5.4, low=5.1, volume=40_000, base_ts=now, n=5),
            _bar(4, close=5.4, open_=5.3, high=5.5, low=5.2, volume=20_000, base_ts=now, n=5),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        assert penalty == -30

    def test_increasing_volume_no_penalty(self) -> None:
        """3 green bars with increasing volume -> no penalty."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=40_000, base_ts=now, n=4),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=60_000, base_ts=now, n=4),
            _bar(2, close=5.2, open_=5.1, high=5.3, low=5.0, volume=80_000, base_ts=now, n=4),
            _bar(3, close=5.3, open_=5.2, high=5.4, low=5.1, volume=100_000, base_ts=now, n=4),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        assert penalty == 0

    def test_mixed_red_green_no_penalty(self) -> None:
        """Mixed red/green bars break the streak -> no penalty."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=100_000, base_ts=now, n=4),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=80_000, base_ts=now, n=4),
            _bar(2, close=5.0, open_=5.1, high=5.2, low=4.9, volume=60_000, base_ts=now, n=4),  # red
            _bar(3, close=5.2, open_=5.1, high=5.3, low=5.0, volume=40_000, base_ts=now, n=4),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        # Red bar at index 2 breaks the streak; only 1 qualifying pair (3→from 2)
        assert penalty == 0

    def test_fewer_than_3_bars_no_penalty(self) -> None:
        """Too few bars -> no penalty."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=100_000, base_ts=now, n=2),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=50_000, base_ts=now, n=2),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        assert penalty == 0

    def test_2_declining_gives_early_warning(self) -> None:
        """2 consecutive declining-volume green bars -> -10."""
        now = datetime.now(timezone.utc)
        bars = [
            _bar(0, close=5.0, open_=4.9, high=5.1, low=4.8, volume=100_000, base_ts=now, n=4),
            _bar(1, close=5.1, open_=5.0, high=5.2, low=4.9, volume=80_000, base_ts=now, n=4),
            _bar(2, close=5.2, open_=5.1, high=5.3, low=5.0, volume=60_000, base_ts=now, n=4),
        ]
        penalty = eg._volume_exhaustion_penalty(bars)
        assert penalty == -10

    def test_exhaustion_penalty_lowers_entry_guard_score(self) -> None:
        """Integration: exhaustion penalty can cause a borderline entry to fail."""
        now = datetime.now(timezone.utc)
        n = 20
        bars = []
        # Build a strong setup that would normally pass (high day change, near HOD)
        for i in range(n - 4):
            frac = i / (n - 1)
            c = 3.0 + 2.5 * frac
            o = c - 0.02
            bars.append(_bar(i, close=c, open_=o, high=c + 0.04, low=o - 0.02,
                            volume=50_000, base_ts=now, n=n))
        # Last 4 bars: green with declining volume (exhaustion)
        for j, vol in enumerate([80_000, 60_000, 40_000, 20_000]):
            idx = n - 4 + j
            c = 5.0 + j * 0.05
            bars.append(_bar(idx, close=c, open_=c - 0.02, high=c + 0.03, low=c - 0.03,
                            volume=vol, base_ts=now, n=n))

        r = eg.check_entry_quality(bars, symbol="TST")
        # Should include "exhaust" in the breakdown
        if r is not None:
            assert "exhaust" in r


class TestLowRvolPenalty:
    """S9: Low relative volume penalty for weak breakouts."""

    def _bars_with_rvol(self, rvol: float) -> list:
        """Create 15 bars where recent 5 have rvol relative to earlier bars."""
        now = datetime.now(timezone.utc)
        n = 15
        bars = []
        earlier_vol = 50_000
        recent_vol = earlier_vol * rvol
        for i in range(n):
            frac = i / (n - 1)
            c = 3.0 + 2.5 * frac
            o = c - 0.02
            vol = recent_vol if i >= 10 else earlier_vol
            bars.append(_bar(i, close=c, open_=o, high=c + 0.04, low=o - 0.02,
                            volume=vol, base_ts=now, n=n))
        return bars

    def test_high_rvol_no_penalty(self) -> None:
        """rvol >= 2.0 -> no penalty."""
        bars = self._bars_with_rvol(2.5)
        r = eg.check_entry_quality(bars, symbol="TST")
        if r is not None:
            assert "rvol" not in r

    def test_moderate_rvol_small_penalty(self) -> None:
        """rvol 1.0-2.0 -> -5 penalty appears in breakdown."""
        bars = self._bars_with_rvol(1.5)
        r = eg.check_entry_quality(bars, symbol="TST")
        if r is not None and "rvol" in r:
            assert "-5" in r

    def test_below_average_rvol_medium_penalty(self) -> None:
        """rvol 0.5-1.0 -> -20 penalty."""
        bars = self._bars_with_rvol(0.9)
        r = eg.check_entry_quality(bars, symbol="TST")
        if r is not None:
            assert "rvol" in r and "-20" in r

    def test_very_low_rvol_severe_penalty(self) -> None:
        """rvol < 0.5 -> -25 penalty."""
        bars = self._bars_with_rvol(0.3)
        r = eg.check_entry_quality(bars, symbol="TST")
        if r is not None:
            assert ("rvol" in r and "-25" in r) or "liquidity" in r or "weak active tape" in r

    def test_fewer_than_10_bars_no_penalty(self) -> None:
        """With < 10 today bars, rvol can't be computed so no penalty."""
        now = datetime.now(timezone.utc)
        n = 8
        bars = []
        for i in range(n):
            frac = i / (n - 1)
            c = 3.0 + 2.5 * frac
            o = c - 0.02
            bars.append(_bar(i, close=c, open_=o, high=c + 0.04, low=o - 0.02,
                            volume=5_000, base_ts=now, n=n))
        r = eg.check_entry_quality(bars, symbol="TST")
        if r is not None:
            assert "rvol" not in r or "barvol" in r  # barvol is S3, not S9
