from __future__ import annotations

from datetime import datetime, timezone

from daytrading.scanner.premarket_gap import PremarketGapScanner
from daytrading.scanner.volume_spike import VolumeSpikeScanner
from daytrading.scanner.vwap_deviation import VWAPDeviationScanner
from daytrading.scanner.composite import CompositeScanner
from daytrading.models import Bar

TS = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)


def _bar(
    symbol: str, close: float, volume: float = 200_000,
    open_: float | None = None, high: float | None = None, low: float | None = None,
) -> Bar:
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 0.5
    lo = low if low is not None else close - 0.5
    return Bar(symbol=symbol, ts=TS, open=o, high=h, low=lo, close=close, volume=volume)


# ---------- PremarketGapScanner ----------

def test_gap_scanner_detects_gap_up() -> None:
    scanner = PremarketGapScanner(min_gap_pct=3.0, min_volume=100_000)
    bars = [_bar("AAPL", 100.0), _bar("AAPL", 110.0, open_=105.0)]
    hits = scanner.scan({"AAPL": bars})
    assert len(hits) == 1
    assert hits[0].symbol == "AAPL"
    assert hits[0].criteria["gap_pct"] > 0


def test_gap_scanner_skips_small_gap() -> None:
    scanner = PremarketGapScanner(min_gap_pct=5.0)
    bars = [_bar("X", 100.0), _bar("X", 103.0, open_=102.0)]
    hits = scanner.scan({"X": bars})
    assert len(hits) == 0


# ---------- VolumeSpikeScanner ----------

def test_volume_spike_detects() -> None:
    base = [_bar("SPY", 400.0, volume=100_000) for _ in range(21)]
    spike = _bar("SPY", 401.0, volume=500_000)
    bars = base + [spike]
    scanner = VolumeSpikeScanner(min_rvol=2.0, lookback=20, min_avg_volume=50_000)
    hits = scanner.scan({"SPY": bars})
    assert len(hits) == 1
    assert hits[0].criteria["rvol"] >= 2.0


def test_volume_spike_skips_low_rvol() -> None:
    bars = [_bar("SPY", 400.0, volume=100_000) for _ in range(22)]
    scanner = VolumeSpikeScanner(min_rvol=3.0, lookback=20)
    hits = scanner.scan({"SPY": bars})
    assert len(hits) == 0


# ---------- VWAPDeviationScanner ----------

def test_vwap_deviation_above() -> None:
    bars = [
        _bar("TSLA", 200.0, volume=100_000, high=201.0, low=199.0),
        _bar("TSLA", 210.0, volume=200_000, high=211.0, low=209.0),
    ]
    scanner = VWAPDeviationScanner(min_dev_pct=1.0, min_volume=50_000)
    hits = scanner.scan({"TSLA": bars})
    assert len(hits) == 1
    assert hits[0].criteria["direction"] == "above"


# ---------- CompositeScanner ----------

def test_composite_merges() -> None:
    gap = PremarketGapScanner(min_gap_pct=3.0, min_volume=50_000)
    vol = VolumeSpikeScanner(min_rvol=2.0, lookback=2, min_avg_volume=50_000)

    bars = [
        _bar("NVDA", 100.0, volume=100_000),
        _bar("NVDA", 100.0, volume=100_000),
        _bar("NVDA", 110.0, open_=105.0, volume=300_000),
    ]
    composite = CompositeScanner([gap, vol])
    hits = composite.scan({"NVDA": bars})
    assert len(hits) == 1
    assert hits[0].scanner_name == "composite"
    assert hits[0].criteria["scanner_count"] >= 1
