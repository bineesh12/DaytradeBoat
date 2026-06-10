from __future__ import annotations

from types import SimpleNamespace

from daytrading.data.watchlist_scanner import WatchlistScanner


class _TradingStub:
    def __init__(self, assets: list[SimpleNamespace]) -> None:
        self.assets = assets

    def get_all_assets(self, _request: object) -> list[SimpleNamespace]:
        return self.assets


def _asset(symbol: str, *, exchange: str = "NASDAQ", tradable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, exchange=exchange, tradable=tradable)


def test_symbol_universe_keeps_short_real_tickers_ending_in_w() -> None:
    scanner = WatchlistScanner.__new__(WatchlistScanner)
    scanner._all_symbols = []
    scanner._symbols_loaded_at = 0.0
    scanner._trading = _TradingStub([
        _asset("WNW"),
        _asset("ABCDW"),
        _asset("MASK"),
        _asset("SOXL"),
        _asset("OTC", exchange="OTC"),
        _asset("HALT", tradable=False),
    ])

    symbols = scanner._load_all_symbols()

    assert "WNW" in symbols
    assert "MASK" in symbols
    assert "ABCDW" not in symbols
    assert "SOXL" not in symbols
    assert "OTC" not in symbols
    assert "HALT" not in symbols


def _scanner_for_extract(*, premarket: bool) -> WatchlistScanner:
    scanner = WatchlistScanner.__new__(WatchlistScanner)
    scanner._is_premarket = premarket
    scanner._min_price = 1.0
    scanner._max_price = 20.0
    scanner._min_volume = 1_000_000
    scanner._premarket_min_volume = 10_000
    scanner._min_change = 0.0
    return scanner


def _snapshot(
    *,
    daily_close: float,
    daily_volume: float,
    previous_close: float,
    latest_price: float,
    minute_volume: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        daily_bar=SimpleNamespace(close=daily_close, volume=daily_volume),
        previous_daily_bar=SimpleNamespace(close=previous_close),
        latest_trade=SimpleNamespace(price=latest_price),
        minute_bar=SimpleNamespace(volume=minute_volume),
    )


def test_premarket_extract_uses_yesterday_close_not_two_day_old_close() -> None:
    scanner = _scanner_for_extract(premarket=True)
    snap = _snapshot(
        daily_close=2.57,
        daily_volume=85_221_933,
        previous_close=0.92,
        latest_price=2.04,
        minute_volume=25_000,
    )

    row = scanner._extract("WCT", snap)

    assert row is not None
    assert row["prev_close"] == 2.57
    assert row["change_pct"] == -20.62
    assert row["abs_change_pct"] == 20.62
    assert row["volume"] == 25_000


def test_premarket_extract_does_not_use_stale_daily_volume() -> None:
    scanner = _scanner_for_extract(premarket=True)
    snap = _snapshot(
        daily_close=2.57,
        daily_volume=85_221_933,
        previous_close=0.92,
        latest_price=2.04,
        minute_volume=5_000,
    )

    row = scanner._extract("WCT", snap)

    assert row is None
