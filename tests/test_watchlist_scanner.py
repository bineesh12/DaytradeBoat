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
