"""Tests for market regime classifier and adaptive router."""

from __future__ import annotations

from datetime import datetime, timezone

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.classifier.router import AdaptiveRouter, StyleConfig
from daytrading.scanner.scalping.momentum_burst import MomentumBurstScanner
from daytrading.strategy.scalping.momentum_scalp import MomentumScalpVerifier
from daytrading.models import Bar, Quote, TradingStyle

TS = datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc)


def _bar(
    symbol: str, close: float, volume: float = 200_000,
    high: float | None = None, low: float | None = None,
) -> Bar:
    h = high if high is not None else close + 0.02
    lo = low if low is not None else close - 0.02
    return Bar(symbol=symbol, ts=TS, open=close - 0.01, high=h, low=lo, close=close, volume=volume)


def _quote(symbol: str, bid: float, ask: float) -> Quote:
    return Quote(symbol=symbol, ts=TS, bid=bid, ask=ask, bid_size=5000, ask_size=5000)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestClassifier:

    def test_scalping_on_5_dollar_stock(self) -> None:
        """$5 stock with tight spread, good volume → scalping."""
        classifier = MarketRegimeClassifier()
        bars = [_bar("LOW", 5.0 + i * 0.001, volume=200_000, high=5.0 + i * 0.001 + 0.02, low=5.0 + i * 0.001 - 0.02) for i in range(30)]
        quotes = [_quote("LOW", 5.029, 5.030) for _ in range(25)]
        regime = classifier.classify("LOW", bars, quotes)
        assert regime.style == TradingStyle.SCALPING
        assert regime.confidence > 0.3

    def test_rejects_expensive_stock(self) -> None:
        """$450 stock → not tradeable (above $20 max)."""
        classifier = MarketRegimeClassifier()
        bars = [_bar("SPY", 450.0 + i * 0.01, volume=1_000_000) for i in range(30)]
        regime = classifier.classify("SPY", bars)
        assert regime.style == TradingStyle.NOT_TRADEABLE
        assert "above maximum" in regime.reasons[0]

    def test_rejects_penny_stock(self) -> None:
        """$0.50 stock → not tradeable (below $1 min)."""
        classifier = MarketRegimeClassifier()
        bars = [_bar("JUNK", 0.50, volume=500, high=0.55, low=0.45) for _ in range(30)]
        regime = classifier.classify("JUNK", bars)
        assert regime.style == TradingStyle.NOT_TRADEABLE

    def test_day_trading_disabled_by_default(self) -> None:
        """With only scalping enabled, a day-trading candidate gets SCALPING or NOT_TRADEABLE."""
        classifier = MarketRegimeClassifier()  # day_trading disabled
        bars = [_bar("MID", 10.0 + i * 0.1, volume=300_000, high=10.0 + i * 0.1 + 0.5, low=10.0 + i * 0.1 - 0.3) for i in range(30)]
        regime = classifier.classify("MID", bars)
        assert regime.style in (TradingStyle.SCALPING, TradingStyle.NOT_TRADEABLE)
        assert regime.style != TradingStyle.DAY_TRADING

    def test_illiquid_stock_is_not_tradeable(self) -> None:
        """Low volume stock in price range → not tradeable due to liquidity."""
        classifier = MarketRegimeClassifier()
        bars = [_bar("THIN", 5.0, volume=100) for _ in range(30)]
        regime = classifier.classify("THIN", bars)
        assert regime.style == TradingStyle.NOT_TRADEABLE

    def test_insufficient_data(self) -> None:
        classifier = MarketRegimeClassifier()
        bars = [_bar("X", 10.0)]
        regime = classifier.classify("X", bars)
        assert regime.style == TradingStyle.NOT_TRADEABLE

    def test_classify_universe(self) -> None:
        classifier = MarketRegimeClassifier()
        universe = {
            "GOOD": [_bar("GOOD", 8.0 + i * 0.001, volume=200_000) for i in range(30)],
            "JUNK": [_bar("JUNK", 0.50, volume=100) for _ in range(30)],
        }
        regimes = classifier.classify_universe(universe)
        assert len(regimes) == 2
        assert regimes["JUNK"].style == TradingStyle.NOT_TRADEABLE

    def test_regime_has_metrics(self) -> None:
        classifier = MarketRegimeClassifier()
        bars = [_bar("QQ", 15.0 + i * 0.01, volume=200_000) for i in range(30)]
        regime = classifier.classify("QQ", bars)
        assert regime.volatility_pct >= 0
        assert regime.relative_volume >= 0
        assert 0 <= regime.trend_strength <= 1
        assert 0 <= regime.liquidity_score <= 1
        assert len(regime.reasons) > 0

    def test_enable_day_trading(self) -> None:
        """When day trading is enabled, a volatile stock can classify as day_trading."""
        classifier = MarketRegimeClassifier(enable_day_trading=True, max_price=20.0)
        bars = [_bar("VOL", 12.0 + i * 0.15, volume=300_000, high=12.0 + i * 0.15 + 0.5, low=12.0 + i * 0.15 - 0.3) for i in range(30)]
        regime = classifier.classify("VOL", bars)
        assert regime.style != TradingStyle.NOT_TRADEABLE


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouter:

    def _make_router(self) -> AdaptiveRouter:
        classifier = MarketRegimeClassifier()  # $1-$20, scalping only
        configs = {
            TradingStyle.SCALPING: StyleConfig(
                scanners=[MomentumBurstScanner(min_burst_pct=0.05, min_velocity=0.001, min_volume=100)],
                verifiers={"momentum_burst": MomentumScalpVerifier(position_size=500)},
            ),
        }
        return AdaptiveRouter(classifier, configs)

    def test_routes_5_dollar_stock_to_scalping(self) -> None:
        router = self._make_router()
        universe = {
            "LOW": [_bar("LOW", 5.0 + i * 0.001, volume=200_000,
                         high=5.0 + i * 0.001 + 0.02,
                         low=5.0 + i * 0.001 - 0.02) for i in range(30)],
        }
        quotes = {"LOW": [_quote("LOW", 5.029, 5.030) for _ in range(25)]}
        result = router.route(universe, quotes)
        assert "LOW" in result.regimes
        assert result.regimes["LOW"].style == TradingStyle.SCALPING
        assert TradingStyle.SCALPING in result.groups

    def test_skips_expensive_stock(self) -> None:
        router = self._make_router()
        universe = {
            "SPY": [_bar("SPY", 450.0, volume=1_000_000) for _ in range(30)],
        }
        result = router.route(universe)
        assert "SPY" in result.skipped

    def test_skips_penny_stock(self) -> None:
        router = self._make_router()
        universe = {
            "PENNY": [_bar("PENNY", 0.10, volume=50) for _ in range(30)],
        }
        result = router.route(universe)
        assert "PENNY" in result.skipped

    def test_multi_symbol_routing(self) -> None:
        router = self._make_router()
        universe = {
            "GOOD": [_bar("GOOD", 8.0 + i * 0.001, volume=200_000) for i in range(30)],
            "TOOHI": [_bar("TOOHI", 50.0, volume=500_000) for _ in range(30)],
            "DEAD": [_bar("DEAD", 0.01, volume=10) for _ in range(30)],
        }
        result = router.route(universe)
        assert "TOOHI" in result.skipped
        assert "DEAD" in result.skipped
        assert result.regimes["GOOD"].style != TradingStyle.NOT_TRADEABLE

    def test_get_config(self) -> None:
        router = self._make_router()
        cfg = router.get_config(TradingStyle.SCALPING)
        assert cfg is not None
        assert len(cfg.scanners) > 0
        assert router.get_config(TradingStyle.SWING) is None
