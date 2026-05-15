"""Adaptive router — bridges the classifier and the pipeline.

Flow:
  1. Classifier labels each symbol: SCALPING / DAY_TRADING / SWING / NOT_TRADEABLE
  2. Router groups symbols by style
  3. Router selects the correct scanner set and verifier set per group
  4. Pipeline runs only the relevant scanners against each group

This means if AAPL is classified as scalping-grade and TSLA as day-trading-grade,
each one gets analyzed by the scanners/strategies that actually fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from daytrading.classifier.regime import MarketRegimeClassifier
from daytrading.scanner.base import Scanner
from daytrading.strategy.verifier import StrategyVerifier
from daytrading.models import Bar, MarketRegime, Quote, TradingStyle

logger = logging.getLogger(__name__)


@dataclass
class StyleConfig:
    """Scanners and verifiers to use for a given trading style."""
    scanners: List[Scanner] = field(default_factory=list)
    verifiers: Dict[str, StrategyVerifier] = field(default_factory=dict)


@dataclass
class RouteResult:
    """Output of the router: which symbols go to which style pipeline."""
    regimes: Dict[str, MarketRegime] = field(default_factory=dict)
    groups: Dict[TradingStyle, Dict[str, Sequence[Bar]]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)


class AdaptiveRouter:
    """Classifies the universe and routes each symbol to the right tools."""

    def __init__(
        self,
        classifier: MarketRegimeClassifier,
        style_configs: Dict[TradingStyle, StyleConfig],
    ) -> None:
        self._classifier = classifier
        self._configs = style_configs
        self._last_status: Dict[str, str] = {}

    def route(
        self,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
    ) -> RouteResult:
        """Classify every symbol and group by trading style."""

        result = RouteResult()

        for symbol, bars in universe.items():
            q = quotes.get(symbol) if quotes else None
            regime = self._classifier.classify(symbol, bars, q)
            result.regimes[symbol] = regime

            status_key = f"{regime.style.value}:{regime.confidence:.2f}"

            if regime.style is TradingStyle.NOT_TRADEABLE:
                result.skipped.append(symbol)
                if self._last_status.get(symbol) != status_key:
                    logger.info(
                        "SKIP %s — not tradeable (confidence=%.2f): %s",
                        symbol, regime.confidence, "; ".join(regime.reasons),
                    )
                    self._last_status[symbol] = status_key
                continue

            group = result.groups.setdefault(regime.style, {})
            group[symbol] = bars

            if self._last_status.get(symbol) != status_key:
                logger.info(
                    "ROUTE %s -> %s (confidence=%.0f%%, vol=%.2f%%, spread=%.3f%%, "
                    "trend=%.2f, rvol=%.1fx, liq=%.2f): %s",
                    symbol, regime.style.value, regime.confidence * 100,
                    regime.volatility_pct, regime.spread_pct,
                    regime.trend_strength, regime.relative_volume,
                    regime.liquidity_score,
                    "; ".join(regime.reasons),
                )
                self._last_status[symbol] = status_key

        return result

    def get_config(self, style: TradingStyle) -> Optional[StyleConfig]:
        """Get the scanner/verifier config for a trading style."""
        return self._configs.get(style)
