"""Market regime classifier.

Analyzes a symbol's recent price action, volume, spread, and volatility
to determine which trading style is best suited RIGHT NOW:

  SCALPING       → tight spread, high liquidity, fast tape, low volatility per bar
  DAY_TRADING    → moderate volatility, decent range, clear intraday trends
  SWING          → wide daily range, strong multi-bar trend, higher timeframe signals
  NOT_TRADEABLE  → too illiquid, spread too wide, or too choppy to trade profitably

The classifier uses a scoring system:

  1. Measure 6 core metrics from bars (+ optional quotes)
  2. Score each trading style 0–100 based on how well the metrics fit
  3. Pick the style with the highest score (above a minimum threshold)

This runs BEFORE any scanner — it tells the pipeline which scanners
and verifiers to even bother running for each symbol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from daytrading.indicators.core import atr, ema, relative_volume, sma
from daytrading.models import Bar, MarketRegime, Quote, TradingStyle

NaN = float("nan")


@dataclass
class _Metrics:
    """Raw metrics computed from price data."""
    volatility_pct: float = 0.0
    spread_pct: float = 0.0
    rvol: float = 1.0
    trend_strength: float = 0.0
    avg_bar_range_pct: float = 0.0
    liquidity_score: float = 0.0


class MarketRegimeClassifier:
    """Classifies each symbol into the best trading style."""

    def __init__(
        self,
        *,
        # Price filter
        min_price: float = 1.0,
        max_price: float = 20.0,

        # Which styles are enabled
        enable_scalping: bool = True,
        enable_day_trading: bool = False,
        enable_swing: bool = False,

        # Volatility thresholds (ATR as % of price)
        scalp_max_volatility: float = 5.0,  # allow volatile movers for scalping
        day_min_volatility: float = 0.3,
        day_max_volatility: float = 3.0,
        swing_min_volatility: float = 1.0,

        # Spread thresholds (% of mid price)
        scalp_max_spread_pct: float = 0.75,  # momentum stocks often have 0.3-0.7% spread
        day_max_spread_pct: float = 0.10,

        # Volume thresholds
        min_rvol_active: float = 0.5,
        scalp_min_rvol: float = 1.0,

        # Trend
        trend_ema_fast: int = 8,
        trend_ema_slow: int = 21,

        # Liquidity (relaxed for low-float momentum stocks)
        min_avg_volume: float = 5_000,
        high_liquidity_volume: float = 500_000,

        # Scoring
        min_confidence: float = 0.3,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._enable_scalping = enable_scalping
        self._enable_day = enable_day_trading
        self._enable_swing = enable_swing
        self._scalp_max_vol = scalp_max_volatility
        self._day_min_vol = day_min_volatility
        self._day_max_vol = day_max_volatility
        self._swing_min_vol = swing_min_volatility
        self._scalp_max_spread = scalp_max_spread_pct
        self._day_max_spread = day_max_spread_pct
        self._min_rvol_active = min_rvol_active
        self._scalp_min_rvol = scalp_min_rvol
        self._trend_fast = trend_ema_fast
        self._trend_slow = trend_ema_slow
        self._min_avg_volume = min_avg_volume
        self._high_liq_volume = high_liquidity_volume
        self._min_confidence = min_confidence

    def classify(
        self,
        symbol: str,
        bars: Sequence[Bar],
        quotes: Optional[Sequence[Quote]] = None,
    ) -> MarketRegime:
        """Classify a single symbol based on its recent bars and optional quotes."""

        if len(bars) < 2:
            return self._not_tradeable(symbol, bars, ["Insufficient bar data"])

        price = bars[-1].close
        if price < self._min_price:
            return self._not_tradeable(symbol, bars, [
                f"Price ${price:.2f} below minimum ${self._min_price:.2f}",
            ])
        if price > self._max_price:
            return self._not_tradeable(symbol, bars, [
                f"Price ${price:.2f} above maximum ${self._max_price:.2f}",
            ])

        metrics = self._compute_metrics(bars, quotes)

        if metrics.liquidity_score < 0.1:
            # For momentum/low-float stocks, historical volume is irrelevant —
            # but they MUST have elevated RVOL today to confirm real buying.
            # Strong trend alone isn't enough (could be thin order book grind).
            if metrics.rvol < 2.0:
                return self._not_tradeable(symbol, bars, [
                    f"Too illiquid: avg_volume below threshold, liquidity={metrics.liquidity_score:.2f}, rvol={metrics.rvol:.1f}x",
                ])

        scores: Dict[TradingStyle, tuple] = {}
        if self._enable_scalping:
            scores[TradingStyle.SCALPING] = self._score_scalping(metrics)
        if self._enable_day:
            scores[TradingStyle.DAY_TRADING] = self._score_day_trading(metrics)
        if self._enable_swing:
            scores[TradingStyle.SWING] = self._score_swing(metrics)

        if not scores:
            return self._not_tradeable(symbol, bars, ["No trading styles enabled"])

        best_style = max(scores, key=lambda s: scores[s][0])
        best_score, best_reasons = scores[best_style]

        confidence = best_score / 100.0

        if confidence < self._min_confidence:
            all_reasons = [f"No style scored above {self._min_confidence:.0%} threshold"]
            for s, (sc, rs) in scores.items():
                all_reasons.append(f"  {s.value}: {sc:.0f}/100 — {', '.join(rs)}")
            return self._not_tradeable(symbol, bars, all_reasons)

        ts = bars[-1].ts
        return MarketRegime(
            symbol=symbol,
            ts=ts,
            style=best_style,
            confidence=confidence,
            volatility_pct=metrics.volatility_pct,
            spread_pct=metrics.spread_pct,
            relative_volume=metrics.rvol,
            trend_strength=metrics.trend_strength,
            avg_bar_range_pct=metrics.avg_bar_range_pct,
            liquidity_score=metrics.liquidity_score,
            reasons=best_reasons,
        )

    def classify_universe(
        self,
        universe: Dict[str, Sequence[Bar]],
        quotes: Optional[Dict[str, Sequence[Quote]]] = None,
    ) -> Dict[str, MarketRegime]:
        """Classify every symbol in the universe."""
        result: Dict[str, MarketRegime] = {}
        for symbol, bars in universe.items():
            q = quotes.get(symbol) if quotes else None
            result[symbol] = self.classify(symbol, bars, q)
        return result

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        bars: Sequence[Bar],
        quotes: Optional[Sequence[Quote]],
    ) -> _Metrics:
        m = _Metrics()
        price = bars[-1].close
        if price <= 0:
            return m

        # --- Volatility: ATR as % of price ---
        atr_period = min(14, len(bars) - 1)
        if atr_period >= 1:
            atr_vals = atr(bars, period=atr_period)
            last_atr = atr_vals[-1]
            if not math.isnan(last_atr):
                m.volatility_pct = (last_atr / price) * 100.0

        # --- Average bar range ---
        ranges = [(b.high - b.low) / b.close * 100.0 for b in bars[-20:] if b.close > 0]
        m.avg_bar_range_pct = sum(ranges) / len(ranges) if ranges else 0.0

        # --- Spread ---
        if quotes and len(quotes) >= 1:
            spreads = [q.spread_pct for q in quotes[-20:]]
            m.spread_pct = sum(spreads) / len(spreads)
        else:
            # estimate from bar data: avg(high-low) as proxy for 2× typical spread
            m.spread_pct = m.avg_bar_range_pct * 0.1

        # --- Relative volume ---
        rvol_period = min(20, len(bars) - 1)
        if rvol_period >= 1:
            rv = relative_volume(bars, period=rvol_period)
            last_rv = rv[-1] if rv else 1.0
            m.rvol = last_rv if not math.isnan(last_rv) else 1.0

        # --- Trend strength ---
        m.trend_strength = self._compute_trend_strength(bars)

        # --- Liquidity score ---
        avg_vol = sum(b.volume for b in bars[-20:]) / min(20, len(bars))
        if avg_vol < self._min_avg_volume:
            m.liquidity_score = 0.0
        elif avg_vol >= self._high_liq_volume:
            m.liquidity_score = 1.0
        else:
            m.liquidity_score = (avg_vol - self._min_avg_volume) / (
                self._high_liq_volume - self._min_avg_volume
            )

        return m

    def _compute_trend_strength(self, bars: Sequence[Bar]) -> float:
        """0 = choppy/range-bound, 1 = strong directional trend.

        Uses EMA crossover consistency: what fraction of the last N bars
        had fast EMA on the same side of slow EMA?
        """
        if len(bars) < self._trend_slow + 1:
            return 0.5  # indeterminate

        fast = ema(bars, self._trend_fast)
        slow = ema(bars, self._trend_slow)

        lookback = min(20, len(bars) - self._trend_slow)
        if lookback <= 0:
            return 0.5

        same_side = 0
        for i in range(len(bars) - lookback, len(bars)):
            f, s = fast[i], slow[i]
            if math.isnan(f) or math.isnan(s):
                continue
            if i > 0:
                prev_f, prev_s = fast[i - 1], slow[i - 1]
                if math.isnan(prev_f) or math.isnan(prev_s):
                    continue
                if (f > s) == (prev_f > prev_s):
                    same_side += 1

        return same_side / lookback if lookback > 0 else 0.5

    # ------------------------------------------------------------------
    # Style scoring (0–100 each)
    # ------------------------------------------------------------------

    def _score_scalping(self, m: _Metrics) -> tuple:
        score = 0.0
        reasons: List[str] = []

        # tight spread is #1 requirement
        if m.spread_pct <= self._scalp_max_spread:
            score += 35
            reasons.append(f"Tight spread {m.spread_pct:.3f}%")
        elif m.spread_pct <= self._scalp_max_spread * 2:
            score += 15
            reasons.append(f"Acceptable spread {m.spread_pct:.3f}%")
        else:
            reasons.append(f"Spread too wide {m.spread_pct:.3f}%")

        # high liquidity
        if m.liquidity_score >= 0.7:
            score += 25
            reasons.append(f"High liquidity {m.liquidity_score:.2f}")
        elif m.liquidity_score >= 0.4:
            score += 12
            reasons.append(f"Moderate liquidity {m.liquidity_score:.2f}")

        # Volatility sweet spot: need SOME movement for scalping edge,
        # but not so much that stops get blown instantly.
        if 0.15 <= m.volatility_pct <= self._scalp_max_vol:
            score += 20
            reasons.append(f"Good volatility {m.volatility_pct:.2f}% (predictable moves)")
        elif m.volatility_pct < 0.15:
            reasons.append(f"Too flat {m.volatility_pct:.2f}% (no edge)")
            score -= 15
        elif m.volatility_pct <= self._scalp_max_vol * 2:
            score += 8
        else:
            reasons.append(f"Too volatile for scalping {m.volatility_pct:.2f}%")

        # Minimum bar range: if bars aren't moving, there's nothing to scalp
        if m.avg_bar_range_pct < 0.1:
            score -= 20
            reasons.append(f"Dead price action {m.avg_bar_range_pct:.3f}% bar range")

        # active tape (volume)
        if m.rvol >= self._scalp_min_rvol:
            score += 15
            reasons.append(f"Active tape RVOL={m.rvol:.1f}x")
        elif m.rvol >= self._min_rvol_active:
            score += 5

        # Warrior Trading: strong trend is a BONUS (we ride momentum, not mean-revert)
        if m.trend_strength > 0.8:
            score += 10
            reasons.append("Strong trend (momentum continuation)")
        elif m.trend_strength > 0.6:
            score += 5

        return max(score, 0), reasons

    def _score_day_trading(self, m: _Metrics) -> tuple:
        score = 0.0
        reasons: List[str] = []

        # needs enough volatility for meaningful intraday moves
        if self._day_min_vol <= m.volatility_pct <= self._day_max_vol:
            score += 30
            reasons.append(f"Good volatility {m.volatility_pct:.2f}%")
        elif m.volatility_pct < self._day_min_vol:
            score += 5
            reasons.append(f"Low volatility {m.volatility_pct:.2f}% (small moves)")
        elif m.volatility_pct > self._day_max_vol:
            score += 10
            reasons.append(f"High volatility {m.volatility_pct:.2f}% (risky but tradeable)")

        # spread must be reasonable
        if m.spread_pct <= self._day_max_spread:
            score += 20
            reasons.append(f"Reasonable spread {m.spread_pct:.3f}%")
        else:
            reasons.append(f"Spread too wide {m.spread_pct:.3f}%")

        # decent bar range
        if 0.3 <= m.avg_bar_range_pct <= 2.0:
            score += 20
            reasons.append(f"Good bar range {m.avg_bar_range_pct:.2f}%")
        elif m.avg_bar_range_pct > 2.0:
            score += 10

        # volume activity
        if m.rvol >= self._min_rvol_active:
            score += 15
            reasons.append(f"Active volume RVOL={m.rvol:.1f}x")

        # moderate trend is ideal for day trading
        if 0.3 <= m.trend_strength <= 0.8:
            score += 15
            reasons.append(f"Moderate trend {m.trend_strength:.2f}")
        elif m.trend_strength > 0.8:
            score += 10
            reasons.append(f"Strong trend {m.trend_strength:.2f}")

        return max(score, 0), reasons

    def _score_swing(self, m: _Metrics) -> tuple:
        score = 0.0
        reasons: List[str] = []

        # needs higher volatility for multi-day moves
        if m.volatility_pct >= self._swing_min_vol:
            score += 30
            reasons.append(f"High volatility {m.volatility_pct:.2f}% (big moves)")
        else:
            score += 5
            reasons.append(f"Low volatility {m.volatility_pct:.2f}% (small swings)")

        # strong trend is the primary swing signal
        if m.trend_strength >= 0.7:
            score += 35
            reasons.append(f"Strong trend {m.trend_strength:.2f} (ride the wave)")
        elif m.trend_strength >= 0.4:
            score += 15
            reasons.append(f"Moderate trend {m.trend_strength:.2f}")
        else:
            reasons.append(f"Weak/choppy trend {m.trend_strength:.2f}")

        # spread is less critical for swing but still matters
        if m.spread_pct <= self._day_max_spread * 2:
            score += 10
        else:
            reasons.append(f"Wide spread {m.spread_pct:.3f}% (acceptable for swing)")
            score += 3

        # volume less critical
        if m.rvol >= self._min_rvol_active:
            score += 10
        else:
            score += 5

        # wide bar range confirms swing potential
        if m.avg_bar_range_pct >= 1.5:
            score += 15
            reasons.append(f"Wide bar range {m.avg_bar_range_pct:.2f}%")
        elif m.avg_bar_range_pct >= 0.5:
            score += 8

        return max(score, 0), reasons

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _not_tradeable(
        self, symbol: str, bars: Sequence[Bar], reasons: List[str],
    ) -> MarketRegime:
        ts = bars[-1].ts if bars else None
        return MarketRegime(
            symbol=symbol,
            ts=ts,  # type: ignore[arg-type]
            style=TradingStyle.NOT_TRADEABLE,
            confidence=0.0,
            volatility_pct=0.0,
            spread_pct=0.0,
            relative_volume=0.0,
            trend_strength=0.0,
            avg_bar_range_pct=0.0,
            liquidity_score=0.0,
            reasons=reasons,
        )
