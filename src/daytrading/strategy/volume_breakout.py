"""Volume breakout strategy verifier.

Logic: stock shows a volume spike (from VolumeSpikeScanner) → verify
that price is also breaking above recent highs or below recent lows →
enter in the breakout direction.

Works with: VolumeSpikeScanner
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from daytrading.indicators.core import atr
from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality


class VolumeBreakoutVerifier:
    """Verify volume spike scan hits for a breakout entry."""

    def __init__(
        self,
        *,
        lookback_bars: int = 5,
        risk_atr_mult: float = 1.5,
        reward_atr_mult: float = 3.0,
        position_size: float = 100,
    ) -> None:
        self._lookback = lookback_bars
        self._risk_atr_mult = risk_atr_mult
        self._reward_atr_mult = reward_atr_mult
        self._position_size = position_size

    @property
    def name(self) -> str:
        return "volume_breakout"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        bars = scan_result.bars
        if len(bars) < self._lookback + 1:
            return None

        latest = bars[-1]
        prior = bars[-(self._lookback + 1):-1]

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            return None

        atr_values = atr(bars, period=min(len(bars) - 1, 14))
        current_atr = atr_values[-1] if atr_values else 0.0
        if current_atr != current_atr:  # NaN
            current_atr = abs(latest.high - latest.low)

        recent_high = max(b.high for b in prior)
        recent_low = min(b.low for b in prior)

        if latest.close > recent_high and latest.close > latest.open:
            reject = check_entry_quality(bars, symbol=scan_result.symbol, now=now)
            if reject is not None:
                return None
            stop = latest.close - current_atr * self._risk_atr_mult
            target = latest.close + current_atr * self._reward_atr_mult
            rvol = scan_result.criteria.get("rvol", 0.0)
            return TradeSignal(
                symbol=scan_result.symbol,
                action=SignalAction.ENTER_LONG,
                quantity=self._position_size,
                entry_price=latest.close,
                stop_loss=stop,
                take_profit=target,
                reason=f"Volume breakout long, RVOL={rvol:.1f}x, broke {recent_high:.2f}",
                scan_result=scan_result,
            )

        # Short selling disabled — buy side only

        return None
