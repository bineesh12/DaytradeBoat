"""VWAP bounce strategy verifier.

Logic: stock is trading above/below VWAP with high relative volume →
if it pulls back TO the VWAP and bounces, enter in the direction of the
prevailing trend (VWAP acts as dynamic support/resistance).

Works with: VWAPDeviationScanner
"""

from __future__ import annotations

from typing import Optional

from daytrading.indicators.core import atr, vwap
from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality


class VWAPBounceVerifier:
    """Verify VWAP deviation scan hits for a trend-continuation entry."""

    def __init__(
        self,
        *,
        risk_atr_mult: float = 1.0,
        reward_atr_mult: float = 2.0,
        position_size: float = 100,
    ) -> None:
        self._risk_atr_mult = risk_atr_mult
        self._reward_atr_mult = reward_atr_mult
        self._position_size = position_size

    @property
    def name(self) -> str:
        return "vwap_bounce"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        bars = scan_result.bars
        if len(bars) < 3:
            return None

        direction = scan_result.criteria.get("direction", "")
        current_vwap = scan_result.criteria.get("vwap", 0.0)
        if current_vwap <= 0:
            return None

        latest = bars[-1]
        prev = bars[-2]

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            return None

        atr_values = atr(bars, period=min(len(bars) - 1, 14))
        current_atr = atr_values[-1] if atr_values else 0.0
        if current_atr != current_atr:  # NaN
            current_atr = abs(latest.high - latest.low)

        if direction == "above":
            # price trending above VWAP — look for pullback bounce
            touched_vwap = prev.low <= current_vwap * 1.002
            bounced = latest.close > latest.open and latest.close > current_vwap
            if touched_vwap and bounced:
                reject = check_entry_quality(bars, symbol=scan_result.symbol)
                if reject is not None:
                    return None
                stop = current_vwap - current_atr * self._risk_atr_mult
                target = latest.close + current_atr * self._reward_atr_mult
                return TradeSignal(
                    symbol=scan_result.symbol,
                    action=SignalAction.ENTER_LONG,
                    quantity=self._position_size,
                    entry_price=latest.close,
                    stop_loss=stop,
                    take_profit=target,
                    reason=f"VWAP bounce long, VWAP={current_vwap:.2f}",
                    scan_result=scan_result,
                )

        # Short selling disabled — buy side only
        return None
