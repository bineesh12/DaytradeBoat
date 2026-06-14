"""Gap reversal strategy verifier.

Logic: stock gaps up ≥ X% pre-market → wait for it to fade back toward
the previous close → enter long anticipating a bounce, OR enter short if
the gap is too extended and RSI is overbought.

Works with: PremarketGapScanner
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from daytrading.indicators.core import atr, rsi
from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality


class GapReversalVerifier:
    """Verify gap-up scan hits for a mean-reversion entry."""

    def __init__(
        self,
        *,
        max_rsi: float = 70.0,
        min_rsi: float = 30.0,
        risk_atr_mult: float = 1.5,
        reward_atr_mult: float = 3.0,
        position_size: float = 100,
    ) -> None:
        self._max_rsi = max_rsi
        self._min_rsi = min_rsi
        self._risk_atr_mult = risk_atr_mult
        self._reward_atr_mult = reward_atr_mult
        self._position_size = position_size

    @property
    def name(self) -> str:
        return "gap_reversal"

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[TradeSignal]:
        bars = scan_result.bars
        if len(bars) < 3:
            return None

        gap_pct = scan_result.criteria.get("gap_pct", 0.0)
        prev_close = scan_result.criteria.get("prev_close", 0.0)
        if prev_close <= 0:
            return None

        rsi_values = rsi(bars, period=min(len(bars) - 1, 14))
        current_rsi = rsi_values[-1] if rsi_values else 50.0

        atr_values = atr(bars, period=min(len(bars) - 1, 14))
        current_atr = atr_values[-1] if atr_values else 0.0
        if current_atr != current_atr:  # NaN check
            current_atr = abs(bars[-1].high - bars[-1].low)

        latest = bars[-1]
        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            return None

        if gap_pct > 0 and current_rsi < self._max_rsi:
            # gap up but RSI not overheated — buy the dip if price faded from open
            if latest.close < latest.open:
                reject = check_entry_quality(bars, symbol=scan_result.symbol, now=now)
                if reject is not None:
                    return None
                stop = latest.close - current_atr * self._risk_atr_mult
                target = latest.close + current_atr * self._reward_atr_mult
                return TradeSignal(
                    symbol=scan_result.symbol,
                    action=SignalAction.ENTER_LONG,
                    quantity=self._position_size,
                    entry_price=latest.close,
                    stop_loss=stop,
                    take_profit=target,
                    reason=f"Gap up {gap_pct:.1f}% faded, RSI {current_rsi:.0f}",
                    scan_result=scan_result,
                )

        # Short selling disabled — buy side only
        return None
