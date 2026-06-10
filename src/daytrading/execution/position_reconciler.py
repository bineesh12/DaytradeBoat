"""Reconcile internal portfolio + exit tracking with Alpaca (broker is source of truth).

Problems this solves:
  - Untracking a position because Alpaca lagged after a fill (orphaned live trades).
  - Exit manager not monitoring positions that exist only at the broker after restart.
  - Internal portfolio qty drifting from broker after partial fills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, TYPE_CHECKING

from daytrading.exits.manager import ExitManager, TrackedPosition
from daytrading.models import PortfolioState, Position, Side

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Alpaca can take a few seconds to show a new position after fill.
DEFAULT_PENDING_GRACE = timedelta(seconds=90)
# Require repeated empty broker reads before declaring flat (avoids API blips).
DEFAULT_BROKER_MISS_LIMIT = 3


@dataclass
class SymbolLedger:
    """Per-symbol sync state between bot and broker."""

    pending_broker_since: Optional[datetime] = None
    broker_miss_count: int = 0
    last_seen_on_broker: Optional[datetime] = None
    broker_stop_order_id: Optional[str] = None


@dataclass
class ReconcileResult:
    adopted: list = field(default_factory=list)
    confirmed: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    still_pending: list = field(default_factory=list)
    accidental_shorts: list = field(default_factory=list)


class PositionReconciler:
    """Keep portfolio + exit manager aligned with Alpaca positions."""

    def __init__(
        self,
        *,
        pending_grace: timedelta = DEFAULT_PENDING_GRACE,
        broker_miss_limit: int = DEFAULT_BROKER_MISS_LIMIT,
    ) -> None:
        self._pending_grace = pending_grace
        self._broker_miss_limit = broker_miss_limit
        self._ledger: Dict[str, SymbolLedger] = {}

    def ledger(self, symbol: str) -> SymbolLedger:
        if symbol not in self._ledger:
            self._ledger[symbol] = SymbolLedger()
        return self._ledger[symbol]

    def mark_entry_pending(self, symbol: str, now: Optional[datetime] = None) -> None:
        """Call immediately after a local entry fill, before broker shows the position."""
        ts = now or datetime.now(timezone.utc)
        led = self.ledger(symbol)
        led.pending_broker_since = ts
        led.broker_miss_count = 0
        logger.info("RECONCILE pending broker confirm for %s", symbol)

    def clear_pending(self, symbol: str) -> None:
        led = self._ledger.get(symbol)
        if led:
            led.pending_broker_since = None
            led.broker_miss_count = 0

    def set_broker_stop_id(self, symbol: str, order_id: Optional[str]) -> None:
        self.ledger(symbol).broker_stop_order_id = order_id

    def get_broker_stop_id(self, symbol: str) -> Optional[str]:
        return self.ledger(symbol).broker_stop_order_id

    def pop_broker_stop_id(self, symbol: str) -> Optional[str]:
        led = self._ledger.get(symbol)
        if not led:
            return None
        oid = led.broker_stop_order_id
        led.broker_stop_order_id = None
        return oid

    def reconcile(
        self,
        broker_positions: Dict[str, dict],
        portfolio: PortfolioState,
        exit_manager: ExitManager,
        *,
        now: Optional[datetime] = None,
    ) -> ReconcileResult:
        """Sync portfolio and exit manager to broker state.

        Never untrack on a single missing broker read. Positions in
        ``pending_broker_since`` grace stay monitored until confirmed or grace expires.
        """
        now = now or datetime.now(timezone.utc)
        result = ReconcileResult()
        broker_syms = set(broker_positions.keys())
        tracked_syms = set(exit_manager.tracked.keys())

        # --- Broker is authoritative for open quantity / avg price ---
        for sym, data in broker_positions.items():
            qty = float(data.get("qty", 0))
            avg = float(data.get("avg_entry", 0))
            if qty < 0:
                result.accidental_shorts.append(sym)
                short_qty = abs(qty)
                led = self.ledger(sym)
                led.pending_broker_since = None
                led.broker_miss_count = 0
                led.last_seen_on_broker = now

                pos = portfolio.positions.get(sym)
                if pos is None:
                    portfolio.positions[sym] = Position(
                        symbol=sym, quantity=qty, avg_price=avg, entry_ts=now,
                    )
                else:
                    pos.quantity = qty
                    pos.avg_price = avg
                tracked = exit_manager.tracked.get(sym)
                if tracked is None or tracked.side is not Side.SELL:
                    self._adopt_unexpected_short(sym, short_qty, avg, now, exit_manager)
                    tracked_syms.add(sym)
                else:
                    tracked.quantity = short_qty
                    tracked.remaining_qty = short_qty
                    tracked.entry_price = avg
                logger.warning(
                    "RECONCILE detected unexpected short %s %.0f @ %.4f",
                    sym, qty, avg,
                )
                continue
            if qty == 0:
                continue

            led = self.ledger(sym)
            led.pending_broker_since = None
            led.broker_miss_count = 0
            led.last_seen_on_broker = now

            pos = portfolio.positions.get(sym)
            if pos is None or pos.is_flat:
                portfolio.positions[sym] = Position(
                    symbol=sym, quantity=qty, avg_price=avg, entry_ts=now,
                )
            else:
                pos.quantity = qty
                pos.avg_price = avg

            if sym not in tracked_syms:
                self._adopt_orphan(sym, qty, avg, now, exit_manager)
                result.adopted.append(sym)
                tracked_syms.add(sym)
            else:
                tracked = exit_manager.tracked[sym]
                if abs(tracked.remaining_qty - qty) > 0.01:
                    logger.info(
                        "RECONCILE qty sync %s: tracked %.0f → broker %.0f",
                        sym, tracked.remaining_qty, qty,
                    )
                    tracked.remaining_qty = qty
                    tracked.quantity = qty
                result.confirmed.append(sym)

        # --- Symbols we track but broker does not show ---
        for sym in list(tracked_syms):
            if sym in broker_syms:
                continue

            led = self.ledger(sym)
            in_grace = (
                led.pending_broker_since is not None
                and (now - led.pending_broker_since) < self._pending_grace
            )
            if in_grace:
                result.still_pending.append(sym)
                logger.debug(
                    "RECONCILE %s not on broker yet (%.0fs pending)",
                    sym, (now - led.pending_broker_since).total_seconds(),
                )
                continue

            led.broker_miss_count += 1
            if led.broker_miss_count < self._broker_miss_limit:
                logger.warning(
                    "RECONCILE %s missing from broker (%d/%d) — still monitoring exits",
                    sym, led.broker_miss_count, self._broker_miss_limit,
                )
                continue

            logger.warning(
                "RECONCILE %s confirmed flat at broker after %d misses — untracking",
                sym, led.broker_miss_count,
            )
            exit_manager.untrack(sym)
            pos = portfolio.positions.get(sym)
            if pos and not pos.is_flat:
                pos.quantity = 0.0
            result.closed.append(sym)
            led.broker_miss_count = 0
            led.pending_broker_since = None

        # --- Portfolio symbols not on broker (no active exit tracking) ---
        for sym, pos in list(portfolio.positions.items()):
            if sym in broker_syms or pos.is_flat:
                continue
            if sym in exit_manager.tracked:
                continue  # handled above
            pos.quantity = 0.0

        return result

    @staticmethod
    def _adopt_orphan(
        symbol: str,
        qty: float,
        avg: float,
        now: datetime,
        exit_manager: ExitManager,
    ) -> None:
        """Broker has a position the bot is not monitoring — adopt with tight scalp defaults.

        Rebuild the normal 1:1 target from the fallback 10-tick risk. This can
        happen briefly around fresh fills, so disabling the target would turn a
        valid scalp into a runner and skip the first profit-taking exit.
        """
        TICK = 0.01
        STOP_TICKS = 10
        stop = round(avg - STOP_TICKS * TICK, 4)

        pos = TrackedPosition(
            symbol=symbol,
            side=Side.BUY,
            quantity=qty,
            remaining_qty=qty,
            entry_price=avg,
            entry_ts=now,
            stop_loss=stop,
            risk_per_share=STOP_TICKS * TICK,
            trend_strength=0.7,
            reason="adopted from broker",
        )
        exit_manager.track(pos)
        logger.warning(
            "RECONCILE adopted orphan %s %.0f @ %.4f (stop=%.4f, target=%.4f)",
            symbol, qty, avg, stop, pos.first_target_price,
        )

    @staticmethod
    def _adopt_unexpected_short(
        symbol: str,
        qty: float,
        avg: float,
        now: datetime,
        exit_manager: ExitManager,
    ) -> None:
        """Track an unexpected broker short with a tight cover stop.

        The live runner also tries to flatten accidental shorts immediately.
        This software tracker is a second line of defense if that cover order
        cannot be submitted/fills fail.
        """
        TICK = 0.01
        STOP_TICKS = 10
        stop = round(avg + STOP_TICKS * TICK, 4)

        pos = TrackedPosition(
            symbol=symbol,
            side=Side.SELL,
            quantity=qty,
            remaining_qty=qty,
            entry_price=avg,
            entry_ts=now,
            stop_loss=stop,
            risk_per_share=STOP_TICKS * TICK,
            trend_strength=0.0,
            reason="unexpected broker short",
        )
        exit_manager.track(pos)
        logger.error(
            "RECONCILE tracking unexpected short %s %.0f @ %.4f (cover_stop=%.4f)",
            symbol, qty, avg, stop,
        )
