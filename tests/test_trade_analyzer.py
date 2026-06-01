"""Tests for the intraday trade analyzer."""

from __future__ import annotations

from daytrading.analytics.trade_analyzer import TradeAnalyzer, TradeRecord


def _trade(
    symbol: str,
    pnl: float,
    *,
    qty: float = 100,
    entry: float = 2.75,
    exit_: float = 2.70,
    entry_time: str = "2026-06-01T14:30:00+00:00",
    exit_time: str = "2026-06-01T14:35:00+00:00",
    reason: str = "stop_loss",
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        side="buy",
        quantity=qty,
        entry_price=entry,
        exit_price=exit_,
        pnl=pnl,
        exit_reason=reason,
        entry_time=entry_time,
        exit_time=exit_time,
        scanner="hod_momentum",
    )


def test_partial_exit_leftovers_count_as_one_trade_for_analysis() -> None:
    analyzer = TradeAnalyzer(min_trades=3, max_block_trades=3)
    trades = [
        _trade("ABTS", -4.92, qty=193, exit_time="2026-06-01T14:35:00+00:00"),
        _trade("ABTS", -0.09, qty=4, exit_time="2026-06-01T14:35:20+00:00"),
        _trade("ABTS", -0.24, qty=11, exit_time="2026-06-01T14:35:40+00:00"),
        _trade("WIN", 20.0, entry=5.00, exit_=5.20, entry_time="2026-06-01T14:36:00+00:00", exit_time="2026-06-01T14:38:00+00:00"),
        _trade("LOSS", -5.0, entry=7.00, exit_=6.95, entry_time="2026-06-01T14:39:00+00:00", exit_time="2026-06-01T14:40:00+00:00"),
    ]

    result = analyzer.analyze(trades)

    assert "ABTS" not in result.blocked_symbols
    assert "ABTS" not in analyzer.blocked_symbols


def test_symbol_blocks_after_three_real_losses() -> None:
    analyzer = TradeAnalyzer(min_trades=3, max_block_trades=3)
    trades = [
        _trade("ABTS", -6.0, exit_time="2026-06-01T14:31:00+00:00"),
        _trade("ABTS", -5.0, entry=2.80, exit_=2.75, entry_time="2026-06-01T14:36:00+00:00", exit_time="2026-06-01T14:38:00+00:00"),
        _trade("ABTS", -7.0, entry=2.90, exit_=2.84, entry_time="2026-06-01T14:42:00+00:00", exit_time="2026-06-01T14:44:00+00:00"),
    ]

    result = analyzer.analyze(trades)

    assert result.blocked_symbols == ["ABTS"]
    assert "3x consecutive losses" in analyzer.blocked_symbols["ABTS"]
