from datetime import datetime, timezone

from daytrading.backtest.replay import JournalReplayRunner
from daytrading.journal.store import TradingJournal
from daytrading.strategy.entry_policy import EntryPolicy


def test_journal_replay_feeds_signal_snapshot_through_entry_policy(tmp_path):
    db_path = tmp_path / "journal.db"
    journal = TradingJournal(
        db_path=str(db_path),
        record_candle_snapshots=True,
        prune_on_start=False,
        daily_prune_enabled=False,
    )
    ts = datetime.now(timezone.utc)
    journal.record(
        "signal",
        {
            "symbol": "BATL",
            "action": "enter_long",
            "entry_price": 1.66,
            "scanner": "pullback_base",
            "pattern": "pullback_base",
            "criteria": {"pattern": "pullback_base", "setup_tier": "A+"},
            "candle_snapshot": [
                {
                    "symbol": "BATL",
                    "ts": ts.isoformat(),
                    "open": 1.60,
                    "high": 1.70,
                    "low": 1.58,
                    "close": 1.66,
                    "volume": 100_000,
                    "timeframe": "1m",
                }
            ],
        },
        ts=ts,
    )

    runner = JournalReplayRunner(
        journal=journal,
        policy=EntryPolicy(guard=lambda *args, **kwargs: None),
    )

    result = runner.replay(day=ts.date().isoformat())

    assert result.skipped == 0
    assert len(result.decisions) == 1
    assert result.decisions[0].passed
    assert result.decisions[0].pattern == "pullback_base"
