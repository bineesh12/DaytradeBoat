from daytrading.strategy.warrior_watch import WarriorWatchBook


def test_warrior_watch_book_resets_all_session_state() -> None:
    book = WarriorWatchBook()

    book.armed["AAA"] = object()
    book.window_high["AAA"] = 5.0
    book.session_anchor_high["AAA"] = 4.5
    book.pending["AAA"] = {"entry_trigger": "warrior_level_pullaway"}
    book.hit_run_counts["AAA"] = 2
    book.hit_run_block_until["AAA"] = object()
    book.symbol_pnl["AAA"] = 10.0
    book.symbol_peak_pnl["AAA"] = 15.0
    book.day_blocked["AAA"] = "daily stop"
    book.rejection_high["AAA"] = 3.5
    book.rejection_reason["AAA"] = "first cheap spike"
    book.target_wins["AAA"] = 1
    book.last_target_at["AAA"] = object()
    book.failed_burst["AAA"] = "stopped"
    book.failed_burst_high["AAA"] = 5.25
    book.post_target_reclaim_allowed["AAA"] = 1
    book.last_entry_trigger["AAA"] = "warrior_low_price_proof_reclaim"
    book.normal_fallback_rejects["AAA"] = 3
    book.normal_fallback_last_reason["AAA"] = "micro base wait"

    book.reset_session()

    assert book.armed == {}
    assert book.window_high == {}
    assert book.session_anchor_high == {}
    assert book.pending == {}
    assert book.hit_run_counts == {}
    assert book.hit_run_block_until == {}
    assert book.symbol_pnl == {}
    assert book.symbol_peak_pnl == {}
    assert book.day_blocked == {}
    assert book.rejection_high == {}
    assert book.rejection_reason == {}
    assert book.target_wins == {}
    assert book.last_target_at == {}
    assert book.failed_burst == {}
    assert book.failed_burst_high == {}
    assert book.post_target_reclaim_allowed == {}
    assert book.last_entry_trigger == {}
    assert book.normal_fallback_rejects == {}
    assert book.normal_fallback_last_reason == {}


def test_warrior_watch_book_replaces_weakest_inactive_symbol() -> None:
    book = WarriorWatchBook()
    book.armed.update({"AAA": 1, "BBB": 1})
    book.window_high.update({"AAA": 4.0, "BBB": 5.0})
    book.session_anchor_high.update({"AAA": 4.0, "BBB": 4.0})

    assert book.ensure_capacity("CCC", capacity=2, candidate_high=3.5)

    assert "AAA" not in book.armed
    assert "BBB" in book.armed


def test_warrior_watch_book_does_not_evict_active_symbols() -> None:
    book = WarriorWatchBook()
    book.armed.update({"AAA": 1, "BBB": 1})
    book.window_high.update({"AAA": 4.0, "BBB": 5.0})

    assert not book.ensure_capacity(
        "CCC",
        capacity=2,
        candidate_high=6.0,
        active_symbols={"AAA", "BBB"},
    )
    assert set(book.armed) == {"AAA", "BBB"}


def test_warrior_watch_book_can_evict_pending_weakest_symbol() -> None:
    book = WarriorWatchBook()
    book.armed.update({"AAA": 1, "BBB": 1})
    book.window_high.update({"AAA": 4.0, "BBB": 8.0})
    book.session_anchor_high.update({"AAA": 4.0, "BBB": 4.0})
    book.pending["AAA"] = {"entry_trigger": "warrior_level_pullaway"}

    assert book.ensure_capacity("CCC", capacity=2, candidate_high=7.0)

    assert "AAA" not in book.armed
    assert "AAA" not in book.pending
    assert set(book.armed) == {"BBB"}


def test_warrior_watch_book_hard_protects_target_winner() -> None:
    book = WarriorWatchBook()
    book.armed.update({"AAA": 1, "BBB": 1})
    book.window_high.update({"AAA": 4.0, "BBB": 5.0})
    book.session_anchor_high.update({"AAA": 4.0, "BBB": 4.0})
    book.target_wins["AAA"] = 1

    assert book.ensure_capacity("CCC", capacity=2, candidate_high=20.0)

    assert "AAA" in book.armed
    assert "BBB" not in book.armed
    assert "CCC" not in book.armed


def test_warrior_watch_book_eviction_clears_paired_risk_state() -> None:
    book = WarriorWatchBook()
    book.armed["AAA"] = object()
    book.window_high["AAA"] = 3.5
    book.session_anchor_high["AAA"] = 2.0
    book.pending["AAA"] = {"entry_trigger": "warrior_low_price_proof_reclaim"}
    book.hit_run_counts["AAA"] = 2
    book.hit_run_block_until["AAA"] = object()
    book.rejection_high["AAA"] = 2.25
    book.rejection_reason["AAA"] = "first cheap spike"
    book.failed_burst["AAA"] = "stopped"
    book.failed_burst_high["AAA"] = 3.75
    book.post_target_reclaim_allowed["AAA"] = 1
    book.last_entry_trigger["AAA"] = "warrior_low_price_proof_reclaim"
    book.normal_fallback_rejects["AAA"] = 3
    book.normal_fallback_last_reason["AAA"] = "micro base wait"
    book.symbol_pnl["AAA"] = -5.0

    book.clear_watch_symbol("AAA")

    assert book.symbol_pnl["AAA"] == -5.0
    assert "AAA" not in book.armed
    assert "AAA" not in book.window_high
    assert "AAA" not in book.session_anchor_high
    assert "AAA" not in book.pending
    assert "AAA" not in book.hit_run_counts
    assert "AAA" not in book.hit_run_block_until
    assert "AAA" not in book.rejection_high
    assert "AAA" not in book.rejection_reason
    assert "AAA" not in book.failed_burst
    assert "AAA" not in book.failed_burst_high
    assert "AAA" not in book.post_target_reclaim_allowed
    assert "AAA" not in book.last_entry_trigger
    assert "AAA" not in book.normal_fallback_rejects
    assert "AAA" not in book.normal_fallback_last_reason


def test_warrior_watch_score_penalizes_day_block_and_rewards_targets() -> None:
    book = WarriorWatchBook()
    book.window_high["AAA"] = 5.0
    book.session_anchor_high["AAA"] = 4.0
    base_score = book.watch_score("AAA")

    book.target_wins["AAA"] = 1
    assert book.watch_score("AAA") > base_score + 90.0

    book.day_blocked["AAA"] = "daily loss stop"
    assert book.watch_score("AAA") < base_score - 800.0
