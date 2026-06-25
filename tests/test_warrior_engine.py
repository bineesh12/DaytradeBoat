from daytrading.strategy.warrior_engine import WarriorEngine


def test_warrior_engine_defaults_to_single_active_trade() -> None:
    engine = WarriorEngine.with_defaults()

    assert engine.allow_entry("AAA", open_positions=0).allowed is True
    blocked = engine.allow_entry("AAA", open_positions=1)
    assert blocked.allowed is False
    assert "max concurrent Warrior trades" in blocked.reason


def test_warrior_engine_can_prepare_two_concurrent_trades() -> None:
    engine = WarriorEngine.with_defaults(max_concurrent_warrior_trades=2)

    assert engine.allow_entry("AAA", open_positions=1).allowed is True
    assert engine.allow_entry("BBB", open_positions=2).allowed is False


def test_warrior_engine_resets_watch_state() -> None:
    engine = WarriorEngine.with_defaults()
    engine.watch.armed["AAA"] = object()
    engine.watch.target_wins["AAA"] = 1

    engine.reset_session()

    assert engine.watch.armed == {}
    assert engine.watch.target_wins == {}

