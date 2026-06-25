from daytrading.strategy.warrior_risk import WarriorRiskAllocator
from daytrading.strategy.warrior_watch import WarriorWatchBook


def test_warrior_risk_allocator_blocks_at_default_single_concurrent_trade() -> None:
    allocator = WarriorRiskAllocator()

    decision = allocator.allow("AAA", WarriorWatchBook(), open_positions=1)

    assert decision.allowed is False
    assert "max concurrent Warrior trades" in decision.reason


def test_warrior_risk_allocator_prepares_multi_symbol_mode() -> None:
    allocator = WarriorRiskAllocator(max_concurrent_warrior_trades=2)

    assert allocator.allow("AAA", WarriorWatchBook(), open_positions=1).allowed is True
    assert allocator.allow("BBB", WarriorWatchBook(), open_positions=2).allowed is False


def test_warrior_risk_allocator_blocks_day_blocked_symbol() -> None:
    allocator = WarriorRiskAllocator(max_concurrent_warrior_trades=2)
    book = WarriorWatchBook()
    book.day_blocked["AAA"] = "daily loss stop"

    decision = allocator.allow("AAA", book, open_positions=0)

    assert decision.allowed is False
    assert "daily loss stop" in decision.reason
