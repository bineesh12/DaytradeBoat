"""Typed refusal authority; only an explicit no-SDK outcome can permit retry."""
from dataclasses import dataclass
from enum import Enum

from daytrading.data.completed_bars import Rejection


class EntryRecovery(str, Enum):
    ORDER_PLAN_SPREAD = 'order_plan_spread'


@dataclass(frozen=True)
class RecoverableEntryRefusal(Rejection):
    recovery: EntryRecovery

    def __post_init__(self):
        if not isinstance(self.recovery, EntryRecovery):
            raise TypeError('An explicit typed recovery code is required')


def recovery_code(value):
    if isinstance(value, RecoverableEntryRefusal):
        value = value.recovery
    # An identically spelled log string does not confer recovery authority.
    return value if isinstance(value, EntryRecovery) and value is EntryRecovery.ORDER_PLAN_SPREAD else None
