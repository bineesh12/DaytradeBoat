"""The signal-to-marketable-limit policy shared by Alpaca and bar replay."""

import math
from dataclasses import dataclass

from daytrading.models import Bar, Order, Side


@dataclass(frozen=True)
class ExecutionPricing:
    market_slippage_pct: float = 0.0075
    buy_slippage_pct: float = 0.0075
    thin_buy_slippage_pct: float = 0.005
    momentum_buy_slippage_pct: float = 0.01
    sell_slippage_pct: float = 0.0075

    def buy_window(self, bar: Bar) -> float:
        if bar.volume < 50000:
            return self.thin_buy_slippage_pct
        if (
            bar.volume >= 100000
            and bar.close > bar.open
            and bar.low > 0
            and (bar.high - bar.low) / bar.low > 0.02
        ):
            return self.momentum_buy_slippage_pct
        return self.buy_slippage_pct


def price_round(price: float) -> float:
    return round(price, 2 if price >= 1 else 4)


def marketable_limit(
    order: Order,
    bar: Bar,
    *,
    quote_price: float | None = None,
    policy: ExecutionPricing | None = None,
    urgent_pct: float | None = None,
) -> float:
    policy = policy or ExecutionPricing()
    base = float(order.limit_price if order.limit_price is not None else bar.close)
    if not math.isfinite(base) or base <= 0:
        raise ValueError("no executable signal price")
    if quote_price is not None and (not math.isfinite(quote_price) or quote_price <= 0):
        raise ValueError("invalid executable quote")
    if order.hard_limit_price is not None:
        hard = order.hard_limit_price
        if not math.isfinite(hard) or hard <= 0 or price_round(hard) != hard:
            raise ValueError('invalid hard execution limit')
        if quote_price is not None and (
            (order.side is Side.BUY and quote_price > hard + 1e-12)
            or (order.side is Side.SELL and quote_price < hard - 1e-12)
        ):
            raise ValueError('quote exceeds hard execution limit')
        # This is an already priced, risk-reserved order. Never add headroom.
        return hard
    if urgent_pct is not None:
        base = float(bar.close or base)
        window = max(0.0, min(urgent_pct, 0.20))
        return price_round(
            base * (1 + window if order.side is Side.BUY else 1 - window)
        )
    if order.limit_price is None:
        base = quote_price if quote_price is not None else base
        return price_round(
            base
            * (
                1 + policy.market_slippage_pct
                if order.side is Side.BUY
                else 1 - policy.market_slippage_pct
            )
        )
    if order.side is Side.SELL:
        return price_round(
            quote_price
            if quote_price is not None
            else base * (1 - policy.sell_slippage_pct)
        )
    maximum = price_round(base * (1 + policy.buy_window(bar)))
    if quote_price is not None and quote_price > maximum:
        raise ValueError("ask exceeds signal slippage window")
    return price_round(quote_price if quote_price is not None else maximum)
