"""Alert label helpers for the HOD Momentum scanner."""

from __future__ import annotations

from typing import List, Optional, Sequence

from daytrading.models import Bar

# Alerts that put symbol on hod_active board for entry gate
HOD_ENTRY_GATE_ALERTS = frozenset({
    "New HOD Breakout",
    "Today HOD Breakout",
    "HOD Reclaim",
    "Low Float - High Rel Vol",
    "Squeeze - Up 5% in 5min",
    "Squeeze - Up 10% in 10min",
    "Gapper Continuation",
})


def fmt_vol(v: float) -> str:
    if v >= 1_000_000:
        return "{:.2f}M".format(v / 1_000_000)
    if v >= 1_000:
        return "{:.2f}K".format(v / 1_000)
    return "{:.0f}".format(v)


def fmt_float(v: float) -> str:
    if v >= 1_000_000:
        return "{:.2f}M".format(v / 1_000_000)
    if v >= 1_000:
        return "{:.2f}K".format(v / 1_000)
    return "{:.0f}".format(v)


def bar_volume_surge(today_bars: Sequence[Bar]) -> float:
    if len(today_bars) < 5:
        return 0.0
    recent = today_bars[-5:]
    recent_avg = sum(b.volume for b in recent) / 5
    if len(today_bars) >= 10:
        earlier = today_bars[:-5]
        earlier_avg = sum(b.volume for b in earlier) / len(earlier)
        return recent_avg / earlier_avg if earlier_avg > 0 else 0.0
    return recent_avg / 20_000 if recent_avg > 0 else 0.0


def change_pct_5m(bars_5m: Optional[Sequence[Bar]]) -> Optional[float]:
    if not bars_5m or len(bars_5m) < 1:
        return None
    b = bars_5m[-1]
    if b.open <= 0:
        return None
    return (b.close - b.open) / b.open * 100


def change_pct_10m(bars: Sequence[Bar]) -> Optional[float]:
    if len(bars) < 10:
        return None
    start = bars[-10].open
    end = bars[-1].close
    if start <= 0:
        return None
    return (end - start) / start * 100


def classify_hod_momentum_alerts(
    *,
    price: float,
    float_shares: Optional[float],
    rel_vol: float,
    bar_rvol: float,
    change_session_pct: float,
    change_5m_pct: Optional[float],
    change_10m_pct: Optional[float],
    include_hod_breakout: bool = False,
    include_today_hod_breakout: bool = False,
    include_hod_reclaim: bool = False,
    max_float: float = 20_000_000,
) -> List[str]:
    """Return alert type names for low-float HOD momentum ($2–$20, float <= max_float)."""
    if float_shares is None or float_shares > max_float:
        return []

    alerts: List[str] = []

    if include_hod_breakout:
        alerts.append("New HOD Breakout")
    if include_today_hod_breakout:
        alerts.append("Today HOD Breakout")
    if include_hod_reclaim:
        alerts.append("HOD Reclaim")

    if rel_vol >= 3.0:
        alerts.append("Low Float - High Rel Vol")

    if change_5m_pct is not None and change_5m_pct >= 5.0:
        alerts.append("Squeeze - Up 5% in 5min")

    if change_10m_pct is not None and change_10m_pct >= 10.0:
        alerts.append("Squeeze - Up 10% in 10min")

    if bar_rvol >= 5.0 and "Squeeze - Up 5% in 5min" not in alerts:
        if change_session_pct >= 3.0:
            alerts.append("Squeeze - Up 5% in 5min")

    return alerts


def alert_row_class(alert_name: str) -> str:
    if alert_name == "Low Float - High Rel Vol":
        return "hod-low-float"
    if alert_name == "New HOD Breakout":
        return "hod-breakout"
    if alert_name == "Today HOD Breakout":
        return "hod-today-breakout"
    if alert_name == "HOD Reclaim":
        return "hod-reclaim"
    if alert_name == "Former Momo Stock":
        return "hod-former-momo"
    if alert_name == "Gapper Continuation":
        return "hod-gapper"
    if "Squeeze" in alert_name:
        return "hod-squeeze"
    return "hod-default"
