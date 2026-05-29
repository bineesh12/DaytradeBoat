"""HOD Momentum alert row model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HODAlertRow:
    """One row in the HOD Momentum dashboard table."""

    symbol: str
    time: str
    price: float
    alert_name: str
    source: str  # "tick" | "bar"
    day_volume: float = 0.0
    float_shares: Optional[float] = None
    rel_vol: float = 0.0
    rel_vol_5m_pct: Optional[float] = None
    bar_rvol: float = 0.0
    change_session_pct: float = 0.0
    change_from_low_pct: float = 0.0
    gap_pct: Optional[float] = None
    change_from_close_pct: Optional[float] = None
    verified: bool = False
    reject_reason: Optional[str] = None
    hot: bool = False
    burst_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        from daytrading.scanner.scanner_alert_labels import (
            alert_row_class,
            fmt_float,
            fmt_vol,
        )

        return {
            "time": self.time,
            "symbol": self.symbol,
            "price": round(self.price, 4),
            "day_volume": self.day_volume,
            "day_volume_fmt": fmt_vol(self.day_volume),
            "float_shares": self.float_shares,
            "float_fmt": fmt_float(self.float_shares) if self.float_shares else "—",
            "rel_vol": round(self.rel_vol, 2),
            "rel_vol_5m_pct": (
                round(self.rel_vol_5m_pct, 2)
                if self.rel_vol_5m_pct is not None else None
            ),
            "bar_rvol": round(self.bar_rvol, 2),
            "change_session_pct": round(self.change_session_pct, 2),
            "change_from_low_pct": round(self.change_from_low_pct, 2),
            "gap_pct": round(self.gap_pct, 2) if self.gap_pct is not None else None,
            "change_from_close_pct": (
                round(self.change_from_close_pct, 2)
                if self.change_from_close_pct is not None else None
            ),
            "alert_name": self.alert_name,
            "source": self.source,
            "row_class": alert_row_class(self.alert_name),
            "verified": self.verified,
            "reject_reason": self.reject_reason,
            "hot": self.hot,
            "burst_text": self.burst_text,
        }
