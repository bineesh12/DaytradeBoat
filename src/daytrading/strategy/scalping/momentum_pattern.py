"""Momentum pattern verifier — Bull Flag, Flat Top Breakout, VWAP Pullback,
Opening Range Breakout & HOD Reclaim.

Implements Warrior Trading's momentum day trading entry rules:
- Entry on the first candle making a new high after pullback (bull flag)
  or breaking above flat resistance (flat top breakout)
- First pullback to VWAP after a rally
- First 5-minute opening range breakout
- High-of-day reclaim after pullback
- Stop loss below the pattern low
- Target at 2:1 reward-to-risk ratio
- Trailing stop to lock profits
- Max 20 cent risk per share; if stop is wider, reduce position size
"""

from __future__ import annotations

import logging
from typing import Optional

from daytrading.indicators.core import vwap
from daytrading.models import PortfolioState, ScanResult, SignalAction, TradeSignal
from daytrading.strategy.entry_guard import check_entry_quality, record_rule_rejection

logger = logging.getLogger(__name__)

TICK = 0.01


class MomentumPatternVerifier:
    """Verifies bull flag and flat top breakout scan hits."""

    def __init__(
        self,
        *,
        max_risk_per_share: float = 0.50,
        reward_risk_ratio: float = 1.0,
        trail_ticks: int = 5,
        max_hold_seconds: int = 600,
        max_dollar_risk: float = 100.0,
        min_price: float = 1.0,
        max_price: float = 500.0,
        float_checker: object = None,
    ) -> None:
        self._max_risk = max_risk_per_share
        self._rr_ratio = reward_risk_ratio
        self._trail = trail_ticks * TICK
        self._max_hold = max_hold_seconds
        self._max_dollar_risk = max_dollar_risk
        self._min_price = min_price
        self._max_price = max_price
        self._float_checker = float_checker
        self._last_reject: Optional[str] = None
        self._bar_aggregator = None  # set by factory/runner for 5m context
        self._tick_buffer = None  # set by runner: Dict[str, deque] of recent ticks
        self._quote_buffer = None  # set by runner: Dict[str, deque] of recent quotes
        self._current_symbol: Optional[str] = None

    @property
    def name(self) -> str:
        return "momentum_pattern"

    def _reject(self, reason: str) -> None:
        self._last_reject = reason
        record_rule_rejection(symbol=self._current_symbol, reason=reason)

    def _get_avg_volume(self, symbol: str) -> Optional[float]:
        if self._float_checker is not None and hasattr(self._float_checker, "get_avg_volume"):
            return self._float_checker.get_avg_volume(symbol)
        return None

    def _get_float_shares(self, symbol: str) -> Optional[float]:
        if self._float_checker is not None and hasattr(self._float_checker, "get_float"):
            return self._float_checker.get_float(symbol)
        return None

    @staticmethod
    def _is_hot_hod_reclaim(pattern: str, scan_result: ScanResult, bars: list) -> bool:
        if pattern != "hod_reclaim" or not bars:
            return False
        price = bars[-1].close
        if price <= 0:
            return False
        hod = float(scan_result.criteria.get("hod") or max(b.high for b in bars))
        distance_from_hod = (hod - price) / hod * 100 if hod > 0 else 100.0
        rally_pct = float(scan_result.criteria.get("rally_pct") or 0.0)
        volume = float(scan_result.criteria.get("volume") or bars[-1].volume or 0.0)
        return (
            distance_from_hod <= 2.0
            and rally_pct >= 20.0
            and volume >= 100_000
            and scan_result.score >= 50.0
        )

    @staticmethod
    def _hot_hod_reclaim_stop(price: float, bars: list) -> float:
        """Use a tactical stop for explosive HOD reclaims.

        The technical pullback low can be far below price after a fast reclaim.
        For quick scalping, that creates an untradeable 15-20% risk even when
        momentum is clean. Cap this setup to a smaller scalp-style stop.
        """
        recent = bars[-2:] if len(bars) >= 2 else bars[-1:]
        recent_lows = [b.low for b in recent if b.low > 0]
        if recent_lows:
            recent_stop = min(recent_lows) - 0.02
            recent_risk_pct = (price - recent_stop) / price if price > 0 else 1.0
            if 0.005 < recent_risk_pct <= 0.08:
                return recent_stop
        return price * 0.94

    @staticmethod
    def _is_fresh_late_reclaim(
        pattern: str,
        bars: list,
        *,
        session_high: float,
        distance_from_hod: float,
        day_move_pct: float,
        scan_result: Optional[ScanResult] = None,
    ) -> bool:
        """Allow a late pullback only after a fresh base reclaim.

        This keeps the normal no-chase HOD-distance rule intact, but gives
        strong runners a second chance after they build a tight base and reclaim
        it with volume. It is meant for DXST/OLOX-style continuation watches,
        not for buying a falling pullback.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if session_high <= 0 or day_move_pct < 25.0:
            return False

        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        score = float(scan_result.score if scan_result is not None else 0.0)
        elite_a_plus_runner = (
            pattern == "vwap_pullback"
            and scan_result is not None
            and str(scan_result.criteria.get("setup_tier") or "") == "A+ setup"
            and score >= 80.0
            and day_move_pct >= 80.0
            and recent_volume >= 250_000
        )
        max_reclaim_distance = 22.0 if elite_a_plus_runner else (18.0 if pattern == "pullback_base" else 16.0)
        if distance_from_hod > max_reclaim_distance:
            if MomentumPatternVerifier._is_a_plus_deep_runner_reclaim(
                pattern,
                bars,
                session_high=session_high,
                distance_from_hod=distance_from_hod,
                day_move_pct=day_move_pct,
                scan_result=scan_result,
            ):
                return True
            if MomentumPatternVerifier._is_a_plus_reclaim_in_progress(
                pattern,
                bars,
                session_high=session_high,
                distance_from_hod=distance_from_hod,
                day_move_pct=day_move_pct,
                scan_result=scan_result,
            ):
                return True
            return False

        latest = bars[-1]
        if latest.close <= latest.open or latest.close <= 0:
            return False
        candle_range = latest.high - latest.low
        body_ratio = (
            (latest.close - latest.open) / candle_range
            if candle_range > 0 else 0.0
        )
        if body_ratio < 0.40:
            return False

        base = list(bars[-4:-1]) if elite_a_plus_runner else list(bars[-5:-1])
        if len(base) < 4:
            if not elite_a_plus_runner or len(base) < 3:
                return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0:
            return False
        base_range_pct = (base_high - base_low) / base_low * 100.0
        max_base_range_pct = 12.0 if elite_a_plus_runner else min(8.0, 4.0 + day_move_pct / 20.0)
        base_too_wide = base_range_pct > max_base_range_pct

        avg_base_vol = sum(b.volume for b in base) / len(base)
        if recent_volume < 75_000:
            return False
        if avg_base_vol > 0 and latest.volume < avg_base_vol * 1.10:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0:
            return False
        above_vwap_pct = (latest.close - current_vwap) / current_vwap * 100.0
        if above_vwap_pct < 1.0:
            return False

        if (
            not base_too_wide
            and latest.close > base_high * 1.002
            and latest.low >= base_low
        ):
            return True

        if MomentumPatternVerifier._is_a_plus_reclaim_in_progress(
            pattern,
            bars,
            session_high=session_high,
            distance_from_hod=distance_from_hod,
            day_move_pct=day_move_pct,
            scan_result=scan_result,
        ):
            return True

        # INHD/RMSG-style continuation: after an extreme runner cools off, the
        # safer second entry can be a VWAP trend reclaim before it retests HOD.
        # Keep it narrow so ordinary late pullbacks still wait for base-high
        # reclaim.
        if pattern == "pullback_base":
            prev = bars[-2]
            recent_base = list(bars[-4:-1])
            recent_base_low = min(b.low for b in recent_base) if recent_base else base_low
            recent_base_high = max(b.high for b in recent_base) if recent_base else base_high
            recent_base_range_pct = (
                (recent_base_high - recent_base_low) / recent_base_low * 100.0
                if recent_base_low > 0 else base_range_pct
            )
            holding_vwap = min(b.low for b in bars[-3:]) >= current_vwap * 0.985
            reclaiming_up = latest.close > prev.close * 1.008 and latest.low >= prev.low
            massive_runner = day_move_pct >= 80.0 and distance_from_hod <= 12.5
            if (
                massive_runner
                and holding_vwap
                and reclaiming_up
                and recent_base_range_pct <= 11.0
                and recent_volume >= 150_000
            ):
                return True

        return False

    @staticmethod
    def _is_a_plus_reclaim_in_progress(
        pattern: str,
        bars: list,
        *,
        session_high: float,
        distance_from_hod: float,
        day_move_pct: float,
        scan_result: Optional[ScanResult] = None,
    ) -> bool:
        """Allow a tiny scout while an A+ runner is reclaiming, before perfection.

        This catches FLD/DSY-style runners where the old HOD is still far away,
        but price is above VWAP, buyers are returning, and the latest candle has
        already reclaimed most of the local base. It is not a full-size entry.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if scan_result is None:
            return False
        if str(scan_result.criteria.get("setup_tier") or "") != "A+ setup":
            return False

        latest = bars[-1]
        price = latest.close
        score = float(scan_result.score or 0.0)
        if price <= 0 or latest.close <= latest.open:
            return False
        if score < 80.0 or day_move_pct < 80.0:
            return False
        if not (12.0 <= distance_from_hod <= 32.0):
            return False

        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent = list(bars[-3:])
        recent_volume = sum(float(b.volume or 0.0) for b in recent)
        latest_volume = float(latest.volume or 0.0)
        latest_volume_floor = 7_000 if price >= 5.0 else 40_000
        if (
            day_volume < 1_000_000
            or recent_volume < 120_000
            or latest_volume < latest_volume_floor
        ):
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or price < current_vwap * 1.005:
            return False

        base = list(bars[-4:-1])
        if len(base) < 3:
            return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0 or base_high <= 0:
            return False
        base_range_pct = (base_high - base_low) / base_low * 100.0
        max_base_range_pct = 18.0 if pattern == "pullback_base" else 20.0
        if pattern == "vwap_pullback" and latest_volume >= 500_000:
            max_base_range_pct = 40.0
        if base_range_pct > max_base_range_pct:
            return False

        base_close_high = max(b.close for b in base)
        reclaim_progress = (
            (price - base_low) / (base_high - base_low)
            if base_high > base_low else 1.0
        )
        holding_base = latest.low >= base_low * 0.985
        reclaiming_close = price >= base_close_high * 0.995
        reclaiming_body = price > bars[-2].close * 1.006
        progress_floor = 0.30 if pattern == "pullback_base" else 0.55
        if not (
            holding_base
            and reclaim_progress >= progress_floor
            and (reclaiming_close or reclaiming_body)
        ):
            return False

        scan_result.criteria["entry_tier"] = "a_plus_reclaim_scout"
        scan_result.criteria["entry_tier_reason"] = (
            "A+ runner reclaim in progress above VWAP; reduced-size scout before full HOD reclaim"
        )
        scan_result.criteria["reclaim_progress_pct"] = round(reclaim_progress * 100.0, 1)
        scan_result.criteria["reclaim_distance_from_hod_pct"] = round(distance_from_hod, 2)
        scan_result.criteria["reclaim_vwap"] = round(current_vwap, 4)
        return True

    @staticmethod
    def _is_a_plus_deep_runner_reclaim(
        pattern: str,
        bars: list,
        *,
        session_high: float,
        distance_from_hod: float,
        day_move_pct: float,
        scan_result: Optional[ScanResult] = None,
    ) -> bool:
        """Allow a small second-chance scout on extreme A+ runner reclaims.

        This targets VSME/DSY-style runners that are too far below the old HOD
        for a normal pullback entry, but have built a fresh local reclaim above
        VWAP. It stays reduced-size and still goes through entry guard.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if scan_result is None:
            return False
        if str(scan_result.criteria.get("setup_tier") or "") != "A+ setup":
            return False

        score = float(scan_result.score or 0.0)
        latest = bars[-1]
        price = latest.close
        if price <= 0 or latest.close <= latest.open:
            return False
        if score < 80.0 or day_move_pct < 100.0:
            return False
        if not (22.0 < distance_from_hod <= 45.0):
            return False

        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent = list(bars[-3:])
        recent_volume = sum(float(b.volume or 0.0) for b in recent)
        latest_volume = float(latest.volume or 0.0)
        if day_volume < 1_000_000 or recent_volume < 100_000 or latest_volume < 15_000:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or latest.close < current_vwap * 1.015:
            return False

        base = list(bars[-4:-1])
        if len(base) < 3:
            return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0 or base_high <= 0:
            return False
        base_range_pct = (base_high - base_low) / base_low * 100.0
        if base_range_pct > 22.0:
            return False

        prior_close_high = max(b.close for b in base)
        reclaiming_local_level = latest.close > prior_close_high * 1.008
        holding_higher_low = latest.low >= base_low * 0.985
        not_stalling_at_lows = latest.close >= base_low + (base_high - base_low) * 0.55
        if not (reclaiming_local_level and holding_higher_low and not_stalling_at_lows):
            return False

        scan_result.criteria["entry_tier"] = "deep_runner_scout"
        scan_result.criteria["entry_tier_reason"] = (
            "A+ runner reclaimed a fresh local base far below old HOD; reduced-size scout only"
        )
        scan_result.criteria["deep_reclaim_distance_from_hod_pct"] = round(distance_from_hod, 2)
        scan_result.criteria["deep_reclaim_vwap"] = round(current_vwap, 4)
        scan_result.criteria["deep_reclaim_session_high"] = round(session_high, 4)
        return True

    @staticmethod
    def _is_hot_gapper_pullback(
        bars: list,
        *,
        day_move_pct: float,
        float_shares: Optional[float],
    ) -> bool:
        """Allow controlled pullbacks on strong gapper-style runners.

        XOS-style setups can have huge prior-close gap momentum while the
        intraday open-to-HOD move is only 12-20%. Require strong tape and
        low-float characteristics so this does not weaken normal pullbacks.
        """
        if len(bars) < 10 or day_move_pct < 12.0:
            return False
        if float_shares is not None and float_shares > 10_000_000:
            return False

        latest = bars[-1]
        price = latest.close
        if price < 1.5 or price > 20.0:
            return False

        session_high = max(b.high for b in bars)
        pullback_pct = (session_high - price) / session_high * 100 if session_high > 0 else 100.0
        if not (3.0 <= pullback_pct <= 8.0):
            return False

        day_volume = sum(b.volume for b in bars)
        recent = bars[-5:] if len(bars) >= 5 else bars
        recent_avg = sum(b.volume for b in recent) / len(recent)
        earlier = bars[:-5]
        earlier_avg = sum(b.volume for b in earlier) / len(earlier) if earlier else 0.0
        bar_rvol = recent_avg / earlier_avg if earlier_avg > 0 else 0.0

        return (
            day_volume >= 2_000_000
            and recent_avg >= 75_000
            and bar_rvol >= 1.5
        )

    @staticmethod
    def _late_pullback_reject(
        pattern: str,
        bars: list,
        *,
        float_shares: Optional[float] = None,
        scan_result: Optional[ScanResult] = None,
    ) -> Optional[str]:
        """Extra quality gate for slower pullback entries."""
        if len(bars) < 10:
            return "late pullback needs more bars"

        latest = bars[-1]
        price = latest.close
        if price <= 0:
            return "invalid price"

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return "invalid session range"

        day_move_pct = (session_high - session_open) / session_open * 100
        distance_from_hod = (session_high - price) / session_high * 100
        if scan_result is not None:
            scan_result.criteria.pop("entry_tier", None)
            scan_result.criteria.pop("entry_tier_reason", None)

        fresh_late_reclaim = False
        max_hod_distance = 8.0 if pattern == "vwap_pullback" else 10.0
        if distance_from_hod > max_hod_distance:
            fresh_late_reclaim = MomentumPatternVerifier._is_fresh_late_reclaim(
                pattern,
                bars,
                session_high=session_high,
                distance_from_hod=distance_from_hod,
                day_move_pct=day_move_pct,
                scan_result=scan_result,
            )
            if not fresh_late_reclaim:
                return (
                    "late pullback too far from HOD {:.1f}% "
                    "(max {:.1f}%; watching for fresh reclaim)"
                ).format(distance_from_hod, max_hod_distance)
            logger.info(
                "SCANNER LATE RECLAIM %s: %.1f%% from HOD but fresh base reclaim",
                pattern, distance_from_hod,
            )
            if scan_result is not None and not scan_result.criteria.get("entry_tier"):
                scan_result.criteria["entry_tier"] = "a_plus_retry_watch"
                scan_result.criteria["entry_tier_reason"] = (
                    "A+ runner reclaimed a fresh base after scanner reject; "
                    "retry still requires full entry guard/final confirmation"
                )

        if pattern == "pullback_base" and day_move_pct < 20.0:
            hot_gapper_pullback = MomentumPatternVerifier._is_hot_gapper_pullback(
                bars,
                day_move_pct=day_move_pct,
                float_shares=float_shares,
            )
            scoutable_mid_move = (
                15.0 <= day_move_pct < 20.0
                and distance_from_hod <= 8.0
                and scan_result is not None
                and float(scan_result.score or 0.0) >= 60.0
                and float(bars[-1].volume or 0) > 0
            )
            if not hot_gapper_pullback and not scoutable_mid_move:
                return (
                    "pullback base move too small {:.1f}% "
                    "(need 20%+ full size or 15%+ scout-quality pullback)"
                ).format(day_move_pct)
            if scoutable_mid_move:
                if scan_result is not None:
                    scan_result.criteria["entry_tier"] = "pullback_scout"
                    scan_result.criteria["entry_tier_reason"] = (
                        "15-20% mid-move pullback; reduced-size scout only"
                    )
                logger.info(
                    "SCANNER MID-MOVE PULLBACK SCOUT: %.1f%% intraday move "
                    "near HOD allowed as reduced-size scout candidate",
                    day_move_pct,
                )
            elif hot_gapper_pullback:
                if scan_result is not None:
                    scan_result.criteria.pop("entry_tier", None)
                    scan_result.criteria.pop("entry_tier_reason", None)
                logger.info(
                    "SCANNER HOT GAPPER PULLBACK: %.1f%% intraday move allowed "
                    "with strong low-float tape",
                    day_move_pct,
                )

        structure_reject = MomentumPatternVerifier._pullback_structure_reject(
            pattern,
            bars,
            scan_result=scan_result,
        )
        if structure_reject is not None:
            return structure_reject

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0:
            above_vwap_pct = (price - current_vwap) / current_vwap * 100
            if above_vwap_pct < 1.0:
                return (
                    "late pullback not strong above VWAP {:.1f}% "
                    "(need 1.0%+)"
                ).format(above_vwap_pct)

        recent = bars[-3:]
        recent_volume = sum(b.volume for b in recent)
        recent_avg = recent_volume / len(recent)
        earlier = bars[:-3]
        earlier_avg = (
            sum(b.volume for b in earlier) / len(earlier)
            if earlier else 0.0
        )
        if recent_volume < 75_000:
            return (
                "late pullback tape too slow {:.0f} recent volume "
                "(need 75K+)"
            ).format(recent_volume)
        if (
            earlier_avg > 0
            and recent_avg < earlier_avg * 0.60
            and not fresh_late_reclaim
        ):
            return (
                "late pullback volume faded {:.1f}x prior "
                "(need 0.6x+)"
            ).format(recent_avg / earlier_avg)

        return None

    @staticmethod
    def _pullback_structure_reject(
        pattern: str,
        bars: list,
        *,
        scan_result: Optional[ScanResult] = None,
    ) -> Optional[str]:
        """Reject pullbacks where 1-minute structure still favors sellers."""
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 8:
            return None

        latest = bars[-1]
        if latest.close <= latest.open:
            return "pullback reclaim candle not green"

        pullback = list(bars[-5:-1]) if len(bars) >= 6 else list(bars[:-1])
        if len(pullback) < 2:
            return None

        red_pullback = [b for b in pullback if b.close < b.open]
        if not red_pullback:
            return None

        first_red_idx = next(
            (idx for idx in range(len(bars) - 5, len(bars) - 1)
             if idx >= 0 and bars[idx].close < bars[idx].open),
            None,
        )
        if first_red_idx is None:
            return None
        impulse = list(bars[max(0, first_red_idx - 5):first_red_idx])
        if not impulse:
            return None

        impulse_green = [b for b in impulse if b.close > b.open]
        impulse_volumes = sorted(
            [b.volume for b in (impulse_green or impulse)],
            reverse=True,
        )
        top_n = min(2, len(impulse_volumes))
        impulse_avg_vol = (
            sum(impulse_volumes[:top_n]) / top_n
            if top_n else 0.0
        )
        if impulse_avg_vol <= 0:
            return None

        red_avg_vol = sum(b.volume for b in red_pullback) / len(red_pullback)
        heaviest_red = max(red_pullback, key=lambda b: b.volume)
        red_body_pct = (
            (heaviest_red.open - heaviest_red.close) / heaviest_red.open * 100.0
            if heaviest_red.open > 0 else 0.0
        )
        red_range_pct = (
            (heaviest_red.high - heaviest_red.low) / heaviest_red.open * 100.0
            if heaviest_red.open > 0 else 0.0
        )

        red_volume_ratio = red_avg_vol / impulse_avg_vol
        elite_runner_mild_red_pullback = (
            MomentumPatternVerifier._allows_elite_runner_mild_red_pullback(
                pattern,
                bars,
                scan_result=scan_result,
                red_volume_ratio=red_volume_ratio,
                red_body_pct=red_body_pct,
                red_range_pct=red_range_pct,
            )
        )
        elite_runner_aggressive_reclaim = (
            MomentumPatternVerifier._allows_elite_runner_aggressive_reclaim(
                pattern,
                bars,
                scan_result=scan_result,
                red_volume_ratio=red_volume_ratio,
                red_body_pct=red_body_pct,
                red_range_pct=red_range_pct,
            )
        )

        # A pullback should normally be quieter than the impulse. For extreme
        # HOD runners, a mildly heavier red pullback can still be valid if the
        # reclaim candle proves buyers came back. Keep this exception narrow so
        # real distribution still gets blocked.
        if (
            red_avg_vol >= impulse_avg_vol * 1.10
            and not elite_runner_mild_red_pullback
            and not elite_runner_aggressive_reclaim
        ):
            return (
                "pullback red volume too heavy {:.1f}x impulse "
                "(need lower-volume pullback)"
            ).format(red_volume_ratio)

        # One oversized red candle after a vertical push is the ANY-style trap:
        # it can still be above VWAP, but sellers are in control until a fresh
        # base/reclaim forms.
        if (
            (red_body_pct >= 3.5 or red_range_pct >= 6.0)
            and heaviest_red.volume >= impulse_avg_vol * 0.75
        ):
            if MomentumPatternVerifier._allows_a_plus_dump_reclaim_scout(
                pattern,
                bars,
                scan_result=scan_result,
                red_body_pct=red_body_pct,
                red_range_pct=red_range_pct,
            ):
                return None
            return (
                "pullback has dump candle {:.1f}% body/{:.1f}% range "
                "(wait for new base)"
            ).format(red_body_pct, red_range_pct)

        red_avg = red_avg_vol
        if red_avg > 0 and latest.volume < red_avg * 0.80:
            tier = str(scan_result.criteria.get("entry_tier") or "").lower() if scan_result else ""
            if tier == "a_plus_reclaim_scout":
                session_open = bars[0].open if bars else 0.0
                session_high = max((b.high for b in bars), default=0.0)
                day_move_pct = (
                    (session_high - session_open) / session_open * 100.0
                    if session_open > 0 and session_high > 0 else 0.0
                )
                day_volume = sum(float(b.volume or 0.0) for b in bars)
                recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
                if day_move_pct >= 100.0 and day_volume >= 1_000_000 and recent_volume >= 100_000:
                    return None
            return (
                "pullback reclaim volume weak {:.1f}x red pullback "
                "(need buyers returning)"
            ).format(latest.volume / red_avg)

        return None

    @staticmethod
    def _allows_a_plus_dump_reclaim_scout(
        pattern: str,
        bars: list,
        *,
        scan_result: Optional[ScanResult],
        red_body_pct: float,
        red_range_pct: float,
    ) -> bool:
        """Allow only confirmed A+ reclaim scouts after a prior dump candle.

        The first dump candle stays blocked. This exception is for the later
        candle that has already rebuilt above VWAP/base with heavy participation.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if scan_result is None:
            return False
        if str(scan_result.criteria.get("setup_tier") or "") != "A+ setup":
            return False
        if str(scan_result.criteria.get("entry_tier") or "").lower() != "a_plus_reclaim_scout":
            return False
        if red_body_pct > 8.0 or red_range_pct > 14.0:
            return False

        latest = bars[-1]
        if latest.close <= latest.open or latest.close <= 0:
            return False

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return False
        day_move_pct = (session_high - session_open) / session_open * 100.0
        distance_from_hod = (session_high - latest.close) / session_high * 100.0
        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        latest_volume = float(latest.volume or 0.0)

        if day_move_pct < 100.0 or distance_from_hod > 32.0:
            return False
        if day_volume < 2_000_000 or recent_volume < 250_000 or latest_volume < 75_000:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or latest.close < current_vwap * 1.01:
            return False

        base = list(bars[-4:-1])
        if len(base) < 3:
            return False
        base_high = max(b.high for b in base)
        base_low = min(b.low for b in base)
        if base_low <= 0 or base_high <= base_low:
            return False
        reclaim_progress = (latest.close - base_low) / (base_high - base_low)
        if reclaim_progress < 0.70 or latest.low < base_low * 0.985:
            return False

        scan_result.criteria["entry_tier_reason"] = (
            "A+ runner reclaimed after dump candle; reduced-size scout still requires guard"
        )
        return True

    @staticmethod
    def _allows_elite_runner_mild_red_pullback(
        pattern: str,
        bars: list,
        *,
        scan_result: Optional[ScanResult],
        red_volume_ratio: float,
        red_body_pct: float,
        red_range_pct: float,
    ) -> bool:
        """Allow PAVS-style mild red volume only on elite runners.

        This is not a broad loosening. It targets high-volume runners where the
        red pullback is only slightly heavier than the impulse, the latest
        candle has already reclaimed, and the setup is still close enough to HOD.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if not (1.10 <= red_volume_ratio <= 1.35):
            return False
        if red_body_pct >= 3.5 or red_range_pct >= 8.0:
            return False

        latest = bars[-1]
        if latest.close <= latest.open:
            return False

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return False
        day_move_pct = (session_high - session_open) / session_open * 100.0
        distance_from_hod = (session_high - latest.close) / session_high * 100.0
        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        score = float(scan_result.score if scan_result is not None else 0.0)

        if day_move_pct < 80.0 or distance_from_hod > 8.0:
            return False
        if day_volume < 2_000_000 or recent_volume < 150_000 or score < 80.0:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or latest.close < current_vwap * 1.01:
            return False

        pullback = list(bars[-5:-1])
        base_low = min(b.low for b in pullback) if pullback else latest.low
        if base_low <= 0 or latest.low < base_low * 0.985:
            return False

        return True

    @staticmethod
    def _allows_elite_runner_aggressive_reclaim(
        pattern: str,
        bars: list,
        *,
        scan_result: Optional[ScanResult],
        red_volume_ratio: float,
        red_body_pct: float,
        red_range_pct: float,
    ) -> bool:
        """Allow a narrow DSY-style retry after an A+ runner proves reclaim.

        This does not allow the first heavy red pullback. It only allows an
        already-identified A+ runner to retry when price is close to HOD again,
        the latest candle is a reclaim candle, and participation is extreme.
        """
        if pattern not in ("vwap_pullback", "pullback_base") or len(bars) < 10:
            return False
        if scan_result is None:
            return False
        if str(scan_result.criteria.get("setup_tier") or "") != "A+ setup":
            return False
        if not (1.35 < red_volume_ratio <= 2.25):
            return False
        if red_body_pct >= 3.5 or red_range_pct >= 8.0:
            return False

        latest = bars[-1]
        if latest.close <= latest.open or latest.close <= 0:
            return False

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return False
        day_move_pct = (session_high - session_open) / session_open * 100.0
        distance_from_hod = (session_high - latest.close) / session_high * 100.0
        day_volume = sum(float(b.volume or 0.0) for b in bars)
        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
        score = float(scan_result.score or 0.0)

        if day_move_pct < 100.0 or distance_from_hod > 4.0:
            return False
        if day_volume < 5_000_000 or recent_volume < 500_000 or score < 100.0:
            return False

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap <= 0 or latest.close < current_vwap * 1.03:
            return False

        base = list(bars[-4:-1])
        base_high = max((b.high for b in base), default=0.0)
        base_low = min((b.low for b in base), default=0.0)
        if base_high <= 0 or base_low <= 0:
            return False
        base_range_pct = (base_high - base_low) / base_low * 100.0
        if base_range_pct > 12.0:
            return False
        if latest.close <= base_high * 1.002 or latest.low < base_low * 0.985:
            return False

        if not scan_result.criteria.get("entry_tier"):
            scan_result.criteria["entry_tier"] = "a_plus_retry_watch"
            scan_result.criteria["entry_tier_reason"] = (
                "elite A+ runner reclaimed after heavy red pullback; reduced-size retry"
            )
        return True

    @staticmethod
    def _late_continuation_reject(
        pattern: str,
        scan_result: ScanResult,
        bars: list,
    ) -> Optional[str]:
        """Extra no-chase gate for late bull-flag/flat-top continuations.

        These patterns can fire after a stock has already made the clean move.
        Keep them tradable only when the breakout is still strong: above VWAP,
        enough live volume, and reclaiming structure instead of drifting.
        """
        if pattern not in ("bull_flag", "flat_top_breakout"):
            return None
        if len(bars) < 8:
            return "late continuation needs more bars"

        latest = bars[-1]
        price = latest.close
        if price <= 0:
            return "invalid price"

        session_open = bars[0].open
        session_high = max(b.high for b in bars)
        if session_open <= 0 or session_high <= 0:
            return "invalid session range"

        day_move_pct = (session_high - session_open) / session_open * 100.0
        if scan_result.score < 3.0 and day_move_pct < 20.0:
            return (
                "late continuation too weak score {:.1f} "
                "(need stronger pattern or 20%+ move)"
            ).format(scan_result.score)

        vwap_vals = vwap(bars)
        current_vwap = vwap_vals[-1] if vwap_vals else 0.0
        if current_vwap > 0:
            above_vwap_pct = (price - current_vwap) / current_vwap * 100.0
            needed = 0.75 if scan_result.score < 5.0 else 0.50
            if above_vwap_pct < needed:
                return (
                    "late continuation weak VWAP {:.1f}% "
                    "(need {:.1f}%+)"
                ).format(above_vwap_pct, needed)

        recent = bars[-3:]
        recent_volume = sum(b.volume for b in recent)
        if recent_volume < 100_000:
            return (
                "late continuation tape too slow {:.0f} recent volume "
                "(need 100K+)"
            ).format(recent_volume)

        prior = bars[-4:-1]
        if prior:
            prior_close_high = max(b.close for b in prior)
            if price <= prior_close_high * 1.002:
                return "late continuation not reclaiming prior close"

            prior_low = min(b.low for b in prior)
            if latest.low < prior_low and scan_result.score < 8.0:
                return "late continuation no higher-low structure"

        return None

    @staticmethod
    def _pattern_active_tape_reject(
        pattern: str,
        scan_result: ScanResult,
        bars: list,
    ) -> Optional[str]:
        """Pattern-specific tape checks before generic entry guard.

        The generic guard protects account risk. This layer protects setup
        quality: breakout setups need volume at the level, while continuation
        setups need enough live participation to avoid slow chop.
        """
        if len(bars) < 3:
            return None

        latest = bars[-1]
        price = latest.close
        latest_volume = float(
            scan_result.criteria.get("volume") or latest.volume or 0.0
        )
        recent = bars[-3:]
        recent_volume = float(sum(b.volume for b in recent))
        recent_avg = recent_volume / len(recent)

        if pattern == "level_breakout_reclaim":
            breakout_level = float(scan_result.criteria.get("breakout_level") or 0.0)
            if breakout_level > 0 and price > breakout_level * 1.025:
                return (
                    "breakout already extended {:.1f}% above level "
                    "(need <=2.5%)"
                ).format((price - breakout_level) / breakout_level * 100.0)
            if latest_volume < 100_000:
                return (
                    "breakout volume too light {:.0f} "
                    "(need 100K+ at level)"
                ).format(latest_volume)

        if pattern == "shallow_stair_continuation":
            if recent_volume < 150_000:
                return (
                    "stair-step tape too slow {:.0f} recent volume "
                    "(need 150K+)"
                ).format(recent_volume)
            base_high = float(scan_result.criteria.get("base_high") or 0.0)
            if base_high > 0 and price <= base_high * 1.002:
                return "stair-step breakout lost mini-level"

        if pattern == "early_vwap_reclaim_scout":
            if recent_volume < 60_000 and latest_volume < 25_000:
                return (
                    "early VWAP reclaim tape too slow {:.0f} recent / {:.0f} latest "
                    "(need buyers active)"
                ).format(recent_volume, latest_volume)
            vwap_level = float(scan_result.criteria.get("vwap") or 0.0)
            if vwap_level > 0 and price < vwap_level * 1.003:
                return "early VWAP reclaim not holding above VWAP"

        if pattern in ("abc_continuation", "first_pullback_reclaim"):
            if recent_volume < 60_000 and latest_volume < 25_000:
                return (
                    "setup tape too slow {:.0f} recent / {:.0f} latest "
                    "(need buyers active)"
                ).format(recent_volume, latest_volume)

        if pattern == "hod_reclaim":
            if recent_volume < 100_000 and latest_volume < 50_000:
                return (
                    "HOD reclaim tape too light {:.0f} recent / {:.0f} latest "
                    "(need real reclaim volume)"
                ).format(recent_volume, latest_volume)

        if pattern == "pullback_base":
            base = bars[-6:-1] if len(bars) >= 7 else bars[:-1]
            if base:
                session_open = bars[0].open if bars else 0.0
                session_high = max((b.high for b in bars), default=0.0)
                day_move_pct = (
                    (session_high - session_open) / session_open * 100.0
                    if session_open > 0 and session_high > 0 else 0.0
                )
                if day_move_pct >= 80.0 and len(bars) >= 5:
                    base = bars[-4:-1]
                base_avg = sum(b.volume for b in base) / len(base)
                if base_avg > 0 and latest.volume < base_avg * 0.70:
                    tier = str(scan_result.criteria.get("entry_tier") or "").lower()
                    if tier == "a_plus_reclaim_scout":
                        day_volume = sum(float(b.volume or 0.0) for b in bars)
                        recent_volume = sum(float(b.volume or 0.0) for b in bars[-3:])
                        if day_move_pct >= 100.0 and day_volume >= 1_000_000 and recent_volume >= 100_000:
                            return None
                    return (
                        "base reclaim volume weak {:.1f}x base "
                        "(need buyers returning)"
                    ).format(latest.volume / base_avg)

        if price >= 5.0 and recent_avg < 20_000 and latest_volume < 20_000:
            return (
                "active tape too thin {:.0f} recent avg / {:.0f} latest "
                "(need 20K+)"
            ).format(recent_avg, latest_volume)

        return None

    @staticmethod
    def _setup_quality_factor(pattern: str, scan_result: ScanResult, bars: list) -> tuple[float, str]:
        """Return a conservative sizing factor for lower-confidence entries."""
        tier = str(scan_result.criteria.get("entry_tier") or "").lower()
        if tier == "abc_scout":
            return 0.35, "A+ ABC scout"
        if tier == "pullback_scout":
            return 0.35, "pullback scout"
        if tier == "a_plus_reclaim_scout":
            return 0.35, "A+ reclaim scout"
        if tier == "deep_runner_scout":
            return 0.35, "A+ deep runner scout"
        if tier == "stair_scout":
            return 0.45, "stair-step scout"
        if tier == "level_scout":
            return 0.45, "level breakout scout"
        if pattern == "early_vwap_reclaim_scout":
            return 0.45, "early VWAP reclaim scout"

        score = float(scan_result.score or 0.0)
        latest_volume = float(scan_result.criteria.get("volume") or bars[-1].volume or 0.0)
        recent_volume = float(sum(b.volume for b in bars[-3:])) if len(bars) >= 3 else latest_volume

        if pattern == "level_breakout_reclaim" and score >= 25.0 and latest_volume >= 200_000:
            return 1.0, "A breakout"
        if pattern in ("vwap_pullback", "first_pullback_reclaim", "abc_continuation") and recent_volume >= 150_000:
            return 1.0, "A pullback"
        if pattern in ("hod_reclaim", "pullback_base") and recent_volume >= 200_000:
            return 1.0, "A momentum"
        return 0.70, "normal quality"

    @staticmethod
    def _abc_scout_tactical_stop(price: float, scan_result: ScanResult, bars: list) -> Optional[float]:
        """Return a tighter reduced-size stop for exceptional ABC continuations.

        A strong ABC can be valid while the full B-low stop is too wide for a
        scalp. In that case we only allow a scout and use the current reclaim
        candle as the invalidation area, so a failed reclaim exits quickly.
        """
        if price <= 0 or len(bars) < 3:
            return None
        if str(scan_result.criteria.get("setup_tier") or "") != "A+ setup":
            return None

        volume = float(scan_result.criteria.get("volume") or bars[-1].volume or 0.0)
        c_volume_surge = float(scan_result.criteria.get("c_volume_surge") or 0.0)
        b_retrace = float(scan_result.criteria.get("b_retrace_pct") or 0.0)
        if volume < 500_000 or c_volume_surge < 1.25:
            return None
        session_open = bars[0].open if bars else 0.0
        session_high = max((b.high for b in bars), default=0.0)
        day_move_pct = (
            (session_high - session_open) / session_open * 100.0
            if session_open > 0 and session_high > 0 else 0.0
        )
        distance_from_hod = (
            (session_high - price) / session_high * 100.0
            if session_high > 0 else 100.0
        )
        elite_sub2_reclaim = (
            1.5 <= price < 2.0
            and volume >= 1_000_000
            and c_volume_surge >= 1.5
            and day_move_pct >= 80.0
            and distance_from_hod <= 12.0
        )
        max_b_retrace = 55.0 if elite_sub2_reclaim else 45.0
        if not (20.0 <= b_retrace <= max_b_retrace):
            return None

        latest = bars[-1]
        if latest.close <= latest.open:
            return None
        body_range = latest.high - latest.low
        body_ratio = (
            (latest.close - latest.open) / body_range
            if body_range > 0 else 0.0
        )
        min_body_ratio = 0.15 if elite_sub2_reclaim else 0.30
        if body_ratio < min_body_ratio:
            return None

        raw_stop = min(float(latest.low or price), price * 0.96) - 0.02
        capped_stop = max(raw_stop, price * 0.94)
        risk_pct = (price - capped_stop) / price
        if 0.005 < risk_pct <= 0.08:
            return capped_stop
        return None

    def verify(
        self,
        scan_result: ScanResult,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        self._current_symbol = scan_result.symbol
        bars = scan_result.bars
        if len(bars) < 3:
            self._reject("insufficient bar data")
            return None

        latest = bars[-1]
        price = latest.close

        if not (self._min_price <= price <= self._max_price):
            self._reject("price ${:.2f} outside range".format(price))
            return None

        pos = portfolio.positions.get(scan_result.symbol)
        if pos and not pos.is_flat:
            self._reject("already in position")
            return None

        pattern = scan_result.criteria.get("pattern", "")
        direction = scan_result.criteria.get("direction", "")
        if direction != "up":
            self._reject("not an upward pattern")
            return None

        # Momentum burst scanner hits use the recent low as stop
        is_momentum_burst = scan_result.scanner_name == "momentum_burst"

        known_patterns = (
            "bull_flag", "flat_top_breakout", "vwap_pullback",
            "opening_range_breakout", "hod_reclaim", "pullback_base",
            "abc_continuation", "first_pullback_reclaim",
            "level_breakout_reclaim", "runner_reclaim_continuation",
            "shallow_stair_continuation", "early_vwap_reclaim_scout",
            "level_breakout_watch",
        )
        if not is_momentum_burst and pattern not in known_patterns:
            self._reject("unknown pattern: {}".format(pattern))
            return None
        if pattern == "level_breakout_watch":
            self._reject(
                "watch only: level breakout has not reclaimed with a clean close"
            )
            return None

        float_shares = self._get_float_shares(scan_result.symbol)

        if pattern in ("vwap_pullback", "pullback_base"):
            late_reject = self._late_pullback_reject(
                pattern,
                bars,
                float_shares=float_shares,
                scan_result=scan_result,
            )
            if late_reject is not None:
                logger.info("SCANNER REJECT %s: %s", scan_result.symbol, late_reject)
                self._reject(late_reject)
                return None

        if pattern in ("bull_flag", "flat_top_breakout"):
            continuation_reject = self._late_continuation_reject(
                pattern,
                scan_result,
                bars,
            )
            if continuation_reject is not None:
                logger.info("SCANNER REJECT %s: %s", scan_result.symbol, continuation_reject)
                self._reject(continuation_reject)
                return None

        tape_reject = self._pattern_active_tape_reject(pattern, scan_result, bars)
        if tape_reject is not None:
            logger.info("SCANNER REJECT %s: %s", scan_result.symbol, tape_reject)
            self._reject(tape_reject)
            return None

        bars_5m = None
        if self._bar_aggregator is not None:
            bars_5m = self._bar_aggregator.get_5m_bars(scan_result.symbol)

        reject = check_entry_quality(
            bars,
            symbol=scan_result.symbol,
            avg_daily_volume=self._get_avg_volume(scan_result.symbol),
            bars_5m=bars_5m,
            float_shares=float_shares,
            ticks=list(self._tick_buffer.get(scan_result.symbol, [])) if self._tick_buffer else None,
            quotes=list(self._quote_buffer.get(scan_result.symbol, [])) if self._quote_buffer else None,
            entry_pattern=str(pattern or scan_result.scanner_name),
            setup_tier=str(scan_result.criteria.get("setup_tier") or ""),
            entry_tier=str(scan_result.criteria.get("entry_tier") or ""),
        )
        if reject is not None:
            logger.info("ENTRY GUARD REJECT %s: %s", scan_result.symbol, reject)
            self._last_reject = reject
            return None

        logger.info("ENTRY GUARD PASS %s pattern=%s price=%.2f",
                     scan_result.symbol, pattern or scan_result.scanner_name, price)

        # --- Stop placement: Warrior Trading rule ---
        # The stop goes at the technical level where the pattern is invalidated.
        # We do NOT clamp it to an arbitrary % range.
        # If the risk per share is too large for our account, we take FEWER
        # shares to keep total dollar risk at $100. If the risk is absurdly
        # wide (>10% of price), we skip the trade entirely — the setup is
        # too loose to scalp.

        if is_momentum_burst:
            pattern = "momentum_burst"
            # For big movers, use last 3 bars (tighter stop near current action)
            # For normal stocks, use last 5 bars
            lookback = 3 if len(bars) >= 3 and (bars[-1].close / bars[0].open - 1) > 0.5 else 5
            lookback = min(lookback, len(bars))
            recent_low = min(b.low for b in bars[-lookback:]) if len(bars) >= lookback else price * 0.97
            pattern_stop = recent_low - 0.02
        elif pattern == "bull_flag":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            if pullback_low <= 0:
                self._reject("no pullback low")
                return None
            pattern_stop = pullback_low - 0.02
        elif pattern == "flat_top_breakout":
            resistance = float(scan_result.criteria.get("resistance", 0))
            if resistance <= 0:
                self._reject("no resistance level")
                return None
            pattern_stop = resistance - 0.05
        elif pattern == "vwap_pullback":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "opening_range_breakout":
            orb_high = float(scan_result.criteria.get("orb_high", 0))
            orb_low = float(scan_result.criteria.get("orb_low", 0))
            if orb_high > 0:
                pattern_stop = orb_high - 0.02
            elif orb_low > 0:
                pattern_stop = orb_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "hod_reclaim":
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "pullback_base":
            base_low = float(scan_result.criteria.get("base_low", 0))
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            if base_low > 0:
                pattern_stop = base_low - 0.02
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            else:
                recent_low = min(b.low for b in bars[-5:]) if len(bars) >= 5 else price * 0.97
                pattern_stop = recent_low - 0.02
        elif pattern == "abc_continuation":
            b_low = float(scan_result.criteria.get("b_low", 0))
            if b_low <= 0:
                self._reject("no ABC B-low")
                return None
            pattern_stop = b_low - 0.02
        elif pattern in (
            "first_pullback_reclaim",
            "runner_reclaim_continuation",
            "early_vwap_reclaim_scout",
        ):
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            base_low = float(scan_result.criteria.get("base_low", 0))
            pullback_low = float(scan_result.criteria.get("pullback_low", 0))
            washout_low = float(scan_result.criteria.get("washout_low", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif base_low > 0:
                pattern_stop = base_low - 0.02
            elif pullback_low > 0:
                pattern_stop = pullback_low - 0.02
            elif washout_low > 0:
                pattern_stop = washout_low - 0.02
            else:
                self._reject("no first-pullback low")
                return None
        elif pattern == "level_breakout_reclaim":
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            base_low = float(scan_result.criteria.get("base_low", 0))
            breakout_level = float(scan_result.criteria.get("breakout_level", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif base_low > 0:
                pattern_stop = base_low - 0.02
            elif breakout_level > 0:
                pattern_stop = breakout_level - 0.05
            else:
                self._reject("no breakout level")
                return None
        elif pattern == "shallow_stair_continuation":
            stop_from_criteria = float(scan_result.criteria.get("stop_price", 0))
            base_low = float(scan_result.criteria.get("base_low", 0))
            if stop_from_criteria > 0:
                pattern_stop = stop_from_criteria
            elif base_low > 0:
                pattern_stop = max(base_low - 0.02, price * 0.94)
            else:
                self._reject("no stair-step base low")
                return None
        else:
            self._reject("unhandled pattern")
            return None

        # Use the technical stop as-is. Only sanity-check it.
        stop_price = pattern_stop

        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            self._reject("stop above entry")
            return None

        risk_pct = risk_per_share / price
        # For big intraday movers, allow slightly wider risk (up to 15%)
        # but if still too wide, tighten the stop to last 3 bars low
        max_risk_pct = 0.10
        if len(bars) >= 5:
            day_move = (bars[-1].close - bars[0].open) / bars[0].open if bars[0].open > 0 else 0
            if day_move > 0.30:
                max_risk_pct = 0.15

        if risk_pct > max_risk_pct:
            # Try tighter stop: last 3 bars low
            tight_stop = min(b.low for b in bars[-3:]) - 0.02
            tight_risk = price - tight_stop
            tight_risk_pct = tight_risk / price if price > 0 else 1.0
            if self._is_hot_hod_reclaim(pattern, scan_result, bars):
                hot_stop = self._hot_hod_reclaim_stop(price, bars)
                hot_risk = price - hot_stop
                hot_risk_pct = hot_risk / price if price > 0 else 1.0
                if 0.005 < hot_risk_pct <= 0.08:
                    stop_price = hot_stop
                    risk_per_share = hot_risk
                    risk_pct = hot_risk_pct
                    logger.info(
                        "ENTRY GUARD %s: hot HOD reclaim tactical stop from $%.2f to $%.2f (risk %.0f%% → %.0f%%)",
                        scan_result.symbol, pattern_stop, stop_price,
                        (price - pattern_stop) / price * 100, risk_pct * 100,
                    )
                else:
                    logger.info(
                        "ENTRY GUARD REJECT %s: hot HOD reclaim risk too wide: $%.2f (%.0f%% of $%.2f)",
                        scan_result.symbol, hot_risk, hot_risk_pct * 100, price,
                    )
                    self._reject(
                        "hot HOD reclaim risk too wide: ${:.2f} ({:.0f}% of ${:.2f})".format(
                            hot_risk, hot_risk_pct * 100, price)
                    )
                    return None
            elif 0.005 < tight_risk_pct <= max_risk_pct:
                stop_price = tight_stop
                risk_per_share = tight_risk
                risk_pct = tight_risk_pct
                logger.info("ENTRY GUARD %s: tightened stop from $%.2f to $%.2f (risk %.0f%% → %.0f%%)",
                            scan_result.symbol, pattern_stop, stop_price, 
                            (price - pattern_stop) / price * 100, risk_pct * 100)
            else:
                abc_scout_stop = (
                    self._abc_scout_tactical_stop(price, scan_result, bars)
                    if pattern == "abc_continuation" else None
                )
                if abc_scout_stop is not None:
                    stop_price = abc_scout_stop
                    risk_per_share = price - stop_price
                    risk_pct = risk_per_share / price
                    scan_result.criteria["entry_tier"] = "abc_scout"
                    scan_result.criteria["entry_tier_reason"] = (
                        "A+ ABC continuation; full B-low stop too wide, using reduced-size tactical reclaim stop"
                    )
                    scan_result.criteria["stop_price"] = round(stop_price, 4)
                    logger.info(
                        "ENTRY GUARD %s: A+ ABC scout tactical stop from $%.2f to $%.2f "
                        "(risk %.0f%% → %.0f%%)",
                        scan_result.symbol, pattern_stop, stop_price,
                        (price - pattern_stop) / price * 100, risk_pct * 100,
                    )
                else:
                    logger.info("ENTRY GUARD REJECT %s: risk too wide: $%.2f (%.0f%% of $%.2f) — skip loose setup",
                                scan_result.symbol, risk_per_share, risk_pct * 100, price)
                    self._reject(
                        "risk too wide: ${:.2f} ({:.0f}% of ${:.2f}) — skip loose setup".format(
                            risk_per_share, risk_pct * 100, price)
                    )
                    return None
        if risk_pct < 0.005:
            logger.info("ENTRY GUARD REJECT %s: risk too tight: $%.2f (%.1f%% of $%.2f) — will stop on noise",
                        scan_result.symbol, risk_per_share, risk_pct * 100, price)
            self._reject(
                "risk too tight: ${:.2f} ({:.1f}% of ${:.2f}) — will stop on noise".format(
                    risk_per_share, risk_pct * 100, price)
            )
            return None

        target_price = price + (risk_per_share * self._rr_ratio)

        # Position sizing based on max dollar risk
        quality_factor, quality_label = self._setup_quality_factor(pattern, scan_result, bars)
        quantity = int((self._max_dollar_risk * quality_factor) / risk_per_share)
        quantity = max(1, min(quantity, 2000))

        scan_result.criteria["setup_quality"] = quality_label
        scan_result.criteria["size_factor"] = round(quality_factor, 2)

        reason = "{} {} ${:.2f}, stop=${:.2f} (risk=${:.2f}), target=${:.2f} (1:1, {})".format(
            pattern.replace("_", " ").title(),
            scan_result.symbol,
            price,
            stop_price,
            risk_per_share,
            target_price,
            quality_label,
        )

        return TradeSignal(
            symbol=scan_result.symbol,
            action=SignalAction.ENTER_LONG,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_price,
            take_profit=target_price,
            trailing_stop_offset=self._trail,
            max_hold_seconds=self._max_hold,
            reason=reason,
            scan_result=scan_result,
        )
