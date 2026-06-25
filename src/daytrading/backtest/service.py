from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from daytrading.backtest.data_loader import (
    fetch_alpaca_10s_bars_for_day,
    fetch_alpaca_bars_for_day,
)
from daytrading.backtest.broker import BacktestBroker
from daytrading.backtest.driver import PipelineBacktestDriver
from daytrading.backtest.report import build_backtest_scorecard
from daytrading.config import Settings
from daytrading.models import Bar, PortfolioState
from daytrading.pipeline.factory import create_scalping_pipeline


SUPPORTED_FLAGS = {
    "fresh_vwap_reclaim_scout",
    "vwap_reclaim_scout",
    "level_breakout_scout",
    "elite_wide_spread",
    "momentum_burst_live",
    "level_capped_entry",
    "execution_timer_10s",
    "ten_second_breakout_scout",
    "level_reclaim_10s_scout",
    "breakout_scalp_replay",
    "momentum_burst_replay",
    "momentum_burst_hit_run",
    "warrior_squeeze_playbook",
    "warrior_ignition_model",
    "live_like_10s",
}


_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_symbol(value: str) -> str:
    """Validate and canonicalize a ticker before it touches the filesystem.

    The symbol is interpolated into cache file paths, so reject anything that
    is not a plain ticker (blocks path traversal like ``../../etc``).
    """
    sym = str(value or "").upper().strip()
    if not sym:
        raise ValueError("symbol is required")
    if not _SYMBOL_RE.match(sym):
        raise ValueError("invalid symbol: {!r}".format(value))
    return sym


def normalize_session_date(value: str | date) -> date:
    """Accept ISO dates plus common dashboard shorthand like DD/MM."""
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("date is required")
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass

    normalized = text.replace(".", "/").replace("-", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError("invalid date format; use YYYY-MM-DD or DD/MM")

    if len(parts) == 2:
        day, month = int(parts[0]), int(parts[1])
        year = date.today().year
    else:
        if len(parts[0]) == 4:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError("invalid date format; use YYYY-MM-DD or DD/MM") from exc


def normalize_start_time(value: Optional[str], session_date: date) -> Optional[datetime]:
    """Parse optional dashboard replay start.

    A plain ``HH:MM`` is treated as US/Eastern on the selected session date,
    matching the chart language traders use. Full ISO datetimes are accepted
    too and converted to UTC when timezone-aware.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("UTC"))
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        raise ValueError("invalid start time; use HH:MM ET or ISO datetime")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    eastern = ZoneInfo("America/New_York")
    try:
        local_dt = datetime.combine(session_date, time(hour, minute, second), tzinfo=eastern)
    except ValueError as exc:
        raise ValueError("invalid start time; use HH:MM ET or ISO datetime") from exc
    return local_dt.astimezone(ZoneInfo("UTC"))


DEFAULT_EXPERIMENTS: Dict[str, Dict[str, bool]] = {
    "baseline": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "fresh_vwap_reclaim_scout": {
        "fresh_vwap_reclaim_scout": True,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "vwap_reclaim_scout": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": True,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "level_breakout_scout": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": True,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "elite_wide_spread": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": True,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "momentum_burst_live": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": True,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "level_capped_entry": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": True,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "ten_second_breakout_scout": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": True,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "level_reclaim_10s_scout": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "level_reclaim_10s_scout": True,
        "breakout_scalp_replay": False,
        "momentum_burst_hit_run": False,
    },
    "breakout_scalp_replay": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "level_reclaim_10s_scout": False,
        "breakout_scalp_replay": True,
        "momentum_burst_hit_run": False,
        "live_like_10s": True,
    },
    "momentum_burst_replay": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "level_reclaim_10s_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_replay": True,
        "momentum_burst_hit_run": False,
        "live_like_10s": True,
    },
    "momentum_burst_hit_run": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "level_reclaim_10s_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_replay": False,
        "momentum_burst_hit_run": True,
        "live_like_10s": True,
    },
    "warrior_squeeze_playbook": {
        "fresh_vwap_reclaim_scout": False,
        "vwap_reclaim_scout": False,
        "level_breakout_scout": False,
        "elite_wide_spread": False,
        "momentum_burst_live": False,
        "level_capped_entry": False,
        "execution_timer_10s": True,
        "ten_second_breakout_scout": False,
        "level_reclaim_10s_scout": False,
        "breakout_scalp_replay": False,
        "momentum_burst_replay": False,
        "momentum_burst_hit_run": False,
        "warrior_squeeze_playbook": True,
        "live_like_10s": True,
    },
}


@dataclass(frozen=True)
class BacktestRequest:
    symbol: str
    date: str
    flags: Dict[str, bool]


def normalize_flags(flags: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    raw = flags or {}
    return {
        "fresh_vwap_reclaim_scout": bool(raw.get("fresh_vwap_reclaim_scout", False)),
        "vwap_reclaim_scout": bool(raw.get("vwap_reclaim_scout", False)),
        "level_breakout_scout": bool(raw.get("level_breakout_scout", False)),
        "elite_wide_spread": bool(raw.get("elite_wide_spread", False)),
        "momentum_burst_live": bool(raw.get("momentum_burst_live", False)),
        "level_capped_entry": bool(raw.get("level_capped_entry", False)),
        "execution_timer_10s": bool(raw.get("execution_timer_10s", True)),
        "ten_second_breakout_scout": bool(raw.get("ten_second_breakout_scout", False)),
        "level_reclaim_10s_scout": bool(raw.get("level_reclaim_10s_scout", False)),
        "breakout_scalp_replay": bool(raw.get("breakout_scalp_replay", False)),
        "momentum_burst_replay": bool(raw.get("momentum_burst_replay", False)),
        "momentum_burst_hit_run": bool(raw.get("momentum_burst_hit_run", False)),
        "warrior_squeeze_playbook": bool(raw.get("warrior_squeeze_playbook", False)),
        "warrior_ignition_model": bool(raw.get("warrior_ignition_model", False)),
        "live_like_10s": bool(raw.get("live_like_10s", False)),
    }


def normalize_experiments(
    experiments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, bool]]:
    raw = experiments or DEFAULT_EXPERIMENTS
    normalized: Dict[str, Dict[str, bool]] = {}
    for name, flags in raw.items():
        label = str(name or "").strip() or "experiment"
        normalized[label] = normalize_flags(flags)
    if "baseline" not in normalized:
        normalized = {"baseline": normalize_flags({}), **normalized}
    return normalized


@contextmanager
def _elite_wide_spread_flag(enabled: bool):
    import daytrading.strategy.entry_guard as entry_guard

    previous = entry_guard.ELITE_WIDE_SPREAD_ENABLED
    entry_guard.ELITE_WIDE_SPREAD_ENABLED = bool(enabled)
    try:
        yield
    finally:
        entry_guard.ELITE_WIDE_SPREAD_ENABLED = previous


@contextmanager
def _momentum_burst_live_flag(enabled: bool):
    import daytrading.pipeline.engine as engine

    previous_live = engine.LIVE_A_PLUS_SCANNERS
    previous_watch = engine.WATCH_ONLY_SCANNERS
    if enabled:
        engine.LIVE_A_PLUS_SCANNERS = frozenset({
            *previous_live,
            "momentum_burst",
        })
        engine.WATCH_ONLY_SCANNERS = frozenset(
            scanner for scanner in previous_watch
            if scanner != "momentum_burst"
        )
    try:
        yield
    finally:
        engine.LIVE_A_PLUS_SCANNERS = previous_live
        engine.WATCH_ONLY_SCANNERS = previous_watch


@contextmanager
def _quiet_backtest_logs():
    """Keep dashboard backtests from flooding live Cloud Logging.

    Live-like 10s replays can run hundreds of miniature pipeline cycles. At the
    normal live INFO level each scanner logs every cycle, which makes remote
    dashboard backtests slow and can leave the UI looking stuck. Preserve
    warnings/errors, but mute routine scanner/router chatter while replaying.
    """
    names = (
        "daytrading.pipeline.engine",
        "daytrading.classifier.router",
        "daytrading.scanner.scalping",
        "daytrading.strategy.scalping.momentum_pattern",
    )
    loggers = [logging.getLogger(name) for name in names]
    previous = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.WARNING)
        yield
    finally:
        for logger, level in zip(loggers, previous):
            logger.setLevel(level)


def _round_trips(trades: List[dict]) -> List[dict]:
    entries: Dict[str, dict] = {}
    trips: List[dict] = []
    for row in trades:
        sym = str(row.get("symbol") or "")
        if row.get("trade_type") in ("entry", "reentry"):
            entries[sym] = row
            continue
        if row.get("trade_type") != "exit":
            continue
        entry = entries.get(sym, {})
        trips.append({
            "symbol": sym,
            "entry_time": entry.get("entry_time") or row.get("entry_time"),
            "exit_time": row.get("exit_time"),
            "entry_price": entry.get("entry_price") or row.get("entry_price"),
            "exit_price": row.get("exit_price"),
            "quantity": row.get("quantity"),
            "pnl": row.get("pnl"),
            "pattern": entry.get("strategy") or row.get("strategy") or "",
            "mode": entry.get("strategy") or row.get("strategy") or "standard",
            "exit_reason": row.get("exit_reason") or "",
        })
    return trips


def _bars_payload(bars: List[Bar]) -> List[dict]:
    return [
        {
            "ts": bar.ts.isoformat(),
            "open": round(float(bar.open), 4),
            "high": round(float(bar.high), 4),
            "low": round(float(bar.low), 4),
            "close": round(float(bar.close), 4),
            "volume": float(bar.volume),
        }
        for bar in bars
    ]


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        ).strip()
    except Exception:
        return ""


def _cache_file_info(symbol: str, day: date, suffix: str) -> dict:
    cache_root = Path(os.environ.get("DAYTRADING_BACKTEST_CACHE_DIR", "data/backtest_cache"))
    path = cache_root / f"{symbol}_{day.isoformat()}_{suffix}.json"
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        data = path.read_bytes()
    except OSError:
        return {"path": str(path), "exists": True, "readable": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": len(data),
        "sha1": hashlib.sha1(data).hexdigest()[:12],
    }


def _strategy_manifest(settings: Optional[Settings]) -> dict:
    if settings is None:
        return {}
    strategy = settings.strategy
    fields = (
        "momentum_burst_hit_run_enabled",
        "momentum_burst_hit_run_end_et",
        "momentum_burst_hit_run_max_entries",
        "momentum_burst_hit_run_reward_risk",
        "momentum_burst_hit_run_stop_after_giveback",
        "momentum_burst_hit_run_max_giveback",
        "momentum_burst_hit_run_daily_loss_stop",
        "warrior_squeeze_enabled",
        "warrior_squeeze_min_reclaim_price",
        "warrior_squeeze_starter_size_factor",
        "warrior_squeeze_position_value",
        "warrior_squeeze_max_dollar_risk",
        "momentum_burst_live_enabled",
        "momentum_burst_cycle_enabled",
        "runner_trail_pct",
        "runner_trail_adaptive",
        "runner_trail_atr_mult",
        "runner_trail_cap",
        "runner_give_room_after_partial",
        "entry_chase_pct_low",
        "entry_chase_pct_high",
        "missed_a_plus_chase_pct_sub5",
        "missed_a_plus_chase_pct_5plus",
        "late_pullback_max_hod_pct",
        "late_pullback_max_hod_other_pct",
    )
    return {field: getattr(strategy, field, None) for field in fields}


def _backtest_manifest(
    *,
    symbol: str,
    day: date,
    flags: Dict[str, Any],
    settings: Optional[Settings],
    bars_by_symbol: Optional[Dict[str, List[Bar]]],
    timer_bars: Optional[Dict[str, List[Bar]]],
    symbol_bars: Sequence[Bar],
) -> dict:
    timer_count = len((timer_bars or {}).get(symbol, []) or [])
    effective_settings = settings or Settings()
    return {
        "symbol": symbol,
        "date": day.isoformat(),
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "code_version": _git_version(),
        "flags": dict(flags),
        "settings": {
            "initial_cash": effective_settings.initial_cash,
            "commission_per_share": effective_settings.commission_per_share,
            "alpaca_feed": getattr(effective_settings, "alpaca_feed", ""),
            "strategy": _strategy_manifest(effective_settings),
        },
        "data": {
            "source": "in_memory" if bars_by_symbol is not None else "alpaca_cache",
            "bars_1m": len(symbol_bars),
            "bars_10s": timer_count,
            "cache_1m": _cache_file_info(symbol, day, "1m") if bars_by_symbol is None else {},
            "cache_10s": (
                _cache_file_info(symbol, day, "10s_trades")
                if bars_by_symbol is None and (flags.get("execution_timer_10s") or flags.get("live_like_10s"))
                else {}
            ),
        },
    }


def run_backtest(
    symbol: str,
    session_date: str,
    *,
    flags: Optional[Dict[str, Any]] = None,
    start_time: Optional[str] = None,
    bars_by_symbol: Optional[Dict[str, List[Bar]]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    sym = normalize_symbol(symbol)
    day = normalize_session_date(session_date)
    start_dt = normalize_start_time(start_time, day)
    active_flags = normalize_flags(flags)

    bars = bars_by_symbol or fetch_alpaca_bars_for_day(sym, day, settings=settings)
    symbol_bars = list(bars.get(sym, []))
    timer_bars = None
    if not symbol_bars:
        manifest = _backtest_manifest(
            symbol=sym,
            day=day,
            flags=active_flags,
            settings=settings,
            bars_by_symbol=bars_by_symbol,
            timer_bars=None,
            symbol_bars=[],
        )
        return {
            "ok": True,
            "symbol": sym,
            "date": day.isoformat(),
            "bars": 0,
            "bars_data": [],
            "cycles": 0,
            "trades": [],
            "round_trips": [],
            "scan_events": [],
            "entry_decisions": [],
            "rejection_details": [],
            "deferred_signals": [],
            "micro_opportunities": [],
            "scorecard": {},
            "funnel": {},
            "flags": active_flags,
            "manifest": manifest,
            "message": "No bars found for symbol/date",
        }

    with _quiet_backtest_logs():
        with _elite_wide_spread_flag(active_flags["elite_wide_spread"]), _momentum_burst_live_flag(active_flags["momentum_burst_live"]):
            initial_cash = settings.initial_cash if settings else 25_000.0
            broker = BacktestBroker()
            portfolio = PortfolioState(cash=initial_cash)
            timer_bars = (
                fetch_alpaca_10s_bars_for_day(sym, day, settings=settings)
                if (active_flags["execution_timer_10s"] or active_flags["live_like_10s"])
                and bars_by_symbol is None
                else None
            )
            pipeline = create_scalping_pipeline(
                initial_cash=initial_cash,
                commission_per_share=(settings.commission_per_share if settings else 0.0),
                broker=broker,
                portfolio=portfolio,
                fresh_vwap_reclaim_scout_enabled=active_flags["fresh_vwap_reclaim_scout"],
                vwap_reclaim_scout_enabled=active_flags["vwap_reclaim_scout"],
                level_breakout_scout_enabled=active_flags["level_breakout_scout"],
                momentum_burst_live_enabled=active_flags["momentum_burst_live"],
                level_capped_entry_enabled=active_flags["level_capped_entry"],
                late_pullback_max_hod_pct=(
                    settings.strategy.late_pullback_max_hod_pct if settings else 12.0
                ),
                late_pullback_max_hod_other_pct=(
                    settings.strategy.late_pullback_max_hod_other_pct if settings else 10.0
                ),
                runner_trail_pct=(
                    settings.strategy.runner_trail_pct if settings else 0.03
                ),
                runner_min_confirm_pct=(
                    settings.strategy.runner_min_confirm_pct if settings else 0.018
                ),
                runner_trail_adaptive=(
                    settings.strategy.runner_trail_adaptive if settings else False
                ),
                runner_trail_atr_mult=(
                    settings.strategy.runner_trail_atr_mult if settings else 2.5
                ),
                runner_trail_cap=(
                    settings.strategy.runner_trail_cap if settings else 0.10
                ),
                runner_give_room_after_partial=(
                    settings.strategy.runner_give_room_after_partial if settings else False
                ),
            )
            # The chase guards are configured via methods (not factory kwargs), so
            # the backtest must apply settings here or it silently runs the engine
            # defaults and ignores DAYTRADING_ENTRY_CHASE_*/MISSED_A_PLUS_CHASE_*.
            if settings is not None:
                pipeline.configure_missed_a_plus_chase_guard(
                    window_sec=settings.strategy.missed_a_plus_chase_window_sec,
                    pct_sub5=settings.strategy.missed_a_plus_chase_pct_sub5,
                    pct_5plus=settings.strategy.missed_a_plus_chase_pct_5plus,
                    fresh_base_reset=settings.strategy.missed_a_plus_fresh_base_reset,
                    fresh_base_pct=settings.strategy.missed_a_plus_fresh_base_pct,
                )
                pipeline.configure_entry_chase_guard(
                    pct_low=settings.strategy.entry_chase_pct_low,
                    pct_high=settings.strategy.entry_chase_pct_high,
                    price_tier=settings.strategy.entry_chase_price_tier,
                )
                pipeline._max_entry_risk_pct = float(settings.strategy.max_entry_risk_pct)
            result = PipelineBacktestDriver(
                {sym: symbol_bars},
                pipeline=pipeline,
                portfolio=portfolio,
                initial_cash=initial_cash,
                use_execution_timer=active_flags["execution_timer_10s"],
                timer_bars_by_symbol=timer_bars,
                use_micro_breakout_scout=active_flags["ten_second_breakout_scout"],
                use_level_reclaim_10s_scout=active_flags["level_reclaim_10s_scout"],
                use_breakout_scalp_replay=active_flags["breakout_scalp_replay"],
                use_momentum_burst_replay=active_flags["momentum_burst_replay"],
                use_momentum_burst_hit_run=active_flags["momentum_burst_hit_run"],
                use_warrior_squeeze_playbook=active_flags["warrior_squeeze_playbook"],
                use_warrior_ignition_model=active_flags["warrior_ignition_model"],
                momentum_burst_window_sec=(
                    settings.strategy.momentum_burst_window_sec if settings else 300.0
                ),
                momentum_burst_cooldown_sec=(
                    settings.strategy.momentum_burst_scalp_cooldown_sec if settings else 300.0
                ),
                momentum_burst_hit_run_max_entries=(
                    settings.strategy.momentum_burst_hit_run_max_entries if settings else 1
                ),
                momentum_burst_hit_run_win_cooldown_sec=(
                    settings.strategy.momentum_burst_hit_run_win_cooldown_sec if settings else 15.0
                ),
                momentum_burst_hit_run_loss_cooldown_sec=(
                    settings.strategy.momentum_burst_hit_run_loss_cooldown_sec if settings else 90.0
                ),
                momentum_burst_hit_run_max_hold_sec=(
                    settings.strategy.momentum_burst_hit_run_max_hold_sec if settings else 45.0
                ),
                momentum_burst_hit_run_reward_risk=(
                    settings.strategy.momentum_burst_hit_run_reward_risk if settings else 1.0
                ),
                momentum_burst_hit_run_stop_after_giveback=(
                    settings.strategy.momentum_burst_hit_run_stop_after_giveback if settings else True
                ),
                momentum_burst_hit_run_max_giveback=(
                    settings.strategy.momentum_burst_hit_run_max_giveback if settings else 50.0
                ),
                momentum_burst_hit_run_daily_loss_stop=(
                    settings.strategy.momentum_burst_hit_run_daily_loss_stop if settings else 50.0
                ),
                momentum_burst_hit_run_end_et=(
                    settings.strategy.momentum_burst_hit_run_end_et if settings else "11:30"
                ),
                warrior_squeeze_min_reclaim_price=(
                    settings.strategy.warrior_squeeze_min_reclaim_price if settings else 3.5
                ),
                warrior_squeeze_starter_size_factor=(
                    settings.strategy.warrior_squeeze_starter_size_factor if settings else 0.35
                ),
                warrior_squeeze_position_value=(
                    settings.strategy.warrior_squeeze_position_value if settings else 2000.0
                ),
                warrior_squeeze_max_dollar_risk=(
                    settings.strategy.warrior_squeeze_max_dollar_risk if settings else 150.0
                ),
                warrior_squeeze_max_entries=(
                    settings.strategy.warrior_squeeze_max_entries if settings else 3
                ),
                warrior_max_concurrent_trades=(
                    settings.strategy.warrior_max_concurrent_trades if settings else 1
                ),
                warrior_watch_capacity=(
                    settings.strategy.warrior_watch_capacity if settings else 10
                ),
                warrior_watch_until_premarket_end=(
                    settings.strategy.warrior_watch_until_premarket_end
                    if settings
                    else True
                ),
                warrior_squeeze_win_cooldown_sec=(
                    settings.strategy.warrior_squeeze_win_cooldown_sec if settings else 10.0
                ),
                warrior_squeeze_reward_risk=(
                    settings.strategy.warrior_squeeze_reward_risk if settings else 3.0
                ),
                warrior_squeeze_add_reward_risk=(
                    settings.strategy.warrior_squeeze_add_reward_risk if settings else 1.0
                ),
                live_like_10s=active_flags["live_like_10s"],
            ).run(start=start_dt)

    trips = _round_trips(result.trades)
    scorecard = dict(result.scorecard or {})
    manifest = _backtest_manifest(
        symbol=sym,
        day=day,
        flags=active_flags,
        settings=settings,
        bars_by_symbol=bars_by_symbol,
        timer_bars=timer_bars,
        symbol_bars=symbol_bars,
    )
    return {
        "ok": True,
        "symbol": sym,
        "date": day.isoformat(),
        "start_time": start_time or "",
        "start_time_utc": start_dt.isoformat() if start_dt is not None else "",
        "bars": len(symbol_bars),
        "bars_data": _bars_payload(symbol_bars),
        "cycles": result.cycles,
        "fills": len(result.fills),
        "trades": result.trades,
        "round_trips": trips,
        "scan_events": result.scan_events,
        "entry_decisions": result.entry_decisions,
        "rejection_details": result.rejection_details,
        "deferred_signals": result.deferred_signals,
        "micro_opportunities": result.micro_opportunities,
        "rejected_by_layer": result.rejected_by_layer,
        "top_reject_reasons_by_layer": result.rejected_reasons_by_layer,
        "scorecard": scorecard,
        "funnel": scorecard.get("funnel", {}),
        "missed_a_plus": result.missed_a_plus,
        "flags": active_flags,
        "manifest": manifest,
        "execution_timer_source": result.execution_timer_source,
        "unsupported_flags": ["momentum_breakout"],
        "final_cash": round(result.final_portfolio.cash, 2) if result.final_portfolio else None,
        "open_positions": {
            sym: {"quantity": pos.quantity, "avg_price": pos.avg_price}
            for sym, pos in (result.final_portfolio.positions if result.final_portfolio else {}).items()
        },
    }


def _as_symbols(symbols: Sequence[str]) -> List[str]:
    clean = []
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        if sym not in clean:
            clean.append(sym)
    if not clean:
        raise ValueError("at least one symbol is required")
    return clean


def _as_dates(session_dates: Sequence[str | date]) -> List[str]:
    clean = []
    for item in session_dates:
        day = normalize_session_date(item)
        text = day.isoformat()
        if text not in clean:
            clean.append(text)
    if not clean:
        raise ValueError("at least one date is required")
    return clean


def _aggregate_runs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    trades: List[dict] = []
    missed: List[dict] = []
    scan_hits = signals = rejected = deferred = cycles = 0
    bars = fills = 0
    rejected_by_layer: Dict[str, int] = {}
    reason_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        trades.extend(list(row.get("trades") or []))
        missed.extend(list(row.get("missed_a_plus") or []))
        funnel = row.get("funnel") or {}
        scan_hits += int(funnel.get("scan_hits") or 0)
        signals += int(funnel.get("signals") or 0)
        rejected += int(funnel.get("rejected") or 0)
        deferred += int(funnel.get("deferred") or 0)
        for layer, count in dict(funnel.get("rejected_by_layer") or {}).items():
            rejected_by_layer[str(layer)] = rejected_by_layer.get(str(layer), 0) + int(count or 0)
        for layer, reasons in dict(funnel.get("top_reject_reasons_by_layer") or {}).items():
            layer_counts = reason_counts.setdefault(str(layer), {})
            for reason_row in list(reasons or []):
                reason = str(reason_row.get("reason") or "")
                layer_counts[reason] = layer_counts.get(reason, 0) + int(reason_row.get("count") or 0)
        cycles += int(row.get("cycles") or 0)
        bars += int(row.get("bars") or 0)
        fills += int(row.get("fills") or 0)
    rejected_reasons_by_layer = {
        layer: [
            {"reason": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
        for layer, counts in reason_counts.items()
    }
    scorecard = build_backtest_scorecard(
        trades=trades,
        total_scan_hits=scan_hits,
        total_signals=signals,
        total_rejected=rejected,
        total_deferred=deferred,
        cycle_count=cycles,
        missed_a_plus=missed,
        rejected_by_layer=rejected_by_layer,
        rejected_reasons_by_layer=rejected_reasons_by_layer,
    )
    return {
        "runs": len(rows),
        "bars": bars,
        "cycles": cycles,
        "fills": fills,
        "scorecard": scorecard,
        "funnel": scorecard.get("funnel", {}),
    }


def _score_delta(scorecard: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_pnl": round(float(scorecard.get("total_pnl") or 0.0) - float(baseline.get("total_pnl") or 0.0), 2),
        "expectancy_per_trade": round(
            float(scorecard.get("expectancy_per_trade") or 0.0)
            - float(baseline.get("expectancy_per_trade") or 0.0),
            2,
        ),
        "closed_trades": int(scorecard.get("closed_trades") or 0) - int(baseline.get("closed_trades") or 0),
        "trades_taken": int(scorecard.get("trades_taken") or 0) - int(baseline.get("trades_taken") or 0),
        "win_rate": round(float(scorecard.get("win_rate") or 0.0) - float(baseline.get("win_rate") or 0.0), 1),
        "profit_factor": round(float(scorecard.get("profit_factor") or 0.0) - float(baseline.get("profit_factor") or 0.0), 2),
    }


def run_backtest_sweep(
    symbols: Sequence[str],
    session_dates: Sequence[str | date],
    *,
    experiments: Optional[Dict[str, Dict[str, Any]]] = None,
    bars_by_symbol_date: Optional[Dict[Tuple[str, str], Dict[str, List[Bar]]]] = None,
    bar_fetcher: Optional[Callable[[str, date], Dict[str, List[Bar]]]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run baseline plus one-flag experiments across a symbol/date basket.

    The basket is explicit on purpose: until we have a historical top-mover
    source, this avoids pretending hand-picked symbols are unbiased.
    """
    syms = _as_symbols(symbols)
    days = _as_dates(session_dates)
    active_experiments = normalize_experiments(experiments)
    bars_cache: Dict[Tuple[str, str], Dict[str, List[Bar]]] = {}
    source = bars_by_symbol_date or {}

    def _bars_for(sym: str, day_text: str) -> Dict[str, List[Bar]]:
        key = (sym, day_text)
        if key not in bars_cache:
            if key in source:
                bars_cache[key] = source[key]
            elif bar_fetcher is not None:
                bars_cache[key] = bar_fetcher(sym, date.fromisoformat(day_text))
            else:
                bars_cache[key] = fetch_alpaca_bars_for_day(
                    sym,
                    date.fromisoformat(day_text),
                    settings=settings,
                )
        return bars_cache[key]

    experiment_results: Dict[str, Dict[str, Any]] = {}
    for name, flags in active_experiments.items():
        rows: List[Dict[str, Any]] = []
        for day_text in days:
            for sym in syms:
                rows.append(
                    run_backtest(
                        sym,
                        day_text,
                        flags=flags,
                        bars_by_symbol=_bars_for(sym, day_text),
                        settings=settings,
                    )
                )
        aggregate = _aggregate_runs(rows)
        aggregate["flags"] = flags
        aggregate["rows"] = [
            {
                "symbol": row.get("symbol"),
                "date": row.get("date"),
                "bars": row.get("bars"),
                "trades_taken": (row.get("scorecard") or {}).get("trades_taken", 0),
                "closed_trades": (row.get("scorecard") or {}).get("closed_trades", 0),
                "total_pnl": (row.get("scorecard") or {}).get("total_pnl", 0.0),
                "expectancy_per_trade": (row.get("scorecard") or {}).get("expectancy_per_trade", 0.0),
            }
            for row in rows
        ]
        experiment_results[name] = aggregate

    baseline_card = (experiment_results.get("baseline") or {}).get("scorecard") or {}
    deltas = {
        name: _score_delta(result.get("scorecard") or {}, baseline_card)
        for name, result in experiment_results.items()
        if name != "baseline"
    }
    return {
        "ok": True,
        "symbols": syms,
        "dates": days,
        "experiments": experiment_results,
        "deltas_vs_baseline": deltas,
        "unsupported_flags": ["momentum_breakout"],
        "universe_note": (
            "This sweep uses the supplied symbol/date basket. Use a historical "
            "top-mover source before treating results as unbiased."
        ),
    }
