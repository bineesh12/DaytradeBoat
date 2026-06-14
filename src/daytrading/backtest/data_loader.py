from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, List, Sequence

from daytrading.config import Settings
from daytrading.data.alpaca_feed import AlpacaHistoricalFeed
from daytrading.market_calendar import ET
from daytrading.models import Bar, Tick, Timeframe


def parse_timestamp(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("missing timestamp")
    ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def load_bars_csv(path: str, *, symbol: str | None = None) -> Dict[str, List[Bar]]:
    """Load OHLCV bars from CSV.

    Expected columns: symbol, ts, open, high, low, close, volume.
    If the CSV is single-symbol, pass ``symbol=`` and the symbol column may be
    omitted. Rows are sorted oldest-first per symbol.
    """
    grouped: Dict[str, List[Bar]] = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = str(row.get("symbol") or symbol or "").upper().strip()
            if not sym:
                raise ValueError("CSV row is missing symbol")
            grouped[sym].append(
                Bar(
                    symbol=sym,
                    ts=parse_timestamp(str(row.get("ts") or row.get("timestamp") or "")),
                    open=float(row.get("open") or row.get("o") or 0.0),
                    high=float(row.get("high") or row.get("h") or 0.0),
                    low=float(row.get("low") or row.get("l") or 0.0),
                    close=float(row.get("close") or row.get("c") or 0.0),
                    volume=float(row.get("volume") or row.get("v") or 0.0),
                    timeframe=Timeframe(row.get("timeframe") or "1m"),
                )
            )
    return {
        sym: sorted(bars, key=lambda b: b.ts)
        for sym, bars in grouped.items()
    }


def merge_bar_times(bars_by_symbol: Dict[str, Sequence[Bar]]) -> List[datetime]:
    seen = {bar.ts for bars in bars_by_symbol.values() for bar in bars}
    return sorted(seen)


def trim_universe_to_time(
    bars_by_symbol: Dict[str, Sequence[Bar]],
    now: datetime,
    *,
    max_bars: int = 120,
) -> Dict[str, List[Bar]]:
    universe: Dict[str, List[Bar]] = {}
    for symbol, bars in bars_by_symbol.items():
        visible = [bar for bar in bars if bar.ts <= now]
        if visible:
            universe[symbol] = visible[-max_bars:]
    return universe


def load_many_csv(paths: Iterable[str]) -> Dict[str, List[Bar]]:
    combined: Dict[str, List[Bar]] = defaultdict(list)
    for path in paths:
        for symbol, bars in load_bars_csv(path).items():
            combined[symbol].extend(bars)
    return {
        sym: sorted(bars, key=lambda b: b.ts)
        for sym, bars in combined.items()
    }


def _bar_to_dict(bar: Bar) -> dict:
    return {
        "symbol": bar.symbol,
        "ts": bar.ts.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "timeframe": bar.timeframe.value,
    }


def _bar_from_dict(item: dict) -> Bar:
    return Bar(
        symbol=str(item.get("symbol") or "").upper(),
        ts=parse_timestamp(str(item.get("ts") or item.get("timestamp") or "")),
        open=float(item.get("open") or 0.0),
        high=float(item.get("high") or 0.0),
        low=float(item.get("low") or 0.0),
        close=float(item.get("close") or 0.0),
        volume=float(item.get("volume") or 0.0),
        timeframe=Timeframe(item.get("timeframe") or "1m"),
    )


def _tick_to_dict(tick: Tick) -> dict:
    return {
        "symbol": tick.symbol,
        "ts": tick.ts.isoformat(),
        "price": tick.price,
        "size": tick.size,
    }


def _tick_from_dict(item: dict) -> Tick:
    from daytrading.models import Side

    return Tick(
        symbol=str(item.get("symbol") or "").upper(),
        ts=parse_timestamp(str(item.get("ts") or item.get("timestamp") or "")),
        price=float(item.get("price") or 0.0),
        size=float(item.get("size") or 0.0),
        side=Side.BUY,
    )


def aggregate_ticks_to_10s_bars(
    ticks_by_symbol: Dict[str, Sequence[Tick]],
) -> Dict[str, List[Bar]]:
    grouped: Dict[str, Dict[datetime, List[Tick]]] = defaultdict(lambda: defaultdict(list))
    for symbol, ticks in ticks_by_symbol.items():
        for tick in ticks:
            ts = tick.ts.astimezone(timezone.utc)
            bucket = ts.replace(microsecond=0)
            bucket = bucket - timedelta(seconds=bucket.second % 10)
            grouped[symbol.upper()][bucket].append(tick)

    result: Dict[str, List[Bar]] = {}
    for symbol, buckets in grouped.items():
        bars: List[Bar] = []
        for ts, ticks in sorted(buckets.items(), key=lambda item: item[0]):
            ordered = sorted(ticks, key=lambda t: t.ts)
            prices = [float(t.price) for t in ordered if t.price > 0]
            if not prices:
                continue
            bars.append(Bar(
                symbol=symbol,
                ts=ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(float(t.size or 0.0) for t in ordered),
                timeframe=Timeframe.SEC_10,
            ))
        if bars:
            result[symbol] = bars
    return result


def _date_window_utc(day: date) -> tuple[datetime, datetime]:
    start_et = datetime.combine(day, time(4, 0), tzinfo=ET)
    end_et = datetime.combine(day, time(20, 0), tzinfo=ET)
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)


def fetch_alpaca_bars_for_day(
    symbol: str,
    day: str | date,
    *,
    cache_dir: str | None = None,
    settings: Settings | None = None,
    feed: AlpacaHistoricalFeed | None = None,
) -> Dict[str, List[Bar]]:
    """Fetch and cache one symbol's extended-session 1-minute bars."""
    sym = str(symbol or "").upper().strip()
    if not sym:
        raise ValueError("symbol is required")
    session_day = date.fromisoformat(day) if isinstance(day, str) else day
    cache_root = cache_dir or os.environ.get(
        "DAYTRADING_BACKTEST_CACHE_DIR",
        os.path.join("data", "backtest_cache"),
    )
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"{sym}_{session_day.isoformat()}_1m.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        bars = [_bar_from_dict(item) for item in payload.get("bars", [])]
        return {sym: bars} if bars else {}

    cfg = settings or Settings()
    if feed is None:
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise ValueError("missing Alpaca credentials for backtest fetch")
        feed = AlpacaHistoricalFeed(
            cfg.alpaca_api_key,
            cfg.alpaca_secret_key,
            feed=cfg.alpaca_feed,
            bar_fetch_batch_size=1,
            bar_fetch_batch_delay_sec=0.0,
        )
    start, end = _date_window_utc(session_day)
    result = feed.get_bars([sym], timeframe="1Min", start=start, end=end, limit=2000)
    bars = sorted(result.get(sym, []), key=lambda b: b.ts)
    if bars:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({
                "symbol": sym,
                "date": session_day.isoformat(),
                "bars": [_bar_to_dict(bar) for bar in bars],
            }, fh)
    return {sym: bars} if bars else {}


def fetch_alpaca_10s_bars_for_day(
    symbol: str,
    day: str | date,
    *,
    cache_dir: str | None = None,
    settings: Settings | None = None,
    feed: AlpacaHistoricalFeed | None = None,
) -> Dict[str, List[Bar]]:
    """Fetch historical trades and aggregate them into 10-second bars."""
    sym = str(symbol or "").upper().strip()
    if not sym:
        raise ValueError("symbol is required")
    session_day = date.fromisoformat(day) if isinstance(day, str) else day
    cache_root = cache_dir or os.environ.get(
        "DAYTRADING_BACKTEST_CACHE_DIR",
        os.path.join("data", "backtest_cache"),
    )
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"{sym}_{session_day.isoformat()}_10s_trades.json")
    start, end = _date_window_utc(session_day)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        bars = [_bar_from_dict(item) for item in payload.get("bars", [])]
        if bars and bars[-1].ts >= end - timedelta(minutes=5):
            return {sym: bars}

    cfg = settings or Settings()
    if feed is None:
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise ValueError("missing Alpaca credentials for backtest trade fetch")
        feed = AlpacaHistoricalFeed(
            cfg.alpaca_api_key,
            cfg.alpaca_secret_key,
            feed=cfg.alpaca_feed,
            bar_fetch_batch_size=1,
            bar_fetch_batch_delay_sec=0.0,
        )
    def _fetch_ticks_window(window_start: datetime, window_end: datetime, depth: int = 0) -> List[Tick]:
        rows = feed.get_trades([sym], start=window_start, end=window_end, limit=50_000).get(sym, [])
        duration = (window_end - window_start).total_seconds()
        if len(rows) >= 49_000 and duration > 60 and depth < 8:
            midpoint = window_start + (window_end - window_start) / 2
            return (
                _fetch_ticks_window(window_start, midpoint, depth + 1)
                + _fetch_ticks_window(midpoint, window_end, depth + 1)
            )
        return rows

    ticks: Dict[str, List[Tick]] = {sym: []}
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(minutes=30), end)
        ticks[sym].extend(_fetch_ticks_window(chunk_start, chunk_end))
        chunk_start = chunk_end
    ticks[sym] = sorted(ticks.get(sym, []), key=lambda t: t.ts)
    bars = sorted(aggregate_ticks_to_10s_bars(ticks).get(sym, []), key=lambda b: b.ts)
    if bars:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({
                "symbol": sym,
                "date": session_day.isoformat(),
                "source": "historical_trades",
                "bars": [_bar_to_dict(bar) for bar in bars],
            }, fh)
    return {sym: bars} if bars else {}
