"""Explain historical gate failures without inventing execution evidence.

Micro analysis uses the native tick aggregator and scanner, never fixed-grid
OHLC candles as substitute ticks. No broker, quotes or fill outcomes are used.
"""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import os
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

from daytrading.models import Side, Tick

NY = ZoneInfo('America/New_York')


def strategy_breakdown(scan_events, decisions):
    names = sorted({r.get('scanner') or r.get('pattern') for r in scan_events if r.get('scanner') or r.get('pattern')})
    rows = []
    for name in names:
        hits = [r for r in scan_events if (r.get('scanner') or r.get('pattern')) == name]
        rejects = [r for r in decisions if (r.get('scanner') or r.get('pattern')) == name and r.get('passed') is False]
        counts = Counter(r.get('blocked_layer') or r.get('stage') or 'unknown' for r in rejects)
        reasons = Counter(r.get('reason') or 'Unspecified rejection' for r in rejects)
        rows.append(dict(pattern=name, hits=len(hits), rejected_decisions=len(rejects),
                         blocked_by_stage=dict(counts),
                         top_reasons=[dict(reason=r, count=n) for r, n in reasons.most_common(3)],
                         first_hit=hits[0].get('ts'), last_hit=hits[-1].get('ts')))
    return rows


def micro_diagnostics(symbol, day, minute_bars, *, cache_dir=None, ticks=None):
    """Detector-only research: shared rules, actual exchange tick boundaries.

    Original receipt order/delays and quotes are unavailable in this historical
    cache. A detector pass never becomes an order or a profitable backtest here.
    Missing/incomplete raw inputs block this lane; no candle fallback exists.
    """
    base = dict(pattern='micro_pullback_reclaim', execution_status='not_tested',
                signal_status='not_verified', orders_simulated=0, validated=False,
                evidence='historical exchange-ordered trades; original receipt order unavailable',
                missing_execution_inputs=['historical quotes and spread', 'original processing order and portfolio state'],
                windows=0, candidates=0, first_block_counts={}, examples=[], candidate_events=[])
    if ticks is None:
        path = Path(cache_dir or os.environ.get('DAYTRADING_BACKTEST_CACHE_DIR', 'data/backtest_cache')) / f'{symbol}_{day}_event_trades.json'
        if not path.is_file():
            return dict(base, status='missing_trade_recording', reason='Raw historical trades required for the native micro detector. Fixed-grid 10-second candles cannot reproduce its event-anchored boundaries.')
        try:
            raw = path.read_bytes(); payload = json.loads(raw)
            if (payload.get('schema') != 'exchange-trades-v1' or payload.get('complete') is not True
                    or payload.get('symbol') != symbol or payload.get('date') != day
                    or payload.get('feed') != 'sip'):
                raise ValueError('complete matching SIP trade cache required')
            ticks = [Tick(symbol=symbol, ts=datetime.fromisoformat(r['ts']), price=float(r['price']),
                          size=float(r['size']), side=Side.BUY) for r in payload['trades']]
            base['input_sha256'] = hashlib.sha256(raw).hexdigest()
        except (KeyError, TypeError, ValueError):
            return dict(base, status='invalid_trade_recording', reason='Trade cache is incomplete, incompatible or invalid; no micro result produced.')
    if any(t.symbol != symbol or t.ts.tzinfo is None or t.ts.astimezone(NY).date().isoformat() != day
           or not all(isfinite(v) and v > 0 for v in (t.price, t.size)) for t in ticks):
        return dict(base, status='invalid_trade_recording', reason='Mixed-symbol/day, invalid trade or naive timestamp; micro analysis blocked.')
    from daytrading.data.bar_aggregator import BarAggregator
    from daytrading.strategy.scalping.micro_pullback_runtime import MicroPullbackScanner
    aggregator = BarAggregator()
    scanner = MicroPullbackScanner(); scanner._bar_aggregator = aggregator
    minutes = sorted((b for b in minute_bars if b.symbol == symbol), key=lambda b:b.ts)
    events = []; examples = {}; candidates = []
    for tick in sorted(ticks, key=lambda t:t.ts):
        completed = aggregator.on_tick(tick)
        if completed is None:
            continue
        context = [b for b in minutes if b.ts + timedelta(minutes=1) <= tick.ts
                   and b.ts.astimezone(NY).date().isoformat() == day]
        hits = scanner.scan({symbol:context}, now=tick.ts)
        reason = scanner.last_rejection.get(symbol)
        event = dict(time=tick.ts.isoformat(), trigger=completed.ts.isoformat(),
                     price=completed.close, paper_entry_window='04:00' <= tick.ts.astimezone(NY).strftime('%H:%M') < '15:30')
        window = aggregator.get_completed_10s(symbol, count=6)
        event['completion_durations_seconds'] = [(x.closed_at-x.bar.ts).total_seconds() for x in window]
        event['window_start'] = window[0].bar.ts.isoformat()
        if hits:
            hit = hits[0]
            event.update(status='detector_pass', stop=hit.criteria['stop_price'],
                         reason='Micro structure detected; quote, quality, portfolio and execution checks not run.')
            candidates.append(event)
        elif reason:
            event.update(status='rejected', rule=reason['rule'], reason=reason['reason'])
            examples.setdefault(reason['rule'], event)
        events.append(event)
    base.update(status='detector_only', raw_trades=len(ticks), windows=len(events),
                candidates=len(candidates), first_block_counts=dict(scanner.rejections),
                examples=list(examples.values()), candidate_events=candidates,
                events=events, reason='Native micro detector evaluated. Detection is not entry verification or fill validation.')
    return base
