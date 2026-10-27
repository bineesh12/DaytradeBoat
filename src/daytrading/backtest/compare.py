"""Offline matched-data comparison of broad vs diagnostic entry selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

BASELINE = ('vwap_pullback', 'first_pullback_reclaim', 'hod_reclaim')

def compare(path: Path, *, commission_per_share=0.005, float_metadata=None) -> dict:
    # Diagnostic comparison does not train or promote models or contact a broker.
    os.environ['DAYTRADING_ENABLE_ENTRY_ML'] = 'false'
    from daytrading.backtest.broker import FillModel
    from daytrading.backtest.data_loader import load_bars_csv
    from daytrading.backtest.driver import PipelineBacktestDriver
    from daytrading.pipeline.engine import LIVE_A_PLUS_SCANNERS
    from daytrading.pipeline.factory import create_scalping_pipeline
    from daytrading.strategy import entry_guard
    entry_guard.ENTRY_ML_ENABLED = False
    bars = load_bars_csv(str(path))
    floats = float_metadata or {}
    class RecordedFloats:
        def get_float(self, symbol):
            return floats.get(symbol, {}).get('float_shares')
        def get_avg_volume(self, symbol):
            return floats.get(symbol, {}).get('avg_volume')
    output = {
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'symbols': sorted(bars), 'bars': sum(map(len, bars.values())),
        'commission_per_share': commission_per_share,
        'float_metadata_sha256': hashlib.sha256(json.dumps(floats, sort_keys=True).encode()).hexdigest(),
        'limitations': ['Revised engine for both variants; not a replay of an archived old binary.',
                        'Fixed supplied candidate universe; market-wide discovery/HOD admission is not replayed.',
                        'Minute-bar quote and fill estimates; no real broker, stop order or market-impact validation.',
                        'Automatic re-entry and runner re-adds disabled for both variants.',
                        'Float metadata supplied at collection time, not a historical security master.'],
        'variants': {},
    }
    for name, patterns in (('broad', tuple(sorted(LIVE_A_PLUS_SCANNERS))), ('three_patterns', BASELINE)):
        driver = PipelineBacktestDriver(bars, fill_model=FillModel(commission_per_share=commission_per_share), max_bars_per_symbol=1000)
        driver._pipeline = create_scalping_pipeline(portfolio=driver._portfolio, broker=driver._broker,
                                                   min_price=1, max_price=20, entry_patterns=patterns,
                                                   float_checker=RecordedFloats())
        driver._pipeline._max_entry_risk_pct = .06
        result = driver.run()
        output['variants'][name] = {'patterns': list(patterns), 'scorecard': result.scorecard,
                                    'cycles': result.cycles, 'scan_hits': result.scan_hits,
                                    'signals': result.signals, 'fills': len(result.fills),
                                    'rejected_by_layer': result.rejected_by_layer,
                                    'trades': result.trades,
                                    'remaining_positions': {sym: pos.quantity for sym, pos in result.final_portfolio.positions.items() if not pos.is_flat}}
    return output

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--float-metadata', type=Path)
    parser.add_argument('--commission-per-share', type=float, default=.005)
    args = parser.parse_args(argv)
    if args.commission_per_share < 0:
        parser.error('commission must be nonnegative')
    floats = json.loads(args.float_metadata.read_text()) if args.float_metadata else None
    result = compare(args.bars, commission_per_share=args.commission_per_share, float_metadata=floats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False))
    print(json.dumps({name: {'fills': data['fills'], 'signals': data['signals']} for name,data in result['variants'].items()}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
