"""Keep historical experiments, logger settings and ML writes out of PAPER."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_isolated_historical(symbol, session_date, *, flags=None, start_time=None):
    payload = dict(symbol=symbol, session_date=session_date, flags=flags or {}, start_time=start_time)
    return _run_isolated(payload)


def run_isolated_sweep(symbols, dates, *, experiments=None):
    return _run_isolated(dict(symbols=symbols, dates=dates, experiments=experiments))


def _run_isolated(payload):
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = source
    env['DAYTRADING_CONFIG_LOADED'] = 'true'
    cache = Path(env.get('DAYTRADING_BACKTEST_CACHE_DIR', 'data/backtest_cache')).resolve()
    env['DAYTRADING_BACKTEST_CACHE_DIR'] = str(cache)
    with tempfile.TemporaryDirectory(prefix='daytradeboat-research-') as temp:
        env['DAYTRADING_DATA_DIR'] = str(Path(temp)/'data')
        env['DAYTRADING_JOURNAL_DIR'] = str(Path(temp)/'journal')
        process = subprocess.run([sys.executable, '-m', __name__], input=json.dumps(payload),
            text=True, capture_output=True, cwd=temp, env=env, timeout=300)
        if process.returncode:
            raise ValueError('Historical worker failed; paper process and state are unaffected.')
        return json.loads(process.stdout)


def main():
    import logging
    logging.disable(logging.CRITICAL)
    from daytrading.config import Settings
    from daytrading.backtest.service import run_backtest, run_backtest_sweep
    request = json.load(sys.stdin)
    if 'symbols' in request:
        result = run_backtest_sweep(request['symbols'], request['dates'],
                                   experiments=request.get('experiments'), settings=Settings())
    else:
        result = run_backtest(request['symbol'], request['session_date'], flags=request.get('flags'),
                              start_time=request.get('start_time'), settings=Settings())
    json.dump(result, sys.stdout, default=str)


if __name__ == '__main__':
    main()
