"""Day trading platform. Public exports load only when requested."""
from importlib import import_module

_EXPORTS = {
    "BacktestEngine": "backtest", "BacktestResult": "backtest",
    "AdaptiveRouter": "classifier", "MarketRegimeClassifier": "classifier", "StyleConfig": "classifier",
    "DataFeed": "data", "InMemoryBarFeed": "data",
    "Broker": "execution", "PaperBroker": "execution",
    "ExitManager": "exits", "TrackedPosition": "exits",
    "PipelineResult": "pipeline", "TradingPipeline": "pipeline",
    "CompositeScanner": "scanner", "MomentumBurstScanner": "scanner", "PremarketGapScanner": "scanner",
    "Scanner": "scanner", "SpreadFilterScanner": "scanner", "TapeReaderScanner": "scanner",
    "VolumeSpikeScanner": "scanner", "VWAPDeviationScanner": "scanner",
    "GapReversalVerifier": "strategy", "MomentumScalpVerifier": "strategy", "Strategy": "strategy",
    "StrategyVerifier": "strategy", "TapeScalpVerifier": "strategy", "VolumeBreakoutVerifier": "strategy",
    "VWAPBounceVerifier": "strategy",
}
__all__ = [*_EXPORTS, "__version__"]
__version__ = "0.1.0"


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"daytrading.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
