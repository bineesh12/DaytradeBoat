"""Day trading & scalping platform: classify, scan, verify, execute."""

from daytrading.backtest import BacktestEngine, BacktestResult
from daytrading.classifier import AdaptiveRouter, MarketRegimeClassifier, StyleConfig
from daytrading.data import DataFeed, InMemoryBarFeed
from daytrading.execution import Broker, PaperBroker
from daytrading.exits import ExitManager, TrackedPosition
from daytrading.pipeline import PipelineResult, TradingPipeline
from daytrading.scanner import (
    CompositeScanner,
    MomentumBurstScanner,
    PremarketGapScanner,
    Scanner,
    SpreadFilterScanner,
    TapeReaderScanner,
    VolumeSpikeScanner,
    VWAPDeviationScanner,
)
from daytrading.strategy import (
    GapReversalVerifier,
    MomentumScalpVerifier,
    Strategy,
    StrategyVerifier,
    TapeScalpVerifier,
    VolumeBreakoutVerifier,
    VWAPBounceVerifier,
)

__all__ = [
    "AdaptiveRouter",
    "BacktestEngine",
    "BacktestResult",
    "Broker",
    "CompositeScanner",
    "DataFeed",
    "ExitManager",
    "GapReversalVerifier",
    "InMemoryBarFeed",
    "MarketRegimeClassifier",
    "MomentumBurstScanner",
    "MomentumScalpVerifier",
    "PaperBroker",
    "PipelineResult",
    "PremarketGapScanner",
    "Scanner",
    "SpreadFilterScanner",
    "Strategy",
    "StrategyVerifier",
    "StyleConfig",
    "TapeReaderScanner",
    "TapeScalpVerifier",
    "TrackedPosition",
    "TradingPipeline",
    "VolumeBreakoutVerifier",
    "VolumeSpikeScanner",
    "VWAPBounceVerifier",
    "VWAPDeviationScanner",
    "__version__",
]

__version__ = "0.4.0"
