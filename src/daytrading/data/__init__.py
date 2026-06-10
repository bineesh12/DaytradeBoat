from daytrading.data.feed import DataFeed, InMemoryBarFeed
from daytrading.data.market_data_service import MarketDataService

__all__ = ["DataFeed", "InMemoryBarFeed", "MarketDataService"]

try:
    from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed
    __all__.extend(["AlpacaHistoricalFeed", "AlpacaStreamFeed"])
except ImportError:
    pass
