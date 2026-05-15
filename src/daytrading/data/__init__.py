from daytrading.data.feed import DataFeed, InMemoryBarFeed

__all__ = ["DataFeed", "InMemoryBarFeed"]

try:
    from daytrading.data.alpaca_feed import AlpacaHistoricalFeed, AlpacaStreamFeed
    __all__.extend(["AlpacaHistoricalFeed", "AlpacaStreamFeed"])
except ImportError:
    pass
