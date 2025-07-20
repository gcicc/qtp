"""
Market Data Ingestion Module

Provides interfaces and implementations for collecting market data from various sources
including real-time feeds, historical databases, and alternative data providers.

Key Components:
- MarketDataIngester: Main interface for data collection
- DataSourceConnectors: Adapters for different data providers (Yahoo, Alpha Vantage, etc.)
- RealTimeFeeds: WebSocket and REST API connections for live data
- DataCaching: Efficient storage and retrieval of historical data
"""

from .base import MarketDataIngester
from .sources import *
from .feeds import *

__all__ = [
    "MarketDataIngester",
    "YahooFinanceConnector",
    "AlphaVantageConnector", 
    "RealTimeFeed"
]