"""
Base classes for market data ingestion.

Defines the core interfaces and abstract base classes that all data ingestion
components must implement to ensure consistency across different data sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
import pandas as pd


class MarketDataIngester(ABC):
    """
    Abstract base class for market data ingestion.
    
    All data source connectors must implement this interface to provide
    a consistent API for data retrieval across different providers.
    """
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Retrieve historical market data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            interval: Data frequency ('1m', '5m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        pass
    
    @abstractmethod
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve real-time market data for given symbols.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to their current market data
        """
        pass
    
    @abstractmethod
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve fundamental data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing fundamental metrics (P/E, EPS, etc.)
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Test the connection to the data source.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass


class DataSourceConfig:
    """
    Configuration class for data source connections.
    
    Stores API keys, endpoints, rate limits, and other connection parameters
    for different data providers.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit: int = 100,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit  # requests per minute
        self.timeout = timeout  # seconds
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout
        }