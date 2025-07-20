"""
Yahoo Finance data connector with rate limiting and error handling.

This module provides a robust connector for fetching market data from Yahoo Finance
with built-in rate limiting, retry logic, and comprehensive error handling.
"""

import time
import logging
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..ingestion.base import MarketDataIngester, DataSourceConfig
from ..structures import OHLCV, Quote, MarketEvent, MarketEventType

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Implements a token bucket algorithm to ensure we don't exceed
    Yahoo Finance's rate limits while maintaining good throughput.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per time window
            time_window: Time window in seconds (default 60s = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.time()
        
    def acquire(self) -> bool:
        """
        Try to acquire a token for making a request.
        
        Returns:
            True if token acquired, False if rate limit exceeded
        """
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(
            self.max_requests,
            self.tokens + (elapsed / self.time_window) * self.max_requests
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
        
    def wait_for_token(self) -> None:
        """Block until a token is available."""
        while not self.acquire():
            time.sleep(0.1)  # Small sleep to avoid busy waiting


class YahooFinanceConnector(MarketDataIngester):
    """
    Yahoo Finance data connector with comprehensive market data capabilities.
    
    Features:
    - Historical OHLCV data with multiple timeframes
    - Real-time quotes and market data
    - Fundamental company information
    - Rate limiting and retry logic
    - Data validation and cleaning
    - Support for multiple asset classes (stocks, ETFs, forex, crypto)
    
    Example:
        >>> config = DataSourceConfig(rate_limit=100, timeout=30)
        >>> connector = YahooFinanceConnector(config)
        >>> data = connector.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
        >>> print(f"Retrieved {len(data)} days of AAPL data")
    """
    
    BASE_URL = "https://query1.finance.yahoo.com"
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """
        Initialize Yahoo Finance connector.
        
        Args:
            config: Configuration for API limits and timeouts
        """
        self.config = config or DataSourceConfig(rate_limit=100, timeout=30)
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit,
            time_window=60
        )
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers to mimic browser request
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("Yahoo Finance connector initialized")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data from Yahoo Finance.
        
        This method fetches historical market data with proper error handling,
        data validation, and rate limiting. It supports multiple timeframes
        and automatically handles data cleaning and formatting.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'BTC-USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data frequency - supported values:
                - '1m': 1 minute (max 7 days)
                - '2m': 2 minutes (max 60 days)
                - '5m': 5 minutes (max 60 days)
                - '15m': 15 minutes (max 60 days)
                - '30m': 30 minutes (max 60 days)
                - '60m': 1 hour (max 60 days)
                - '90m': 90 minutes (max 60 days)
                - '1h': 1 hour (max 60 days)
                - '1d': 1 day (max 100 years)
                - '5d': 5 days
                - '1wk': 1 week
                - '1mo': 1 month
                - '3mo': 3 months
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Index is DatetimeIndex with timezone-aware timestamps
            
        Raises:
            ValueError: If symbol is invalid or date range is inappropriate
            requests.RequestException: If API request fails
            
        Example:
            >>> connector = YahooFinanceConnector()
            >>> data = connector.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
            >>> print(data.head())
                                    Open     High      Low    Close     Volume  Adj Close
            Date                                                                          
            2023-01-03 00:00:00  130.28  130.90  124.17  125.07  112117471     124.35
        """
        try:
            # Convert dates to timestamps
            start_ts = self._date_to_timestamp(start_date)
            end_ts = self._date_to_timestamp(end_date)
            
            # Validate date range for interval
            self._validate_date_range(start_date, end_date, interval)
            
            # Rate limiting
            self.rate_limiter.wait_for_token()
            
            # Build API URL
            url = f"{self.BASE_URL}/v8/finance/chart/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': interval,
                'events': 'div,splits',
                'includePrePost': 'false'
            }
            
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('chart', {}).get('result'):
                raise ValueError(f"No data returned for symbol {symbol}")
                
            result = data['chart']['result'][0]
            
            # Extract OHLCV data
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': indicators.get('open', []),
                'High': indicators.get('high', []),
                'Low': indicators.get('low', []),
                'Close': indicators.get('close', []),
                'Volume': indicators.get('volume', [])
            })
            
            # Add adjusted close if available
            if 'adjclose' in result['indicators']:
                df['Adj Close'] = result['indicators']['adjclose'][0]['adjclose']
            
            # Set datetime index
            df.index = pd.to_datetime(timestamps, unit='s')
            df.index.name = 'Date'
            
            # Convert to appropriate timezone (market timezone)
            market_tz = result.get('meta', {}).get('exchangeTimezoneName', 'UTC')
            df.index = df.index.tz_localize('UTC').tz_convert(market_tz)
            
            # Clean data
            df = self._clean_ohlcv_data(df, symbol)
            
            logger.info(f"Successfully retrieved {len(df)} records for {symbol}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"API request failed for {symbol}: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data parsing error for {symbol}: {e}")
            raise ValueError(f"Failed to parse data for {symbol}: {e}")
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve real-time market data for multiple symbols.
        
        Fetches current market data including last price, bid/ask, volume,
        and other real-time metrics. Data is returned as a dictionary with
        symbol keys and market data values.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to their current market data:
            {
                'AAPL': {
                    'price': 150.25,
                    'change': 2.30,
                    'change_percent': 1.55,
                    'volume': 50000000,
                    'bid': 150.20,
                    'ask': 150.30,
                    'bid_size': 100,
                    'ask_size': 200,
                    'market_cap': 2500000000000,
                    'timestamp': datetime(...)
                }
            }
            
        Example:
            >>> connector = YahooFinanceConnector()
            >>> data = connector.get_real_time_data(['AAPL', 'MSFT', 'GOOGL'])
            >>> print(f"AAPL current price: ${data['AAPL']['price']}")
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_for_token()
            
            # Build symbol string
            symbol_string = ','.join(symbols)
            
            url = f"{self.BASE_URL}/v7/finance/quote"
            params = {
                'symbols': symbol_string,
                'fields': 'regularMarketPrice,regularMarketChange,regularMarketChangePercent,'
                         'regularMarketVolume,bid,ask,bidSize,askSize,marketCap,'
                         'regularMarketTime,preMarketPrice,postMarketPrice'
            }
            
            logger.info(f"Fetching real-time data for {len(symbols)} symbols")
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('quoteResponse', {}).get('result'):
                raise ValueError("No real-time data returned")
                
            quotes = data['quoteResponse']['result']
            
            result = {}
            for quote in quotes:
                symbol = quote.get('symbol')
                if not symbol:
                    continue
                    
                result[symbol] = {
                    'price': quote.get('regularMarketPrice'),
                    'change': quote.get('regularMarketChange'),
                    'change_percent': quote.get('regularMarketChangePercent'),
                    'volume': quote.get('regularMarketVolume'),
                    'bid': quote.get('bid'),
                    'ask': quote.get('ask'),
                    'bid_size': quote.get('bidSize'),
                    'ask_size': quote.get('askSize'),
                    'market_cap': quote.get('marketCap'),
                    'pre_market_price': quote.get('preMarketPrice'),
                    'post_market_price': quote.get('postMarketPrice'),
                    'timestamp': datetime.fromtimestamp(
                        quote.get('regularMarketTime', time.time())
                    )
                }
            
            logger.info(f"Successfully retrieved real-time data for {len(result)} symbols")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Real-time data request failed: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Real-time data parsing error: {e}")
            raise ValueError(f"Failed to parse real-time data: {e}")
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve fundamental data for a company.
        
        Fetches key fundamental metrics including financial ratios,
        valuation metrics, and company information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing fundamental metrics:
            {
                'market_cap': 2500000000000,
                'pe_ratio': 25.5,
                'forward_pe': 22.1,
                'peg_ratio': 1.2,
                'price_to_book': 4.5,
                'price_to_sales': 6.2,
                'enterprise_value': 2450000000000,
                'profit_margin': 0.25,
                'operating_margin': 0.30,
                'return_on_assets': 0.20,
                'return_on_equity': 0.35,
                'revenue': 400000000000,
                'revenue_per_share': 24.50,
                'quarterly_revenue_growth': 0.08,
                'gross_profit': 170000000000,
                'ebitda': 120000000000,
                'net_income': 100000000000,
                'diluted_eps': 6.15,
                'quarterly_earnings_growth': 0.12,
                'total_cash': 50000000000,
                'total_debt': 120000000000,
                'debt_to_equity': 1.8,
                'current_ratio': 1.1,
                'book_value_per_share': 4.20,
                'operating_cash_flow': 110000000000,
                'levered_free_cash_flow': 80000000000,
                'dividend_rate': 0.88,
                'dividend_yield': 0.006,
                'beta': 1.2,
                '52_week_high': 180.0,
                '52_week_low': 120.0,
                'shares_outstanding': 16000000000,
                'float_shares': 15500000000,
                'held_by_insiders': 0.001,
                'held_by_institutions': 0.60,
                'short_ratio': 1.5,
                'short_percent_of_float': 0.015
            }
            
        Example:
            >>> connector = YahooFinanceConnector()
            >>> fundamentals = connector.get_fundamental_data("AAPL")
            >>> print(f"AAPL P/E Ratio: {fundamentals['pe_ratio']}")
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_for_token()
            
            url = f"{self.BASE_URL}/v10/finance/quoteSummary/{symbol}"
            params = {
                'modules': 'financialData,defaultKeyStatistics,summaryDetail,'
                          'balanceSheetHistory,incomeStatementHistory,cashflowStatementHistory'
            }
            
            logger.info(f"Fetching fundamental data for {symbol}")
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('quoteSummary', {}).get('result'):
                raise ValueError(f"No fundamental data returned for {symbol}")
                
            result = data['quoteSummary']['result'][0]
            
            # Extract key metrics from different modules
            financial_data = result.get('financialData', {})
            key_stats = result.get('defaultKeyStatistics', {})
            summary_detail = result.get('summaryDetail', {})
            
            fundamentals = {
                # Valuation metrics
                'market_cap': self._extract_value(key_stats.get('marketCap')),
                'enterprise_value': self._extract_value(key_stats.get('enterpriseValue')),
                'pe_ratio': self._extract_value(summary_detail.get('trailingPE')),
                'forward_pe': self._extract_value(summary_detail.get('forwardPE')),
                'peg_ratio': self._extract_value(key_stats.get('pegRatio')),
                'price_to_book': self._extract_value(key_stats.get('priceToBook')),
                'price_to_sales': self._extract_value(key_stats.get('priceToSalesTrailing12Months')),
                
                # Profitability metrics
                'profit_margin': self._extract_value(financial_data.get('profitMargins')),
                'operating_margin': self._extract_value(financial_data.get('operatingMargins')),
                'return_on_assets': self._extract_value(financial_data.get('returnOnAssets')),
                'return_on_equity': self._extract_value(financial_data.get('returnOnEquity')),
                
                # Financial metrics
                'revenue': self._extract_value(financial_data.get('totalRevenue')),
                'revenue_per_share': self._extract_value(financial_data.get('revenuePerShare')),
                'quarterly_revenue_growth': self._extract_value(financial_data.get('revenueGrowth')),
                'gross_profit': self._extract_value(financial_data.get('grossProfits')),
                'ebitda': self._extract_value(financial_data.get('ebitda')),
                'net_income': self._extract_value(financial_data.get('netIncomeToCommon')),
                'diluted_eps': self._extract_value(key_stats.get('trailingEps')),
                'quarterly_earnings_growth': self._extract_value(financial_data.get('earningsGrowth')),
                
                # Balance sheet metrics
                'total_cash': self._extract_value(financial_data.get('totalCash')),
                'total_debt': self._extract_value(financial_data.get('totalDebt')),
                'debt_to_equity': self._extract_value(financial_data.get('debtToEquity')),
                'current_ratio': self._extract_value(financial_data.get('currentRatio')),
                'book_value_per_share': self._extract_value(key_stats.get('bookValue')),
                
                # Cash flow metrics
                'operating_cash_flow': self._extract_value(financial_data.get('operatingCashflow')),
                'levered_free_cash_flow': self._extract_value(financial_data.get('freeCashflow')),
                
                # Dividend and yield metrics
                'dividend_rate': self._extract_value(summary_detail.get('dividendRate')),
                'dividend_yield': self._extract_value(summary_detail.get('dividendYield')),
                
                # Risk metrics
                'beta': self._extract_value(key_stats.get('beta')),
                
                # Price metrics
                '52_week_high': self._extract_value(summary_detail.get('fiftyTwoWeekHigh')),
                '52_week_low': self._extract_value(summary_detail.get('fiftyTwoWeekLow')),
                
                # Share metrics
                'shares_outstanding': self._extract_value(key_stats.get('sharesOutstanding')),
                'float_shares': self._extract_value(key_stats.get('floatShares')),
                'held_by_insiders': self._extract_value(key_stats.get('heldPercentInsiders')),
                'held_by_institutions': self._extract_value(key_stats.get('heldPercentInstitutions')),
                'short_ratio': self._extract_value(key_stats.get('shortRatio')),
                'short_percent_of_float': self._extract_value(key_stats.get('shortPercentOfFloat'))
            }
            
            logger.info(f"Successfully retrieved fundamental data for {symbol}")
            return fundamentals
            
        except requests.RequestException as e:
            logger.error(f"Fundamental data request failed for {symbol}: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Fundamental data parsing error for {symbol}: {e}")
            raise ValueError(f"Failed to parse fundamental data for {symbol}: {e}")
    
    def validate_connection(self) -> bool:
        """
        Test the connection to Yahoo Finance API.
        
        Performs a simple API call to verify connectivity and API availability.
        
        Returns:
            True if connection is successful, False otherwise
            
        Example:
            >>> connector = YahooFinanceConnector()
            >>> if connector.validate_connection():
            ...     print("Yahoo Finance API is accessible")
            ... else:
            ...     print("Cannot connect to Yahoo Finance API")
        """
        try:
            # Try to fetch a simple quote for a well-known symbol
            test_data = self.get_real_time_data(['AAPL'])
            return 'AAPL' in test_data and test_data['AAPL'].get('price') is not None
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def _date_to_timestamp(self, date_input: Union[str, date, datetime]) -> int:
        """Convert various date formats to Unix timestamp."""
        if isinstance(date_input, str):
            dt = pd.to_datetime(date_input)
        elif isinstance(date_input, date):
            dt = pd.to_datetime(date_input)
        else:
            dt = date_input
            
        return int(dt.timestamp())
    
    def _validate_date_range(self, start_date: Union[str, date, datetime], 
                           end_date: Union[str, date, datetime], interval: str) -> None:
        """Validate date range is appropriate for the requested interval."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
            
        # Check interval-specific limitations
        days_diff = (end_dt - start_dt).days
        
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        if interval in intraday_intervals and days_diff > 60:
            raise ValueError(f"Intraday interval {interval} limited to 60 days maximum")
        elif interval == '1m' and days_diff > 7:
            raise ValueError("1-minute interval limited to 7 days maximum")
    
    def _clean_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward fill missing values (common in financial data)
        df = df.fillna(method='ffill')
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        # Ensure positive prices and volumes
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = df[col] <= 0
                if invalid_prices.any():
                    logger.warning(f"Removing {invalid_prices.sum()} invalid {col} prices for {symbol}")
                    df = df[~invalid_prices]
        
        if 'Volume' in df.columns:
            # Volume can be 0 (weekends/holidays) but not negative
            invalid_volume = df['Volume'] < 0
            if invalid_volume.any():
                logger.warning(f"Removing {invalid_volume.sum()} negative volume records for {symbol}")
                df = df[~invalid_volume]
        
        # Validate OHLC relationships
        if all(col in df.columns for col in price_cols):
            # High should be >= all other prices
            high_violations = (
                (df['High'] < df['Open']) |
                (df['High'] < df['Low']) |
                (df['High'] < df['Close'])
            )
            
            # Low should be <= all other prices
            low_violations = (
                (df['Low'] > df['Open']) |
                (df['Low'] > df['High']) |
                (df['Low'] > df['Close'])
            )
            
            violations = high_violations | low_violations
            if violations.any():
                logger.warning(f"Removing {violations.sum()} OHLC relationship violations for {symbol}")
                df = df[~violations]
        
        return df
    
    def _extract_value(self, value_dict: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract numeric value from Yahoo Finance API response format."""
        if not value_dict or not isinstance(value_dict, dict):
            return None
            
        # Yahoo Finance returns values in 'raw' and 'fmt' format
        raw_value = value_dict.get('raw')
        if raw_value is not None and not (isinstance(raw_value, str) and raw_value.lower() in ['n/a', 'null']):
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                pass
                
        return None