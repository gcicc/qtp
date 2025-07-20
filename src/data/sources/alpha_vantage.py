"""
Alpha Vantage API connector for financial market data.

This module provides a comprehensive connector for Alpha Vantage's API,
offering access to stock data, forex, cryptocurrency, and economic indicators
with proper rate limiting and error handling.
"""

import time
import logging
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..ingestion.base import MarketDataIngester, DataSourceConfig

logger = logging.getLogger(__name__)


class AlphaVantageConnector(MarketDataIngester):
    """
    Alpha Vantage API connector for comprehensive financial market data.
    
    Alpha Vantage provides high-quality financial market data including:
    - Stock market data (intraday, daily, weekly, monthly)
    - Forex (FX) data
    - Cryptocurrency data  
    - Technical indicators
    - Fundamental data
    - Economic indicators
    - News and sentiment data
    
    Features:
    - Built-in rate limiting (5 API calls per minute for free tier)
    - Comprehensive error handling and retry logic
    - Data validation and cleaning
    - Support for multiple data types and timeframes
    - Automatic handling of API response formats
    
    Example:
        >>> config = DataSourceConfig(api_key="your_api_key", rate_limit=5)
        >>> connector = AlphaVantageConnector(config)
        >>> data = connector.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
        >>> print(f"Retrieved {len(data)} days of AAPL data")
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # API function mappings for different data types
    STOCK_FUNCTIONS = {
        "intraday": "TIME_SERIES_INTRADAY",
        "daily": "TIME_SERIES_DAILY_ADJUSTED",
        "weekly": "TIME_SERIES_WEEKLY_ADJUSTED",
        "monthly": "TIME_SERIES_MONTHLY_ADJUSTED"
    }
    
    INTERVAL_MAPPING = {
        "1m": "1min",
        "5m": "5min", 
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        "1d": "daily",
        "1wk": "weekly",
        "1mo": "monthly"
    }
    
    def __init__(self, config: DataSourceConfig):
        """
        Initialize Alpha Vantage connector.
        
        Args:
            config: Configuration containing API key and connection parameters
                   
        Raises:
            ValueError: If API key is not provided
        """
        if not config.api_key:
            raise ValueError("Alpha Vantage API key is required")
            
        self.config = config
        self.api_key = config.api_key
        
        # Alpha Vantage free tier: 5 API calls per minute, 500 per day
        # Premium tiers have higher limits
        self.rate_limit = config.rate_limit or 5
        self.call_timestamps = []
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Alpha Vantage connector initialized with rate limit: {self.rate_limit} calls/minute")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Retrieve historical stock market data from Alpha Vantage.
        
        Fetches OHLCV data with automatic handling of different timeframes
        and proper data validation. The method intelligently selects the
        appropriate Alpha Vantage API function based on the requested interval.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data frequency - supported values:
                - '1m', '5m', '15m', '30m', '60m', '1h': Intraday data
                - '1d': Daily data
                - '1wk': Weekly data  
                - '1mo': Monthly data
                
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Index is DatetimeIndex
            
        Raises:
            ValueError: If symbol is invalid or interval not supported
            requests.RequestException: If API request fails
            
        Example:
            >>> connector = AlphaVantageConnector(config)
            >>> data = connector.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
            >>> print(data.head())
                           Open     High      Low    Close     Volume  Adj Close
            Date                                                               
            2023-01-03  130.28  130.90  124.17  125.07  112117471     124.35
        """
        try:
            # Wait for rate limiting
            self._wait_for_rate_limit()
            
            # Determine API function based on interval
            av_interval = self.INTERVAL_MAPPING.get(interval)
            if not av_interval:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Build API parameters
            if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
                function = self.STOCK_FUNCTIONS["intraday"]
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': av_interval,
                    'outputsize': 'full',  # Get full data instead of compact (last 100 points)
                    'apikey': self.api_key
                }
            else:
                function = self.STOCK_FUNCTIONS[av_interval]
                params = {
                    'function': function,
                    'symbol': symbol,
                    'outputsize': 'full',
                    'apikey': self.api_key
                }
            
            logger.info(f"Fetching {av_interval} data for {symbol} from Alpha Vantage")
            
            response = self.session.get(
                self.BASE_URL, 
                params=params, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if 'Note' in data:
                # Rate limit hit
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                time.sleep(60)  # Wait 1 minute and retry
                return self.get_historical_data(symbol, start_date, end_date, interval)
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
                    
            if not time_series_key or time_series_key not in data:
                raise ValueError(f"No time series data found for {symbol}")
                
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Standardize column names (Alpha Vantage uses numbered keys)
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High', 
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume',
                '6. dividend amount': 'Dividend',
                '7. split coefficient': 'Split'
            }
            
            # Rename columns
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Convert data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set datetime index
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Sort by date (Alpha Vantage returns newest first)
            df = df.sort_index()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df.loc[start_dt:end_dt]
            
            # Add adjusted close if not present (for daily+ data)
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            
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
        Retrieve real-time quote data for multiple symbols.
        
        Uses Alpha Vantage's Global Quote function to get current market data
        including price, change, volume, and other key metrics.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to their current market data:
            {
                'AAPL': {
                    'price': 150.25,
                    'change': 2.30,
                    'change_percent': '1.55%',
                    'volume': 50000000,
                    'open': 148.50,
                    'high': 151.00,
                    'low': 147.80,
                    'previous_close': 147.95,
                    'timestamp': datetime(...)
                }
            }
            
        Example:
            >>> connector = AlphaVantageConnector(config)
            >>> data = connector.get_real_time_data(['AAPL', 'MSFT'])
            >>> print(f"AAPL price: ${data['AAPL']['price']}")
        """
        result = {}
        
        for symbol in symbols:
            try:
                # Wait for rate limiting
                self._wait_for_rate_limit()
                
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                logger.info(f"Fetching real-time data for {symbol}")
                
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for errors
                if 'Error Message' in data:
                    logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                    continue
                    
                if 'Note' in data:
                    logger.warning(f"Rate limit hit for {symbol}")
                    time.sleep(60)
                    continue
                
                # Extract quote data
                quote_key = 'Global Quote'
                if quote_key not in data:
                    logger.warning(f"No quote data found for {symbol}")
                    continue
                    
                quote = data[quote_key]
                
                result[symbol] = {
                    'symbol': quote.get('01. symbol'),
                    'open': self._safe_float(quote.get('02. open')),
                    'high': self._safe_float(quote.get('03. high')),
                    'low': self._safe_float(quote.get('04. low')),
                    'price': self._safe_float(quote.get('05. price')),
                    'volume': self._safe_int(quote.get('06. volume')),
                    'latest_trading_day': quote.get('07. latest trading day'),
                    'previous_close': self._safe_float(quote.get('08. previous close')),
                    'change': self._safe_float(quote.get('09. change')),
                    'change_percent': quote.get('10. change percent'),
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Failed to get real-time data for {symbol}: {e}")
                continue
                
        logger.info(f"Retrieved real-time data for {len(result)} symbols")
        return result
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve fundamental data using Alpha Vantage's Company Overview function.
        
        Fetches comprehensive company information including financial metrics,
        valuation ratios, and company details.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing fundamental metrics including:
            - Company information (name, description, sector, industry)
            - Financial metrics (market cap, P/E ratio, EPS, etc.)
            - Valuation ratios (P/B, P/S, PEG, etc.)
            - Profitability metrics (ROA, ROE, profit margin, etc.)
            - Balance sheet data (book value, debt ratios, etc.)
            
        Example:
            >>> connector = AlphaVantageConnector(config)
            >>> fundamentals = connector.get_fundamental_data("AAPL")
            >>> print(f"AAPL P/E Ratio: {fundamentals['pe_ratio']}")
        """
        try:
            # Wait for rate limiting
            self._wait_for_rate_limit()
            
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            logger.info(f"Fetching fundamental data for {symbol}")
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
                
            if 'Note' in data:
                logger.warning("Rate limit hit for fundamental data")
                time.sleep(60)
                return self.get_fundamental_data(symbol)
            
            # Convert and clean the data
            fundamentals = {
                # Company information
                'symbol': data.get('Symbol'),
                'name': data.get('Name'),
                'description': data.get('Description'),
                'exchange': data.get('Exchange'),
                'currency': data.get('Currency'),
                'country': data.get('Country'),
                'sector': data.get('Sector'),
                'industry': data.get('Industry'),
                
                # Financial metrics
                'market_cap': self._safe_int(data.get('MarketCapitalization')),
                'pe_ratio': self._safe_float(data.get('PERatio')),
                'peg_ratio': self._safe_float(data.get('PEGRatio')),
                'price_to_book': self._safe_float(data.get('PriceToBookRatio')),
                'price_to_sales': self._safe_float(data.get('PriceToSalesRatioTTM')),
                'ev_to_revenue': self._safe_float(data.get('EVToRevenue')),
                'ev_to_ebitda': self._safe_float(data.get('EVToEBITDA')),
                
                # Profitability metrics
                'profit_margin': self._safe_float(data.get('ProfitMargin')),
                'operating_margin': self._safe_float(data.get('OperatingMarginTTM')),
                'return_on_assets': self._safe_float(data.get('ReturnOnAssetsTTM')),
                'return_on_equity': self._safe_float(data.get('ReturnOnEquityTTM')),
                
                # Financial health
                'revenue_ttm': self._safe_int(data.get('RevenueTTM')),
                'gross_profit_ttm': self._safe_int(data.get('GrossProfitTTM')),
                'diluted_eps_ttm': self._safe_float(data.get('DilutedEPSTTM')),
                'quarterly_earnings_growth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                'quarterly_revenue_growth': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                
                # Balance sheet
                'book_value': self._safe_float(data.get('BookValue')),
                'shares_outstanding': self._safe_int(data.get('SharesOutstanding')),
                
                # Dividend information
                'dividend_per_share': self._safe_float(data.get('DividendPerShare')),
                'dividend_yield': self._safe_float(data.get('DividendYield')),
                'dividend_date': data.get('DividendDate'),
                'ex_dividend_date': data.get('ExDividendDate'),
                
                # Trading metrics
                'beta': self._safe_float(data.get('Beta')),
                '52_week_high': self._safe_float(data.get('52WeekHigh')),
                '52_week_low': self._safe_float(data.get('52WeekLow')),
                '50_day_ma': self._safe_float(data.get('50DayMovingAverage')),
                '200_day_ma': self._safe_float(data.get('200DayMovingAverage')),
                
                # Analyst data
                'analyst_target_price': self._safe_float(data.get('AnalystTargetPrice')),
                
                # Dates
                'latest_quarter': data.get('LatestQuarter'),
                'fiscal_year_end': data.get('FiscalYearEnd')
            }
            
            logger.info(f"Successfully retrieved fundamental data for {symbol}")
            return fundamentals
            
        except requests.RequestException as e:
            logger.error(f"Fundamental data request failed for {symbol}: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Fundamental data parsing error for {symbol}: {e}")
            raise ValueError(f"Failed to parse fundamental data for {symbol}: {e}")
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicator: str, 
        interval: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """
        Get technical indicators from Alpha Vantage.
        
        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator name (SMA, EMA, RSI, MACD, etc.)
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            **kwargs: Additional parameters for the indicator
            
        Returns:
            DataFrame with the technical indicator values
        """
        try:
            # Wait for rate limiting
            self._wait_for_rate_limit()
            
            params = {
                'function': indicator.upper(),
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key
            }
            
            # Add indicator-specific parameters
            params.update(kwargs)
            
            logger.info(f"Fetching {indicator} indicator for {symbol}")
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
                
            if 'Note' in data:
                logger.warning("Rate limit hit for technical indicators")
                time.sleep(60)
                return self.get_technical_indicators(symbol, indicator, interval, **kwargs)
            
            # Find the technical analysis key
            tech_key = None
            for key in data.keys():
                if 'Technical Analysis' in key:
                    tech_key = key
                    break
                    
            if not tech_key:
                raise ValueError(f"No technical analysis data found for {indicator}")
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[tech_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to get {indicator} for {symbol}: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """
        Test the connection to Alpha Vantage API.
        
        Performs a simple API call to verify API key validity and connectivity.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to fetch a simple quote
            test_data = self.get_real_time_data(['AAPL'])
            return 'AAPL' in test_data and test_data['AAPL'].get('price') is not None
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
        
        # If we've made too many calls, wait
        if len(self.call_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.call_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                
        # Record this call
        self.call_timestamps.append(now)
    
    def _clean_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward fill missing values
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
            invalid_volume = df['Volume'] < 0
            if invalid_volume.any():
                logger.warning(f"Removing {invalid_volume.sum()} negative volume records for {symbol}")
                df = df[~invalid_volume]
        
        return df
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return int(float(value))  # Handle cases like "1000000.0"
        except (ValueError, TypeError):
            return None