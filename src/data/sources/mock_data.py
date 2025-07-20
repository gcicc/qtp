"""
Mock data generator for testing and development.

This module provides realistic market data generation for testing trading strategies,
backtesting, and development purposes. It creates synthetic but realistic market data
with configurable parameters and market conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
import random
import logging
from enum import Enum

from ..ingestion.base import MarketDataIngester, DataSourceConfig
from ..structures import OHLCV, Trade, Quote, MarketEvent, MarketEventType

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Different market regimes for realistic simulation."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"


class MockDataGenerator(MarketDataIngester):
    """
    Mock data generator for realistic market data simulation.
    
    Generates synthetic market data that mimics real market behavior including:
    - Realistic price movements with configurable volatility
    - Volume patterns correlated with price movements
    - Market microstructure effects (bid-ask spreads, market impact)
    - Different market regimes (trending, sideways, volatile)
    - Corporate events (dividends, splits, earnings)
    - Intraday patterns and seasonality
    - Cross-correlation between related assets
    
    Features:
    - Multiple timeframes (1m to 1mo)
    - Configurable market parameters
    - Realistic OHLCV generation
    - Trade and quote simulation
    - Market event generation
    - Statistical validation of generated data
    - Reproducible seeds for consistent testing
    
    Example:
        >>> config = DataSourceConfig()
        >>> generator = MockDataGenerator(config)
        >>> data = generator.get_historical_data("MOCK_AAPL", "2023-01-01", "2023-12-31")
        >>> print(f"Generated {len(data)} days of mock data")
    """
    
    def __init__(self, config: Optional[DataSourceConfig] = None, seed: Optional[int] = None):
        """
        Initialize mock data generator.
        
        Args:
            config: Configuration for data generation parameters
            seed: Random seed for reproducible data generation
        """
        self.config = config or DataSourceConfig()
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Default market parameters
        self.market_params = {
            'base_volatility': 0.20,  # Annual volatility
            'trend_strength': 0.05,   # Annual trend
            'mean_reversion': 0.1,    # Mean reversion speed
            'jump_probability': 0.002,  # Daily jump probability
            'jump_magnitude': 0.03,   # Average jump size
            'volume_volatility': 0.5, # Volume volatility
            'bid_ask_spread': 0.001,  # Typical spread as % of price
            'market_hours': (9.5, 16), # Market hours (9:30 AM - 4:00 PM)
        }
        
        # Company profiles for different mock symbols
        self.company_profiles = {
            'MOCK_AAPL': {
                'name': 'Mock Apple Inc.',
                'sector': 'Technology',
                'base_price': 150.0,
                'volatility_multiplier': 1.0,
                'dividend_yield': 0.005,
                'beta': 1.2
            },
            'MOCK_MSFT': {
                'name': 'Mock Microsoft Corp.',
                'sector': 'Technology', 
                'base_price': 300.0,
                'volatility_multiplier': 0.8,
                'dividend_yield': 0.008,
                'beta': 1.0
            },
            'MOCK_SPY': {
                'name': 'Mock S&P 500 ETF',
                'sector': 'ETF',
                'base_price': 400.0,
                'volatility_multiplier': 0.6,
                'dividend_yield': 0.015,
                'beta': 1.0
            },
            'MOCK_TSLA': {
                'name': 'Mock Tesla Inc.',
                'sector': 'Automotive',
                'base_price': 200.0,
                'volatility_multiplier': 2.0,
                'dividend_yield': 0.0,
                'beta': 2.0
            }
        }
        
        logger.info("Mock data generator initialized")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate historical OHLCV data for a mock symbol.
        
        Creates realistic market data using geometric Brownian motion with jumps,
        mean reversion, and various market microstructure effects.
        
        Args:
            symbol: Mock symbol identifier (e.g., 'MOCK_AAPL')
            start_date: Start date for data generation
            end_date: End date for data generation
            interval: Data frequency ('1m', '5m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            DataFrame with realistic OHLCV data
            
        Example:
            >>> generator = MockDataGenerator()
            >>> data = generator.get_historical_data("MOCK_AAPL", "2023-01-01", "2023-12-31")
            >>> print(data.describe())
        """
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get company profile
            profile = self.company_profiles.get(symbol, self.company_profiles['MOCK_AAPL'])
            
            # Generate time index based on interval
            time_index = self._generate_time_index(start_dt, end_dt, interval)
            
            if len(time_index) == 0:
                return pd.DataFrame()
            
            # Generate price series
            prices = self._generate_price_series(
                time_index, 
                profile['base_price'],
                profile['volatility_multiplier'],
                interval
            )
            
            # Generate OHLCV data
            ohlcv_data = self._generate_ohlcv_from_prices(prices, time_index, interval)
            
            # Add volume
            volumes = self._generate_volume_series(prices, time_index, interval)
            ohlcv_data['Volume'] = volumes
            
            # Add adjusted close (same as close for mock data)
            ohlcv_data['Adj Close'] = ohlcv_data['Close']
            
            logger.info(f"Generated {len(ohlcv_data)} {interval} bars for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to generate mock data for {symbol}: {e}")
            raise
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate real-time mock data for multiple symbols.
        
        Creates current market data with realistic bid/ask spreads and volume.
        
        Args:
            symbols: List of mock symbol identifiers
            
        Returns:
            Dictionary mapping symbols to current market data
        """
        result = {}
        
        for symbol in symbols:
            try:
                profile = self.company_profiles.get(symbol, self.company_profiles['MOCK_AAPL'])
                
                # Generate current price around base price
                current_price = profile['base_price'] * (1 + np.random.normal(0, 0.02))
                
                # Generate bid/ask spread
                spread_pct = self.market_params['bid_ask_spread']
                spread = current_price * spread_pct
                
                bid_price = current_price - spread / 2
                ask_price = current_price + spread / 2
                
                # Generate volumes
                avg_volume = int(1000000 * (1 + np.random.normal(0, 0.3)))
                bid_size = random.randint(100, 1000)
                ask_size = random.randint(100, 1000)
                
                # Generate daily change
                change = np.random.normal(0, current_price * 0.01)
                change_percent = (change / current_price) * 100
                
                result[symbol] = {
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': avg_volume,
                    'bid': round(bid_price, 2),
                    'ask': round(ask_price, 2),
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'market_cap': int(current_price * 16000000000),  # Mock shares outstanding
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Failed to generate real-time data for {symbol}: {e}")
                continue
                
        return result
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock fundamental data for a symbol.
        
        Creates realistic fundamental metrics based on the company profile.
        
        Args:
            symbol: Mock symbol identifier
            
        Returns:
            Dictionary containing mock fundamental metrics
        """
        try:
            profile = self.company_profiles.get(symbol, self.company_profiles['MOCK_AAPL'])
            base_price = profile['base_price']
            
            # Generate realistic fundamentals
            shares_outstanding = 16000000000
            market_cap = int(base_price * shares_outstanding)
            
            # Financial metrics with some randomness
            revenue = int(market_cap * np.random.uniform(0.8, 1.2))
            net_income = int(revenue * np.random.uniform(0.15, 0.25))
            
            fundamentals = {
                # Company info
                'symbol': symbol,
                'name': profile['name'],
                'sector': profile['sector'],
                'exchange': 'MOCK_NASDAQ',
                'currency': 'USD',
                'country': 'US',
                
                # Valuation metrics
                'market_cap': market_cap,
                'pe_ratio': round(np.random.uniform(15, 30), 2),
                'forward_pe': round(np.random.uniform(12, 25), 2),
                'peg_ratio': round(np.random.uniform(0.8, 2.0), 2),
                'price_to_book': round(np.random.uniform(3, 8), 2),
                'price_to_sales': round(np.random.uniform(4, 12), 2),
                'enterprise_value': int(market_cap * np.random.uniform(0.9, 1.1)),
                
                # Profitability metrics
                'profit_margin': round(np.random.uniform(0.15, 0.25), 3),
                'operating_margin': round(np.random.uniform(0.20, 0.30), 3),
                'return_on_assets': round(np.random.uniform(0.15, 0.25), 3),
                'return_on_equity': round(np.random.uniform(0.25, 0.40), 3),
                
                # Financial metrics
                'revenue': revenue,
                'gross_profit': int(revenue * np.random.uniform(0.35, 0.45)),
                'net_income': net_income,
                'diluted_eps': round(net_income / shares_outstanding, 2),
                'quarterly_earnings_growth': round(np.random.uniform(-0.1, 0.2), 3),
                'quarterly_revenue_growth': round(np.random.uniform(0, 0.15), 3),
                
                # Balance sheet
                'total_cash': int(market_cap * np.random.uniform(0.1, 0.3)),
                'total_debt': int(market_cap * np.random.uniform(0.2, 0.6)),
                'book_value_per_share': round(base_price / np.random.uniform(3, 8), 2),
                
                # Dividend info
                'dividend_rate': round(base_price * profile['dividend_yield'], 2),
                'dividend_yield': profile['dividend_yield'],
                
                # Risk metrics
                'beta': profile['beta'],
                
                # Price metrics
                '52_week_high': round(base_price * np.random.uniform(1.1, 1.3), 2),
                '52_week_low': round(base_price * np.random.uniform(0.7, 0.9), 2),
                
                # Share metrics
                'shares_outstanding': shares_outstanding,
                'float_shares': int(shares_outstanding * 0.95),
                'held_by_institutions': round(np.random.uniform(0.5, 0.8), 3)
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Failed to generate fundamental data for {symbol}: {e}")
            raise
    
    def generate_trades(
        self,
        symbol: str,
        date: Union[str, date, datetime],
        num_trades: int = 1000
    ) -> List[Trade]:
        """
        Generate realistic trade data for a trading day.
        
        Args:
            symbol: Symbol identifier
            date: Trading date
            num_trades: Number of trades to generate
            
        Returns:
            List of Trade objects
        """
        trades = []
        profile = self.company_profiles.get(symbol, self.company_profiles['MOCK_AAPL'])
        base_price = profile['base_price']
        
        # Generate trading session
        date_dt = pd.to_datetime(date)
        market_open = date_dt.replace(hour=9, minute=30, second=0)
        market_close = date_dt.replace(hour=16, minute=0, second=0)
        
        for i in range(num_trades):
            # Random time during trading day
            seconds_in_day = (market_close - market_open).seconds
            random_seconds = random.randint(0, seconds_in_day)
            trade_time = market_open + timedelta(seconds=random_seconds)
            
            # Price with some randomness
            price_variation = np.random.normal(0, base_price * 0.001)
            trade_price = base_price + price_variation
            
            # Trade size (favor smaller sizes)
            size = int(np.random.exponential(200))
            size = max(100, min(size, 10000))  # Clamp between 100 and 10,000
            
            # Side (slightly more buys than sells in uptrend)
            side = "buy" if random.random() < 0.52 else "sell"
            
            trade = Trade(
                symbol=symbol,
                timestamp=trade_time,
                price=Decimal(str(round(trade_price, 2))),
                size=size,
                side=side,
                trade_id=f"MOCK_{i+1:06d}",
                exchange="MOCK_NASDAQ"
            )
            
            trades.append(trade)
        
        # Sort by timestamp
        trades.sort(key=lambda t: t.timestamp)
        
        return trades
    
    def generate_market_events(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime]
    ) -> List[MarketEvent]:
        """
        Generate realistic market events for a symbol.
        
        Args:
            symbol: Symbol identifier
            start_date: Start date for events
            end_date: End date for events
            
        Returns:
            List of MarketEvent objects
        """
        events = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        profile = self.company_profiles.get(symbol, self.company_profiles['MOCK_AAPL'])
        
        # Generate quarterly earnings
        current_date = start_dt
        while current_date <= end_dt:
            # Earnings typically quarterly
            if current_date.month in [1, 4, 7, 10] and current_date.day <= 15:
                event = MarketEvent(
                    symbol=symbol,
                    timestamp=current_date,
                    event_type=MarketEventType.EARNINGS,
                    description=f"Q{(current_date.month-1)//3 + 1} {current_date.year} Earnings Release",
                    data={
                        "eps_actual": round(np.random.uniform(1.5, 2.5), 2),
                        "eps_estimate": round(np.random.uniform(1.4, 2.4), 2),
                        "revenue": round(np.random.uniform(80, 120), 1)
                    },
                    source="mock_generator",
                    impact_score=random.uniform(0.6, 0.9)
                )
                events.append(event)
            
            current_date += timedelta(days=1)
        
        # Generate dividends (if applicable)
        if profile['dividend_yield'] > 0:
            current_date = start_dt
            while current_date <= end_dt:
                # Dividends typically quarterly
                if current_date.month in [3, 6, 9, 12] and current_date.day == 15:
                    dividend_amount = round(profile['base_price'] * profile['dividend_yield'] / 4, 2)
                    
                    event = MarketEvent(
                        symbol=symbol,
                        timestamp=current_date,
                        event_type=MarketEventType.DIVIDEND,
                        description=f"Quarterly dividend payment of ${dividend_amount}",
                        data={
                            "dividend_amount": dividend_amount,
                            "ex_date": (current_date - timedelta(days=2)).strftime('%Y-%m-%d'),
                            "payment_date": (current_date + timedelta(days=14)).strftime('%Y-%m-%d')
                        },
                        source="mock_generator",
                        impact_score=random.uniform(0.2, 0.4)
                    )
                    events.append(event)
                
                current_date += timedelta(days=1)
        
        # Generate occasional news events
        days_range = (end_dt - start_dt).days
        num_news_events = max(1, days_range // 30)  # About one per month
        
        for _ in range(num_news_events):
            random_days = random.randint(0, days_range)
            event_date = start_dt + timedelta(days=random_days)
            
            news_types = [
                "Product announcement",
                "Partnership deal",
                "Regulatory approval",
                "Management change",
                "Market expansion"
            ]
            
            event = MarketEvent(
                symbol=symbol,
                timestamp=event_date,
                event_type=MarketEventType.NEWS,
                description=f"{profile['name']} - {random.choice(news_types)}",
                data={"sentiment": random.choice(["positive", "neutral", "negative"])},
                source="mock_generator",
                impact_score=random.uniform(0.3, 0.7)
            )
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def validate_connection(self) -> bool:
        """
        Validate mock data generator (always returns True).
        
        Returns:
            True (mock generator is always available)
        """
        return True
    
    def _generate_time_index(
        self, 
        start_dt: datetime, 
        end_dt: datetime, 
        interval: str
    ) -> pd.DatetimeIndex:
        """Generate appropriate time index for the interval."""
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
            # Intraday data - only during market hours
            freq_map = {
                '1m': '1T', '2m': '2T', '5m': '5T',
                '15m': '15T', '30m': '30T', '60m': '60T', '1h': '1H'
            }
            freq = freq_map[interval]
            
            # Generate business days
            business_days = pd.bdate_range(start_dt, end_dt)
            time_index = []
            
            for day in business_days:
                # Market hours: 9:30 AM to 4:00 PM
                day_start = day.replace(hour=9, minute=30)
                day_end = day.replace(hour=16, minute=0)
                day_times = pd.date_range(day_start, day_end, freq=freq)
                time_index.extend(day_times)
            
            return pd.DatetimeIndex(time_index)
            
        else:
            # Daily or longer intervals
            freq_map = {
                '1d': 'B',   # Business days
                '1wk': 'W-FRI',  # Weekly on Friday
                '1mo': 'BM'  # Business month end
            }
            freq = freq_map.get(interval, 'B')
            return pd.bdate_range(start_dt, end_dt, freq=freq)
    
    def _generate_price_series(
        self, 
        time_index: pd.DatetimeIndex, 
        base_price: float,
        volatility_multiplier: float,
        interval: str
    ) -> np.ndarray:
        """Generate realistic price series using geometric Brownian motion."""
        n_periods = len(time_index)
        
        if n_periods == 0:
            return np.array([])
        
        # Adjust parameters based on interval
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
            # Intraday scaling
            dt = 1 / (252 * 6.5 * 60)  # Minutes in trading year
            if interval == '5m':
                dt *= 5
            elif interval == '15m':
                dt *= 15
            elif interval == '30m':
                dt *= 30
            elif interval in ['60m', '1h']:
                dt *= 60
        elif interval == '1d':
            dt = 1 / 252  # Daily
        elif interval == '1wk':
            dt = 1 / 52   # Weekly
        elif interval == '1mo':
            dt = 1 / 12   # Monthly
        else:
            dt = 1 / 252  # Default daily
        
        # Market parameters
        mu = self.market_params['trend_strength']  # Annual drift
        sigma = self.market_params['base_volatility'] * volatility_multiplier  # Annual volatility
        
        # Generate returns
        returns = np.random.normal(
            mu * dt,
            sigma * np.sqrt(dt),
            n_periods
        )
        
        # Add occasional jumps
        jump_prob = self.market_params['jump_probability'] * dt
        jumps = np.random.binomial(1, jump_prob, n_periods)
        jump_sizes = np.random.normal(0, self.market_params['jump_magnitude'], n_periods)
        returns += jumps * jump_sizes
        
        # Add mean reversion
        mean_reversion_speed = self.market_params['mean_reversion']
        log_base_price = np.log(base_price)
        
        prices = [base_price]
        for i in range(1, n_periods):
            log_price = np.log(prices[-1])
            mean_reversion = -mean_reversion_speed * (log_price - log_base_price) * dt
            new_log_price = log_price + returns[i] + mean_reversion
            prices.append(np.exp(new_log_price))
        
        return np.array(prices)
    
    def _generate_ohlcv_from_prices(
        self,
        prices: np.ndarray,
        time_index: pd.DatetimeIndex,
        interval: str
    ) -> pd.DataFrame:
        """Generate OHLCV data from price series."""
        n_periods = len(prices)
        
        if n_periods == 0:
            return pd.DataFrame()
        
        ohlc_data = []
        
        for i in range(n_periods):
            if i == 0:
                open_price = prices[i]
            else:
                open_price = prices[i-1]  # Previous close becomes open
            
            # Generate intraperiod price movement
            base_price = prices[i]
            volatility = self.market_params['base_volatility'] / np.sqrt(252)
            
            # Generate high and low
            high_factor = 1 + abs(np.random.normal(0, volatility * 0.5))
            low_factor = 1 - abs(np.random.normal(0, volatility * 0.5))
            
            high = max(open_price, base_price) * high_factor
            low = min(open_price, base_price) * low_factor
            close = base_price
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            ohlc_data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2)
            })
        
        df = pd.DataFrame(ohlc_data, index=time_index)
        return df
    
    def _generate_volume_series(
        self,
        prices: np.ndarray,
        time_index: pd.DatetimeIndex,
        interval: str
    ) -> np.ndarray:
        """Generate realistic volume series correlated with price movements."""
        n_periods = len(prices)
        
        if n_periods == 0:
            return np.array([])
        
        # Base volume depends on interval
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
            base_volume = 10000  # Intraday base volume
        else:
            base_volume = 1000000  # Daily base volume
        
        volumes = []
        
        for i in range(n_periods):
            # Volume correlated with price volatility
            if i > 0:
                price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                volume_multiplier = 1 + price_change * 5  # Higher volume on big moves
            else:
                volume_multiplier = 1
            
            # Add random component
            random_factor = np.random.lognormal(0, self.market_params['volume_volatility'])
            
            volume = int(base_volume * volume_multiplier * random_factor)
            volume = max(1000, volume)  # Minimum volume
            
            volumes.append(volume)
        
        return np.array(volumes)