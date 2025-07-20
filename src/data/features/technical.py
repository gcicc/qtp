"""
Technical Analysis Indicators

Implements common technical analysis indicators used in quantitative trading.
All indicators include proper handling of missing data and edge cases.
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Collection of technical analysis indicators.
    
    Provides static methods for calculating common technical indicators
    with proper data validation and error handling.
    """
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices: Series of price data
            window: Number of periods for the moving average
            
        Returns:
            Series with SMA values
        """
        if window <= 0:
            raise ValueError("Window must be positive")
        if len(prices) < window:
            logger.warning(f"Insufficient data for SMA calculation: {len(prices)} < {window}")
            
        return prices.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices: Series of price data
            window: Number of periods for the moving average
            alpha: Smoothing factor (if None, calculated as 2/(window+1))
            
        Returns:
            Series with EMA values
        """
        if alpha is None:
            alpha = 2.0 / (window + 1)
            
        return prices.ewm(alpha=alpha, min_periods=1).mean()
    
    @staticmethod
    def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over the specified period
        
        Args:
            prices: Series of price data
            window: Number of periods (default 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        if len(prices) < window + 1:
            logger.warning(f"Insufficient data for RSI calculation: {len(prices)} < {window + 1}")
            return pd.Series(index=prices.index, dtype=float)
            
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window, min_periods=1).mean()
        avg_losses = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of price data
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
            
        # Calculate EMAs
        ema_fast = TechnicalIndicators.exponential_moving_average(prices, fast)
        ema_slow = TechnicalIndicators.exponential_moving_average(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of price data
            window: Number of periods for moving average (default 20)
            num_std: Number of standard deviations for bands (default 2.0)
            
        Returns:
            DataFrame with upper band, middle band (SMA), and lower band
        """
        # Calculate middle band (SMA)
        middle_band = TechnicalIndicators.simple_moving_average(prices, window)
        
        # Calculate standard deviation
        std_dev = prices.rolling(window=window, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        })
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        %D = Simple Moving Average of %K
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_window: Period for %K calculation (default 14)
            d_window: Period for %D calculation (default 3)
            
        Returns:
            DataFrame with %K and %D values
        """
        # Calculate highest high and lowest low over the window
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        
        # Calculate %K
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        })
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, 
                          window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = Simple Moving Average of True Range
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Number of periods for averaging (default 14)
            
        Returns:
            Series with ATR values
        """
        # Calculate previous close
        prev_close = close.shift(1)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    @staticmethod
    def williams_percent_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                          window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Number of periods (default 14)
            
        Returns:
            Series with Williams %R values (-100 to 0)
        """
        # Calculate highest high and lowest low over the window
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        # Calculate Williams %R
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    @staticmethod
    def volume_weighted_average_price(prices: pd.Series, volumes: pd.Series, 
                                    window: Optional[int] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        VWAP = Sum(Price * Volume) / Sum(Volume)
        
        Args:
            prices: Series of price data (typically close prices)
            volumes: Series of volume data
            window: Number of periods (if None, calculates cumulative VWAP)
            
        Returns:
            Series with VWAP values
        """
        # Calculate price * volume
        price_volume = prices * volumes
        
        if window is None:
            # Cumulative VWAP
            cumulative_pv = price_volume.cumsum()
            cumulative_volume = volumes.cumsum()
            vwap = cumulative_pv / cumulative_volume
        else:
            # Rolling VWAP
            rolling_pv = price_volume.rolling(window=window, min_periods=1).sum()
            rolling_volume = volumes.rolling(window=window, min_periods=1).sum()
            vwap = rolling_pv / rolling_volume
            
        return vwap
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Money Flow Index is similar to RSI but incorporates volume.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            window: Number of periods (default 14)
            
        Returns:
            Series with MFI values (0-100)
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate money flow
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flows
        typical_price_change = typical_price.diff()
        positive_flow = money_flow.where(typical_price_change > 0, 0)
        negative_flow = money_flow.where(typical_price_change < 0, 0)
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=window, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=window, min_periods=1).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi