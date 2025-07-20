"""
Enhanced feature engineering for quantitative trading.

This module provides comprehensive feature engineering capabilities including:
- Advanced technical indicators
- Statistical features and transformations
- Market microstructure features
- Cross-asset and macro-economic features
- Custom metrics and factor construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import stats
from scipy.signal import argrelextrema
import warnings

from .features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class AdvancedTechnicalIndicators:
    """
    Advanced technical indicators beyond basic OHLCV analysis.
    
    Implements sophisticated technical analysis indicators including:
    - Market structure analysis
    - Volume profile indicators
    - Volatility-based indicators
    - Momentum oscillators
    - Pattern recognition features
    """
    
    @staticmethod
    def vwap_bands(prices: pd.Series, volumes: pd.Series, window: int = 20, 
                   num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (VWAP) bands.
        
        VWAP bands combine volume-weighted pricing with volatility bands,
        providing dynamic support and resistance levels based on both
        price and volume data.
        
        Args:
            prices: Series of price data (typically close prices)
            volumes: Series of volume data
            window: Period for VWAP calculation
            num_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with VWAP, upper band, and lower band
        """
        # Calculate VWAP
        vwap = TechnicalIndicators.volume_weighted_average_price(prices, volumes, window)
        
        # Calculate volume-weighted standard deviation
        pv = prices * volumes
        rolling_pv = pv.rolling(window=window).sum()
        rolling_v = volumes.rolling(window=window).sum()
        rolling_vwap = rolling_pv / rolling_v
        
        # Variance calculation
        price_diff_squared = (prices - rolling_vwap) ** 2
        weighted_variance = (price_diff_squared * volumes).rolling(window=window).sum() / rolling_v
        vwap_std = np.sqrt(weighted_variance)
        
        # Calculate bands
        upper_band = vwap + (vwap_std * num_std)
        lower_band = vwap - (vwap_std * num_std)
        
        return pd.DataFrame({
            'VWAP': vwap,
            'VWAP_Upper': upper_band,
            'VWAP_Lower': lower_band,
            'VWAP_Std': vwap_std
        })
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        window: int = 20, atr_period: int = 10, 
                        multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Keltner Channels use Average True Range (ATR) to set channel
        distance from the moving average, providing dynamic volatility-based
        support and resistance levels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Period for the moving average
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for channel width
            
        Returns:
            DataFrame with middle line, upper channel, and lower channel
        """
        # Calculate middle line (EMA)
        middle_line = TechnicalIndicators.exponential_moving_average(close, window)
        
        # Calculate ATR
        atr = TechnicalIndicators.average_true_range(high, low, close, atr_period)
        
        # Calculate channels
        upper_channel = middle_line + (atr * multiplier)
        lower_channel = middle_line - (atr * multiplier)
        
        return pd.DataFrame({
            'KC_Middle': middle_line,
            'KC_Upper': upper_channel,
            'KC_Lower': lower_channel,
            'KC_Width': upper_channel - lower_channel
        })
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.DataFrame:
        """
        Calculate Donchian Channels.
        
        Donchian Channels track the highest high and lowest low over
        a specified period, providing breakout-based support and resistance.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Lookback period
            
        Returns:
            DataFrame with upper channel, lower channel, and middle line
        """
        upper_channel = high.rolling(window=window).max()
        lower_channel = low.rolling(window=window).min()
        middle_line = (upper_channel + lower_channel) / 2
        
        return pd.DataFrame({
            'DC_Upper': upper_channel,
            'DC_Lower': lower_channel,
            'DC_Middle': middle_line,
            'DC_Width': upper_channel - lower_channel
        })
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                      af_start: float = 0.02, af_increment: float = 0.02,
                      af_max: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).
        
        Parabolic SAR provides dynamic stop-loss levels that trail price,
        helping identify trend reversals and position management points.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            af_start: Starting acceleration factor
            af_increment: Acceleration factor increment
            af_max: Maximum acceleration factor
            
        Returns:
            Series with Parabolic SAR values
        """
        psar = pd.Series(index=close.index, dtype=float)
        
        if len(close) < 2:
            return psar
        
        # Initialize
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = af_start
        ep = high.iloc[0]  # Extreme point
        psar.iloc[0] = low.iloc[0]
        
        for i in range(1, len(close)):
            # Calculate PSAR
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            
            # Check for trend reversal
            if trend == 1:  # Uptrend
                if low.iloc[i] <= psar.iloc[i]:
                    # Trend reversal to downtrend
                    trend = -1
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
                else:
                    # Continue uptrend
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # Ensure PSAR doesn't exceed recent lows
                    psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1])
                    if i > 1:
                        psar.iloc[i] = min(psar.iloc[i], low.iloc[i-2])
            
            else:  # Downtrend
                if high.iloc[i] >= psar.iloc[i]:
                    # Trend reversal to uptrend
                    trend = 1
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
                else:
                    # Continue downtrend
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # Ensure PSAR doesn't exceed recent highs
                    psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1])
                    if i > 1:
                        psar.iloc[i] = max(psar.iloc[i], high.iloc[i-2])
        
        return psar
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan_period: int = 9, kijun_period: int = 26,
                      senkou_b_period: int = 52) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.
        
        Ichimoku provides a comprehensive trend analysis system including
        support/resistance, momentum, and trend direction signals.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            tenkan_period: Tenkan-sen (conversion line) period
            kijun_period: Kijun-sen (base line) period
            senkou_b_period: Senkou Span B period
            
        Returns:
            DataFrame with Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A) - shifted forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span) - shifted backward
        chikou_span = close.shift(-kijun_period)
        
        return pd.DataFrame({
            'Tenkan_Sen': tenkan_sen,
            'Kijun_Sen': kijun_sen,
            'Senkou_Span_A': senkou_span_a,
            'Senkou_Span_B': senkou_span_b,
            'Chikou_Span': chikou_span,
            'Cloud_Top': np.maximum(senkou_span_a, senkou_span_b),
            'Cloud_Bottom': np.minimum(senkou_span_a, senkou_span_b)
        })
    
    @staticmethod
    def zigzag(high: pd.Series, low: pd.Series, threshold: float = 0.05) -> pd.Series:
        """
        Calculate ZigZag indicator for trend analysis.
        
        ZigZag filters out price changes smaller than the threshold,
        helping identify significant trend changes and support/resistance levels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            threshold: Minimum percentage change to register new direction
            
        Returns:
            Series with ZigZag values (NaN for non-reversal points)
        """
        zigzag = pd.Series(index=high.index, dtype=float)
        
        if len(high) < 3:
            return zigzag
        
        # Initialize
        direction = 0  # 0: unknown, 1: up, -1: down
        last_pivot_idx = 0
        last_pivot_value = high.iloc[0]
        
        for i in range(1, len(high)):
            if direction == 0:
                # Determine initial direction
                if high.iloc[i] > last_pivot_value * (1 + threshold):
                    direction = 1
                    zigzag.iloc[last_pivot_idx] = last_pivot_value
                elif low.iloc[i] < last_pivot_value * (1 - threshold):
                    direction = -1
                    zigzag.iloc[last_pivot_idx] = last_pivot_value
                    
            elif direction == 1:  # Uptrend
                if high.iloc[i] > last_pivot_value:
                    # Update high
                    last_pivot_value = high.iloc[i]
                    last_pivot_idx = i
                elif low.iloc[i] < last_pivot_value * (1 - threshold):
                    # Reversal to downtrend
                    zigzag.iloc[last_pivot_idx] = last_pivot_value
                    direction = -1
                    last_pivot_value = low.iloc[i]
                    last_pivot_idx = i
                    
            else:  # Downtrend
                if low.iloc[i] < last_pivot_value:
                    # Update low
                    last_pivot_value = low.iloc[i]
                    last_pivot_idx = i
                elif high.iloc[i] > last_pivot_value * (1 + threshold):
                    # Reversal to uptrend
                    zigzag.iloc[last_pivot_idx] = last_pivot_value
                    direction = 1
                    last_pivot_value = high.iloc[i]
                    last_pivot_idx = i
        
        # Mark final pivot
        if direction != 0:
            zigzag.iloc[last_pivot_idx] = last_pivot_value
        
        return zigzag


class StatisticalFeatures:
    """
    Statistical feature engineering for quantitative analysis.
    
    Provides features based on statistical analysis of price and volume data:
    - Rolling statistical moments
    - Correlation and cointegration features
    - Regime detection features
    - Distribution-based features
    """
    
    @staticmethod
    def rolling_moments(data: pd.Series, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate rolling statistical moments.
        
        Computes rolling mean, standard deviation, skewness, and kurtosis
        over multiple time windows to capture statistical properties.
        
        Args:
            data: Time series data
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling moments for each window
        """
        results = pd.DataFrame(index=data.index)
        
        for window in windows:
            prefix = f"rolling_{window}"
            
            results[f"{prefix}_mean"] = data.rolling(window).mean()
            results[f"{prefix}_std"] = data.rolling(window).std()
            results[f"{prefix}_skew"] = data.rolling(window).skew()
            results[f"{prefix}_kurt"] = data.rolling(window).kurt()
            
            # Additional moments
            results[f"{prefix}_var"] = data.rolling(window).var()
            results[f"{prefix}_min"] = data.rolling(window).min()
            results[f"{prefix}_max"] = data.rolling(window).max()
            results[f"{prefix}_median"] = data.rolling(window).median()
            results[f"{prefix}_quantile_25"] = data.rolling(window).quantile(0.25)
            results[f"{prefix}_quantile_75"] = data.rolling(window).quantile(0.75)
        
        return results
    
    @staticmethod
    def rolling_correlations(data1: pd.Series, data2: pd.Series, 
                           windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Calculate rolling correlations between two series.
        
        Args:
            data1: First time series
            data2: Second time series
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling correlations
        """
        results = pd.DataFrame(index=data1.index)
        
        for window in windows:
            results[f"corr_{window}"] = data1.rolling(window).corr(data2)
        
        return results
    
    @staticmethod
    def z_score_features(data: pd.Series, windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Calculate Z-score features for mean reversion analysis.
        
        Args:
            data: Time series data
            windows: List of lookback windows
            
        Returns:
            DataFrame with Z-score features
        """
        results = pd.DataFrame(index=data.index)
        
        for window in windows:
            rolling_mean = data.rolling(window).mean()
            rolling_std = data.rolling(window).std()
            
            results[f"zscore_{window}"] = (data - rolling_mean) / rolling_std
            results[f"zscore_abs_{window}"] = np.abs(results[f"zscore_{window}"])
        
        return results
    
    @staticmethod
    def percentile_rank(data: pd.Series, windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Calculate percentile rank of current value within rolling window.
        
        Args:
            data: Time series data
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with percentile ranks
        """
        results = pd.DataFrame(index=data.index)
        
        for window in windows:
            def rolling_rank(x):
                if len(x) == 0:
                    return np.nan
                return stats.percentileofscore(x[:-1], x[-1]) / 100.0 if len(x) > 1 else 0.5
            
            results[f"percentile_rank_{window}"] = data.rolling(window).apply(
                rolling_rank, raw=False
            )
        
        return results


class MarketMicrostructureFeatures:
    """
    Market microstructure features for high-frequency analysis.
    
    Extracts features related to market microstructure including:
    - Bid-ask spread analysis
    - Order flow indicators
    - Price impact measures
    - Liquidity indicators
    """
    
    @staticmethod
    def spread_features(bid: pd.Series, ask: pd.Series, 
                       mid_price: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate bid-ask spread features.
        
        Args:
            bid: Bid price series
            ask: Ask price series
            mid_price: Optional mid price series (if None, calculated as (bid+ask)/2)
            
        Returns:
            DataFrame with spread features
        """
        if mid_price is None:
            mid_price = (bid + ask) / 2
        
        absolute_spread = ask - bid
        relative_spread = absolute_spread / mid_price
        
        results = pd.DataFrame({
            'absolute_spread': absolute_spread,
            'relative_spread': relative_spread,
            'spread_bps': relative_spread * 10000,
            'bid_ask_ratio': bid / ask,
            'mid_price': mid_price
        }, index=bid.index)
        
        # Rolling spread statistics
        for window in [10, 20, 50]:
            results[f'spread_mean_{window}'] = relative_spread.rolling(window).mean()
            results[f'spread_std_{window}'] = relative_spread.rolling(window).std()
            results[f'spread_max_{window}'] = relative_spread.rolling(window).max()
            results[f'spread_min_{window}'] = relative_spread.rolling(window).min()
        
        return results
    
    @staticmethod
    def volume_profile_features(prices: pd.Series, volumes: pd.Series,
                              price_bins: int = 20) -> pd.DataFrame:
        """
        Calculate volume profile features.
        
        Args:
            prices: Price series
            volumes: Volume series
            price_bins: Number of price bins for volume profile
            
        Returns:
            DataFrame with volume profile features
        """
        results = pd.DataFrame(index=prices.index)
        
        # Rolling volume profile analysis
        window = min(50, len(prices) // 2)
        
        if window > 10:
            def volume_profile_stats(price_vol_data):
                if len(price_vol_data) < 5:
                    return pd.Series([np.nan] * 4)
                
                prices_window = price_vol_data.iloc[:, 0]
                volumes_window = price_vol_data.iloc[:, 1]
                
                # Create price bins
                price_range = prices_window.max() - prices_window.min()
                if price_range == 0:
                    return pd.Series([np.nan] * 4)
                
                bins = np.linspace(prices_window.min(), prices_window.max(), price_bins)
                digitized = np.digitize(prices_window, bins)
                
                # Calculate volume per bin
                volume_per_bin = np.zeros(len(bins))
                for i in range(1, len(bins)):
                    mask = digitized == i
                    if mask.any():
                        volume_per_bin[i] = volumes_window[mask].sum()
                
                # Volume profile statistics
                max_volume_idx = np.argmax(volume_per_bin)
                poc_price = bins[max_volume_idx] if max_volume_idx < len(bins) else np.nan
                
                total_volume = volume_per_bin.sum()
                if total_volume > 0:
                    volume_above_poc = volume_per_bin[max_volume_idx:].sum() / total_volume
                    volume_below_poc = volume_per_bin[:max_volume_idx].sum() / total_volume
                    
                    # Value area (70% of volume)
                    sorted_indices = np.argsort(volume_per_bin)[::-1]
                    cumulative_volume = 0
                    value_area_indices = []
                    
                    for idx in sorted_indices:
                        cumulative_volume += volume_per_bin[idx]
                        value_area_indices.append(idx)
                        if cumulative_volume >= total_volume * 0.7:
                            break
                    
                    value_area_high = bins[max(value_area_indices)] if value_area_indices else np.nan
                    value_area_low = bins[min(value_area_indices)] if value_area_indices else np.nan
                else:
                    volume_above_poc = np.nan
                    volume_below_poc = np.nan
                    value_area_high = np.nan
                    value_area_low = np.nan
                
                return pd.Series([poc_price, volume_above_poc, value_area_high, value_area_low])
            
            combined_data = pd.concat([prices, volumes], axis=1)
            vp_stats = combined_data.rolling(window).apply(
                volume_profile_stats, raw=False
            )
            
            if not vp_stats.empty and vp_stats.shape[1] >= 4:
                results['poc_price'] = vp_stats.iloc[:, 0]
                results['volume_above_poc'] = vp_stats.iloc[:, 1]
                results['value_area_high'] = vp_stats.iloc[:, 2]
                results['value_area_low'] = vp_stats.iloc[:, 3]
        
        return results


class MacroeconomicFeatures:
    """
    Macroeconomic and cross-asset features.
    
    Provides features based on relationships with broader market indicators:
    - Market regime indicators
    - Cross-asset correlations
    - Sector rotation features
    - Risk-on/risk-off indicators
    """
    
    @staticmethod
    def market_regime_features(returns: pd.Series, 
                             lookback_windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Calculate market regime features.
        
        Args:
            returns: Return series
            lookback_windows: List of lookback periods
            
        Returns:
            DataFrame with regime features
        """
        results = pd.DataFrame(index=returns.index)
        
        for window in lookback_windows:
            # Volatility regime
            rolling_vol = returns.rolling(window).std()
            vol_percentile = rolling_vol.rolling(window * 2).rank(pct=True)
            results[f'vol_regime_{window}'] = vol_percentile
            
            # Trend regime
            cumulative_returns = (1 + returns).rolling(window).apply(np.prod) - 1
            trend_strength = np.abs(cumulative_returns)
            results[f'trend_strength_{window}'] = trend_strength
            
            # Mean reversion regime
            rolling_mean = returns.rolling(window).mean()
            deviations = returns - rolling_mean
            mean_reversion = -1 * deviations.rolling(window // 4).mean()
            results[f'mean_reversion_{window}'] = mean_reversion
        
        return results
    
    @staticmethod
    def cross_asset_features(primary_returns: pd.Series, 
                           market_returns: pd.Series,
                           windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Calculate cross-asset features.
        
        Args:
            primary_returns: Returns of primary asset
            market_returns: Returns of market benchmark
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with cross-asset features
        """
        results = pd.DataFrame(index=primary_returns.index)
        
        for window in windows:
            # Rolling beta
            covariance = primary_returns.rolling(window).cov(market_returns)
            market_variance = market_returns.rolling(window).var()
            beta = covariance / market_variance
            results[f'beta_{window}'] = beta
            
            # Rolling correlation
            correlation = primary_returns.rolling(window).corr(market_returns)
            results[f'correlation_{window}'] = correlation
            
            # Relative strength
            primary_cumret = (1 + primary_returns).rolling(window).apply(np.prod) - 1
            market_cumret = (1 + market_returns).rolling(window).apply(np.prod) - 1
            relative_strength = primary_cumret - market_cumret
            results[f'relative_strength_{window}'] = relative_strength
        
        return results


class FeatureEngineering:
    """
    Comprehensive feature engineering pipeline.
    
    Orchestrates the creation of all feature types and provides
    a unified interface for feature generation and selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineering pipeline.
        
        Args:
            config: Configuration dictionary for feature parameters
        """
        self.config = config or self._default_config()
        self.technical_indicators = TechnicalIndicators()
        self.advanced_technical = AdvancedTechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        self.microstructure_features = MarketMicrostructureFeatures()
        self.macro_features = MacroeconomicFeatures()
    
    def create_comprehensive_features(self, data: pd.DataFrame, 
                                    symbol: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Optional symbol identifier
            
        Returns:
            DataFrame with all generated features
        """
        logger.info(f"Creating comprehensive features for {symbol or 'unknown symbol'}")
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Basic technical indicators
            if self.config.get('include_basic_technical', True):
                features = pd.concat([features, self._create_basic_technical_features(data)], axis=1)
            
            # Advanced technical indicators
            if self.config.get('include_advanced_technical', True):
                features = pd.concat([features, self._create_advanced_technical_features(data)], axis=1)
            
            # Statistical features
            if self.config.get('include_statistical', True):
                features = pd.concat([features, self._create_statistical_features(data)], axis=1)
            
            # Market microstructure features (if bid/ask data available)
            if self.config.get('include_microstructure', False) and self._has_microstructure_data(data):
                features = pd.concat([features, self._create_microstructure_features(data)], axis=1)
            
            # Volume features
            if self.config.get('include_volume', True) and 'Volume' in data.columns:
                features = pd.concat([features, self._create_volume_features(data)], axis=1)
            
            # Price action features
            if self.config.get('include_price_action', True):
                features = pd.concat([features, self._create_price_action_features(data)], axis=1)
            
            # Handle infinite values and NaN
            features = self._clean_features(features)
            
            logger.info(f"Created {features.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            raise
        
        return features
    
    def _create_basic_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical indicator features."""
        features = pd.DataFrame(index=data.index)
        
        if 'Close' in data.columns:
            close = data['Close']
            
            # Moving averages
            for window in self.config.get('ma_windows', [5, 10, 20, 50]):
                features[f'sma_{window}'] = self.technical_indicators.simple_moving_average(close, window)
                features[f'ema_{window}'] = self.technical_indicators.exponential_moving_average(close, window)
                
                # Price relative to moving averages
                features[f'price_to_sma_{window}'] = close / features[f'sma_{window}'] - 1
                features[f'price_to_ema_{window}'] = close / features[f'ema_{window}'] - 1
            
            # RSI
            for period in self.config.get('rsi_periods', [14, 21]):
                features[f'rsi_{period}'] = self.technical_indicators.relative_strength_index(close, period)
            
            # MACD
            macd_data = self.technical_indicators.macd(close)
            for col in macd_data.columns:
                features[f'macd_{col.lower()}'] = macd_data[col]
        
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            high, low, close = data['High'], data['Low'], data['Close']
            
            # Bollinger Bands
            bb_data = self.technical_indicators.bollinger_bands(close)
            for col in bb_data.columns:
                features[f'bb_{col.lower()}'] = bb_data[col]
            
            # Bollinger Band position
            features['bb_position'] = (close - bb_data['Lower']) / (bb_data['Upper'] - bb_data['Lower'])
            
            # Stochastic
            stoch_data = self.technical_indicators.stochastic_oscillator(high, low, close)
            for col in stoch_data.columns:
                features[f'stoch_{col.lower().replace("%", "pct")}'] = stoch_data[col]
            
            # ATR
            features['atr'] = self.technical_indicators.average_true_range(high, low, close)
            features['atr_pct'] = features['atr'] / close
            
            # Williams %R
            features['williams_r'] = self.technical_indicators.williams_percent_r(high, low, close)
        
        return features
    
    def _create_advanced_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicator features."""
        features = pd.DataFrame(index=data.index)
        
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            high, low, close = data['High'], data['Low'], data['Close']
            
            # Keltner Channels
            kc_data = self.advanced_technical.keltner_channels(high, low, close)
            for col in kc_data.columns:
                features[f'kc_{col.lower()}'] = kc_data[col]
            
            # Donchian Channels
            dc_data = self.advanced_technical.donchian_channels(high, low, close)
            for col in dc_data.columns:
                features[f'dc_{col.lower()}'] = dc_data[col]
            
            # Parabolic SAR
            features['parabolic_sar'] = self.advanced_technical.parabolic_sar(high, low, close)
            features['sar_signal'] = np.where(close > features['parabolic_sar'], 1, -1)
            
            # Ichimoku Cloud
            ichimoku_data = self.advanced_technical.ichimoku_cloud(high, low, close)
            for col in ichimoku_data.columns:
                features[f'ichimoku_{col.lower()}'] = ichimoku_data[col]
            
            # ZigZag
            features['zigzag'] = self.advanced_technical.zigzag(high, low)
        
        if all(col in data.columns for col in ['Close', 'Volume']):
            close, volume = data['Close'], data['Volume']
            
            # VWAP and bands
            vwap_data = self.advanced_technical.vwap_bands(close, volume)
            for col in vwap_data.columns:
                features[f'{col.lower()}'] = vwap_data[col]
            
            # Money Flow Index
            if all(col in data.columns for col in ['High', 'Low']):
                features['mfi'] = self.technical_indicators.money_flow_index(
                    data['High'], data['Low'], close, volume
                )
        
        return features
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        features = pd.DataFrame(index=data.index)
        
        if 'Close' in data.columns:
            close = data['Close']
            returns = close.pct_change()
            
            # Rolling moments
            moments = self.statistical_features.rolling_moments(returns)
            features = pd.concat([features, moments], axis=1)
            
            # Z-score features
            zscore_features = self.statistical_features.z_score_features(close)
            features = pd.concat([features, zscore_features], axis=1)
            
            # Percentile ranks
            percentile_features = self.statistical_features.percentile_rank(close)
            features = pd.concat([features, percentile_features], axis=1)
            
            # Return-based features
            for window in [5, 10, 20]:
                features[f'return_{window}d'] = returns.rolling(window).sum()
                features[f'volatility_{window}d'] = returns.rolling(window).std()
                features[f'sharpe_{window}d'] = features[f'return_{window}d'] / features[f'volatility_{window}d']
        
        return features
    
    def _create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        features = pd.DataFrame(index=data.index)
        
        if all(col in data.columns for col in ['Bid', 'Ask']):
            spread_features = self.microstructure_features.spread_features(
                data['Bid'], data['Ask']
            )
            features = pd.concat([features, spread_features], axis=1)
        
        return features
    
    def _create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        volume = data['Volume']
        
        # Volume moving averages
        for window in [5, 10, 20]:
            features[f'volume_ma_{window}'] = volume.rolling(window).mean()
            features[f'volume_ratio_{window}'] = volume / features[f'volume_ma_{window}']
        
        # Volume percentiles
        for window in [20, 50]:
            features[f'volume_percentile_{window}'] = volume.rolling(window).rank(pct=True)
        
        # On Balance Volume (OBV)
        if 'Close' in data.columns:
            price_change = data['Close'].diff()
            volume_direction = np.where(price_change > 0, volume, 
                                      np.where(price_change < 0, -volume, 0))
            features['obv'] = volume_direction.cumsum()
        
        return features
    
    def _create_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price action features."""
        features = pd.DataFrame(index=data.index)
        
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            open_price, high, low, close = data['Open'], data['High'], data['Low'], data['Close']
            
            # Candlestick features
            features['body_size'] = np.abs(close - open_price)
            features['upper_shadow'] = high - np.maximum(open_price, close)
            features['lower_shadow'] = np.minimum(open_price, close) - low
            features['total_range'] = high - low
            
            # Relative sizes
            features['body_to_range'] = features['body_size'] / features['total_range']
            features['upper_shadow_to_range'] = features['upper_shadow'] / features['total_range']
            features['lower_shadow_to_range'] = features['lower_shadow'] / features['total_range']
            
            # Price gaps
            features['gap'] = open_price - close.shift(1)
            features['gap_pct'] = features['gap'] / close.shift(1)
            
            # Intraday returns
            features['intraday_return'] = (close - open_price) / open_price
            features['overnight_return'] = (open_price - close.shift(1)) / close.shift(1)
        
        return features
    
    def _has_microstructure_data(self, data: pd.DataFrame) -> bool:
        """Check if data contains microstructure information."""
        return all(col in data.columns for col in ['Bid', 'Ask'])
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling infinite values and excessive NaN."""
        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many NaN values
        nan_threshold = self.config.get('max_nan_percentage', 0.5)
        nan_percentage = features.isnull().sum() / len(features)
        valid_columns = nan_percentage[nan_percentage <= nan_threshold].index
        features = features[valid_columns]
        
        # Forward fill remaining NaN values
        features = features.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        features = features.dropna()
        
        return features
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for feature engineering."""
        return {
            'include_basic_technical': True,
            'include_advanced_technical': True,
            'include_statistical': True,
            'include_microstructure': False,
            'include_volume': True,
            'include_price_action': True,
            'ma_windows': [5, 10, 20, 50],
            'rsi_periods': [14, 21],
            'max_nan_percentage': 0.5
        }