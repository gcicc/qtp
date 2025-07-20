"""
Moving Average Crossover Strategy

Implements a classic moving average crossover strategy where buy/sell signals
are generated when a fast moving average crosses above/below a slow moving average.
"""

from datetime import datetime
from typing import List
import pandas as pd
import numpy as np

from ..base import BaseStrategy, StrategyConfig, Signal, SignalType, SignalStrength
from ...data.features.technical import TechnicalIndicators


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy implementation.
    
    This strategy generates buy signals when a fast moving average crosses above
    a slow moving average, and sell signals when the fast MA crosses below the slow MA.
    
    Strategy Parameters:
    - fast_window: Period for fast moving average (default: 20)
    - slow_window: Period for slow moving average (default: 50)
    - ma_type: Type of moving average ('sma' or 'ema', default: 'sma')
    - min_cross_threshold: Minimum percentage difference for valid crossover (default: 0.001)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize the Moving Average Crossover strategy."""
        super().__init__(config)
        
        # Strategy parameters with defaults
        self.fast_window = config.get_parameter('fast_window', 20)
        self.slow_window = config.get_parameter('slow_window', 50)
        self.ma_type = config.get_parameter('ma_type', 'sma')
        self.min_cross_threshold = config.get_parameter('min_cross_threshold', 0.001)
        
        # Validate parameters
        if self.fast_window >= self.slow_window:
            raise ValueError("Fast window must be less than slow window")
        
        # State variables
        self.last_fast_ma = None
        self.last_slow_ma = None
        self.last_signal_type = None
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data contains required columns and sufficient history.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if data is valid for this strategy
        """
        required_columns = ['Close']
        
        # Check required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False
        
        # Check sufficient data for slow moving average
        if len(data) < self.slow_window:
            return False
        
        # Check for valid price data
        if data['Close'].isnull().any() or (data['Close'] <= 0).any():
            return False
            
        return True
    
    def _initialize_strategy(self, historical_data: pd.DataFrame) -> None:
        """
        Initialize strategy-specific state with historical data.
        
        Args:
            historical_data: Historical market data for initialization
        """
        # Calculate initial moving averages
        close_prices = historical_data['Close']
        
        if self.ma_type == 'sma':
            self.last_fast_ma = TechnicalIndicators.simple_moving_average(
                close_prices, self.fast_window
            ).iloc[-1]
            self.last_slow_ma = TechnicalIndicators.simple_moving_average(
                close_prices, self.slow_window
            ).iloc[-1]
        elif self.ma_type == 'ema':
            self.last_fast_ma = TechnicalIndicators.exponential_moving_average(
                close_prices, self.fast_window
            ).iloc[-1]
            self.last_slow_ma = TechnicalIndicators.exponential_moving_average(
                close_prices, self.slow_window
            ).iloc[-1]
        else:
            raise ValueError(f"Unsupported moving average type: {self.ma_type}")
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Market data DataFrame with recent history
            timestamp: Current timestamp for signal generation
            
        Returns:
            List of Signal objects (empty if no crossover detected)
        """
        signals = []
        
        # Get closing prices
        close_prices = data['Close']
        current_price = close_prices.iloc[-1]
        
        # Calculate current moving averages
        if self.ma_type == 'sma':
            fast_ma = TechnicalIndicators.simple_moving_average(
                close_prices, self.fast_window
            ).iloc[-1]
            slow_ma = TechnicalIndicators.simple_moving_average(
                close_prices, self.slow_window
            ).iloc[-1]
        else:  # ema
            fast_ma = TechnicalIndicators.exponential_moving_average(
                close_prices, self.fast_window
            ).iloc[-1]
            slow_ma = TechnicalIndicators.exponential_moving_average(
                close_prices, self.slow_window
            ).iloc[-1]
        
        # Check for crossover conditions
        signal_type = None
        reason = ""
        confidence = 0.0
        
        if self.last_fast_ma is not None and self.last_slow_ma is not None:
            # Check for bullish crossover (fast MA crosses above slow MA)
            if (self.last_fast_ma <= self.last_slow_ma and 
                fast_ma > slow_ma and 
                (fast_ma - slow_ma) / slow_ma > self.min_cross_threshold):
                
                signal_type = SignalType.BUY
                reason = f"Bullish crossover: {self.ma_type.upper()}({self.fast_window}) crossed above {self.ma_type.upper()}({self.slow_window})"
                confidence = min(0.9, (fast_ma - slow_ma) / slow_ma * 100)  # Scale confidence based on separation
                
            # Check for bearish crossover (fast MA crosses below slow MA)
            elif (self.last_fast_ma >= self.last_slow_ma and 
                  fast_ma < slow_ma and 
                  (slow_ma - fast_ma) / slow_ma > self.min_cross_threshold):
                
                signal_type = SignalType.SELL
                reason = f"Bearish crossover: {self.ma_type.upper()}({self.fast_window}) crossed below {self.ma_type.upper()}({self.slow_window})"
                confidence = min(0.9, (slow_ma - fast_ma) / slow_ma * 100)  # Scale confidence based on separation
        
        # Generate signal if crossover detected
        if signal_type is not None:
            # Determine signal strength based on MA separation
            ma_separation = abs(fast_ma - slow_ma) / slow_ma
            if ma_separation > 0.05:
                strength = SignalStrength.STRONG
            elif ma_separation > 0.02:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calculate position size based on confidence and strength
            base_position_size = self.config.max_position_size * 0.5  # Conservative default
            position_size = base_position_size * confidence * (strength.value / 4.0)
            
            if signal_type == SignalType.SELL:
                position_size = -position_size  # Negative for short positions
            
            # Create the signal
            signal = Signal(
                symbol=data.index.name if hasattr(data.index, 'name') and data.index.name else 'UNKNOWN',
                signal_type=signal_type,
                timestamp=timestamp,
                price=current_price,
                position_size=position_size,
                confidence=confidence,
                strength=strength,
                strategy_name=self.config.name,
                reason=reason,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'fast_window': self.fast_window,
                    'slow_window': self.slow_window,
                    'ma_type': self.ma_type,
                    'ma_separation': ma_separation
                }
            )
            
            signals.append(signal)
            self.last_signal_type = signal_type
        
        # Update state for next iteration
        self.last_fast_ma = fast_ma
        self.last_slow_ma = slow_ma
        
        return signals
    
    def get_strategy_info(self) -> dict:
        """Get information about the strategy configuration."""
        return {
            'strategy_name': self.config.name,
            'strategy_type': 'Moving Average Crossover',
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
            'ma_type': self.ma_type.upper(),
            'min_cross_threshold': self.min_cross_threshold,
            'current_fast_ma': self.last_fast_ma,
            'current_slow_ma': self.last_slow_ma,
            'last_signal': self.last_signal_type.value if self.last_signal_type else None
        }