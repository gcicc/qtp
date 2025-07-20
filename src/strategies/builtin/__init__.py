"""
Built-in Trading Strategies

Collection of pre-implemented trading strategies that demonstrate common
quantitative trading approaches and serve as examples for custom strategy development.

Available Strategies:
- MeanReversionStrategy: Statistical mean reversion trading
- MomentumStrategy: Trend following and momentum strategies  
- MovingAverageCrossover: Simple moving average crossover signals
- BollingerBandsStrategy: Bollinger Bands mean reversion
- RSIStrategy: RSI-based overbought/oversold signals
"""

from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageCrossover
from .bollinger_bands import BollingerBandsStrategy
from .rsi_strategy import RSIStrategy

__all__ = [
    "MeanReversionStrategy",
    "MomentumStrategy", 
    "MovingAverageCrossover",
    "BollingerBandsStrategy",
    "RSIStrategy"
]