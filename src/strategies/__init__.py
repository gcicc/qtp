"""
Strategy Engine Module

Provides the foundation for creating, testing, and deploying quantitative trading strategies.

This module contains:
- Base strategy classes with standardized interfaces
- Built-in trading strategies (mean reversion, momentum, factor-based)
- Strategy validation and backtesting integration
- Signal generation and portfolio construction tools

Key Components:
- BaseStrategy: Abstract base class for all trading strategies
- BuiltinStrategies: Pre-implemented common trading strategies
- StrategyFactory: Factory for creating and configuring strategies
- SignalGenerator: Unified interface for strategy signal generation
"""

from .base import BaseStrategy, StrategyConfig, Signal
from .builtin import *
from .factory import StrategyFactory

__all__ = [
    "BaseStrategy",
    "StrategyConfig", 
    "Signal",
    "StrategyFactory",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MovingAverageCrossover"
]