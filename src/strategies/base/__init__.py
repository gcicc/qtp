"""
Base Strategy Classes

Defines the core interfaces and abstract base classes that all trading strategies
must implement to ensure consistency and interoperability within the QTP platform.
"""

from .strategy import BaseStrategy, StrategyConfig
from .signal import Signal, SignalType, SignalStrength
from .portfolio import PortfolioManager

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "Signal", 
    "SignalType",
    "SignalStrength",
    "PortfolioManager"
]