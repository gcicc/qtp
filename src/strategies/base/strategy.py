"""
Base strategy classes and interfaces.

Defines the abstract base class that all trading strategies must inherit from,
ensuring a consistent API across different strategy implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import logging

from .signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """
    Configuration class for trading strategies.
    
    Contains all parameters and settings needed to configure a strategy,
    including risk parameters, lookback windows, and strategy-specific settings.
    """
    
    # Basic strategy information
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Risk management parameters
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    max_drawdown: float = 0.05      # Maximum allowed drawdown (5%)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Data requirements
    required_data: List[str] = None  # List of required data columns
    lookback_window: int = 252       # Number of periods to look back
    
    # Strategy-specific parameters
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_data is None:
            self.required_data = ['Open', 'High', 'Low', 'Close', 'Volume']
        if self.parameters is None:
            self.parameters = {}
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a strategy-specific parameter with optional default."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a strategy-specific parameter."""
        self.parameters[key] = value


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides common functionality for signal generation, risk management,
    and performance tracking.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the base strategy.
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.signals_history = []
        self.performance_metrics = {}
        self.is_initialized = False
        
        # State tracking
        self.current_positions = {}  # symbol -> position size
        self.last_update = None
        
        logger.info(f"Initialized strategy: {self.config.name}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        This is the core method that each strategy must implement to define
        its trading logic and signal generation rules.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            timestamp: Current timestamp for signal generation
            
        Returns:
            List of Signal objects representing trading recommendations
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the provided data meets strategy requirements.
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid for this strategy, False otherwise
        """
        pass
    
    def initialize(self, historical_data: pd.DataFrame) -> bool:
        """
        Initialize the strategy with historical data.
        
        This method should be called before generating signals to allow
        the strategy to set up any required state or calculations.
        
        Args:
            historical_data: Historical market data for initialization
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self.validate_data(historical_data):
                logger.error(f"Data validation failed for strategy {self.config.name}")
                return False
            
            # Perform strategy-specific initialization
            self._initialize_strategy(historical_data)
            
            self.is_initialized = True
            logger.info(f"Successfully initialized strategy: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy {self.config.name}: {e}")
            return False
    
    def update(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """
        Update the strategy with new data and generate signals.
        
        Args:
            data: Updated market data
            timestamp: Current timestamp
            
        Returns:
            List of new signals generated
        """
        if not self.is_initialized:
            logger.warning(f"Strategy {self.config.name} not initialized")
            return []
        
        try:
            # Generate new signals
            signals = self.generate_signals(data, timestamp)
            
            # Store signals in history
            self.signals_history.extend(signals)
            self.last_update = timestamp
            
            # Apply risk management filters
            filtered_signals = self._apply_risk_filters(signals)
            
            # Update position tracking
            self._update_positions(filtered_signals)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error updating strategy {self.config.name}: {e}")
            return []
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current position sizes for all symbols."""
        return self.current_positions.copy()
    
    def get_signals_history(self, limit: Optional[int] = None) -> List[Signal]:
        """
        Get historical signals generated by this strategy.
        
        Args:
            limit: Maximum number of signals to return (most recent first)
            
        Returns:
            List of historical signals
        """
        if limit is None:
            return self.signals_history.copy()
        return self.signals_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the strategy."""
        return self.performance_metrics.copy()
    
    def reset(self) -> None:
        """Reset strategy state and clear history."""
        self.signals_history.clear()
        self.current_positions.clear()
        self.performance_metrics.clear()
        self.is_initialized = False
        self.last_update = None
        
        logger.info(f"Reset strategy: {self.config.name}")
    
    def _initialize_strategy(self, historical_data: pd.DataFrame) -> None:
        """
        Strategy-specific initialization logic.
        
        Subclasses can override this method to perform custom initialization
        such as calculating initial indicators or setting up state variables.
        
        Args:
            historical_data: Historical data for initialization
        """
        pass
    
    def _apply_risk_filters(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply risk management filters to signals.
        
        Args:
            signals: Raw signals from strategy
            
        Returns:
            Filtered signals after applying risk controls
        """
        filtered_signals = []
        
        for signal in signals:
            # Check position size limits
            if self._check_position_size_limit(signal):
                # Check drawdown limits
                if self._check_drawdown_limit(signal):
                    filtered_signals.append(signal)
                else:
                    logger.warning(f"Signal filtered due to drawdown limit: {signal}")
            else:
                logger.warning(f"Signal filtered due to position size limit: {signal}")
        
        return filtered_signals
    
    def _check_position_size_limit(self, signal: Signal) -> bool:
        """Check if signal respects position size limits."""
        current_position = self.current_positions.get(signal.symbol, 0.0)
        new_position = current_position + signal.position_size
        
        return abs(new_position) <= self.config.max_position_size
    
    def _check_drawdown_limit(self, signal: Signal) -> bool:
        """Check if signal respects maximum drawdown limits."""
        # Placeholder implementation - would calculate actual drawdown
        return True
    
    def _update_positions(self, signals: List[Signal]) -> None:
        """Update position tracking based on executed signals."""
        for signal in signals:
            current_position = self.current_positions.get(signal.symbol, 0.0)
            self.current_positions[signal.symbol] = current_position + signal.position_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.config.name} (v{self.config.version})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return (f"BaseStrategy(name='{self.config.name}', "
                f"version='{self.config.version}', "
                f"initialized={self.is_initialized})")