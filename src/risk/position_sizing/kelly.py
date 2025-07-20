"""
Kelly Criterion Position Sizing

Implements the Kelly Criterion for optimal position sizing based on expected returns,
win rates, and risk characteristics. The Kelly formula maximizes long-term growth
while managing the risk of ruin.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion position sizing implementation.
    
    The Kelly Criterion determines the optimal fraction of capital to allocate
    to a trade based on the expected return and probability of success.
    
    Kelly Formula: f* = (bp - q) / b
    Where:
    - f* = optimal fraction of capital to bet
    - b = odds received on the wager (decimal odds - 1)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, max_position_fraction: float = 0.25, min_kelly_threshold: float = 0.01):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            max_position_fraction: Maximum fraction of capital for any single position
            min_kelly_threshold: Minimum Kelly fraction to generate a position
        """
        self.max_position_fraction = max_position_fraction
        self.min_kelly_threshold = min_kelly_threshold
        
    def calculate_kelly_fraction(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        trading_cost: float = 0.0
    ) -> float:
        """
        Calculate the Kelly fraction for a trading opportunity.
        
        Args:
            win_probability: Probability of a winning trade (0-1)
            avg_win: Average return of winning trades (as decimal, e.g., 0.05 for 5%)
            avg_loss: Average loss of losing trades (as positive decimal, e.g., 0.03 for 3% loss)
            trading_cost: Transaction costs as fraction of position size
            
        Returns:
            Optimal Kelly fraction (0-1)
        """
        if not 0 <= win_probability <= 1:
            raise ValueError("Win probability must be between 0 and 1")
        
        if avg_win <= 0:
            raise ValueError("Average win must be positive")
            
        if avg_loss <= 0:
            raise ValueError("Average loss must be positive")
        
        # Adjust returns for trading costs
        net_avg_win = avg_win - trading_cost
        net_avg_loss = avg_loss + trading_cost
        
        # Calculate Kelly fraction
        loss_probability = 1 - win_probability
        
        if net_avg_win <= 0:
            return 0.0  # No positive edge after costs
        
        # Kelly formula: f = (bp - q) / b
        # Where b = net_avg_win / net_avg_loss (reward-to-risk ratio)
        reward_to_risk = net_avg_win / net_avg_loss
        kelly_fraction = (reward_to_risk * win_probability - loss_probability) / reward_to_risk
        
        # Ensure fraction is non-negative and below maximum
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_position_fraction))
        
        # Apply minimum threshold
        if kelly_fraction < self.min_kelly_threshold:
            kelly_fraction = 0.0
            
        return kelly_fraction
    
    def calculate_kelly_from_strategy_results(
        self,
        returns: pd.Series,
        lookback_window: int = 100
    ) -> float:
        """
        Calculate Kelly fraction from historical strategy returns.
        
        Args:
            returns: Series of historical returns (as decimals)
            lookback_window: Number of recent periods to consider
            
        Returns:
            Kelly fraction based on historical performance
        """
        if len(returns) < 10:
            logger.warning("Insufficient data for Kelly calculation")
            return 0.0
        
        # Use most recent data
        recent_returns = returns.tail(lookback_window)
        
        # Separate wins and losses
        wins = recent_returns[recent_returns > 0]
        losses = recent_returns[recent_returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            logger.warning("No wins or losses in data - cannot calculate Kelly")
            return 0.0
        
        # Calculate statistics
        win_probability = len(wins) / len(recent_returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())  # Convert to positive
        
        return self.calculate_kelly_fraction(win_probability, avg_win, avg_loss)
    
    def fractional_kelly(self, kelly_fraction: float, fraction: float = 0.25) -> float:
        """
        Apply fractional Kelly to reduce position size.
        
        Fractional Kelly reduces the calculated Kelly fraction to decrease
        volatility and drawdowns at the cost of some growth.
        
        Args:
            kelly_fraction: Full Kelly fraction
            fraction: Fraction of Kelly to use (e.g., 0.25 for quarter Kelly)
            
        Returns:
            Fractional Kelly position size
        """
        return kelly_fraction * fraction
    
    def kelly_with_drawdown_constraint(
        self,
        kelly_fraction: float,
        max_drawdown: float,
        win_probability: float,
        avg_loss: float
    ) -> float:
        """
        Adjust Kelly fraction to limit maximum drawdown.
        
        Args:
            kelly_fraction: Calculated Kelly fraction
            max_drawdown: Maximum acceptable drawdown (as decimal)
            win_probability: Probability of winning
            avg_loss: Average loss per trade
            
        Returns:
            Adjusted Kelly fraction to meet drawdown constraint
        """
        # Estimate maximum consecutive losses (simplified approach)
        # Using geometric distribution: P(X >= k) = (1-p)^k
        # Find k such that P(X >= k) = 0.01 (99% confidence)
        max_consecutive_losses = int(np.log(0.01) / np.log(1 - win_probability))
        
        # Calculate maximum position size to limit drawdown
        max_position_for_drawdown = max_drawdown / (max_consecutive_losses * avg_loss)
        
        # Return minimum of Kelly fraction and drawdown-constrained size
        return min(kelly_fraction, max_position_for_drawdown)
    
    def optimal_f(
        self,
        returns: pd.Series,
        num_iterations: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate Optimal F using Ralph Vince's method.
        
        Optimal F maximizes the geometric mean of returns by finding
        the optimal fraction of largest loss to risk.
        
        Args:
            returns: Series of trade returns (as decimals)
            num_iterations: Number of optimization iterations
            
        Returns:
            Tuple of (optimal_f, geometric_mean)
        """
        if len(returns) < 10:
            return 0.0, 0.0
        
        largest_loss = abs(returns.min())
        
        if largest_loss == 0:
            return 0.0, 0.0
        
        def geometric_mean(f: float) -> float:
            """Calculate negative geometric mean for minimization."""
            if f <= 0:
                return float('inf')
            
            # Calculate portfolio values
            portfolio_values = []
            for ret in returns:
                # Position size based on fraction of largest loss
                position_size = f
                # Portfolio return
                portfolio_return = position_size * ret / largest_loss
                portfolio_values.append(1 + portfolio_return)
            
            # Geometric mean
            portfolio_series = pd.Series(portfolio_values)
            if (portfolio_series <= 0).any():
                return float('inf')  # Bankruptcy occurred
            
            geo_mean = portfolio_series.prod() ** (1 / len(portfolio_series))
            return -geo_mean  # Negative for minimization
        
        # Optimize between 0 and 1
        result = minimize_scalar(
            geometric_mean,
            bounds=(0.001, 1.0),
            method='bounded'
        )
        
        optimal_f = result.x if result.success else 0.0
        max_geo_mean = -result.fun if result.success else 0.0
        
        return optimal_f, max_geo_mean
    
    def position_size_from_signal(
        self,
        signal_confidence: float,
        historical_returns: pd.Series,
        portfolio_value: float,
        price_per_share: float,
        trading_cost_rate: float = 0.001
    ) -> int:
        """
        Calculate actual position size (number of shares) from a trading signal.
        
        Args:
            signal_confidence: Confidence level of the trading signal (0-1)
            historical_returns: Historical returns for Kelly calculation
            portfolio_value: Current portfolio value
            price_per_share: Current price per share
            trading_cost_rate: Trading cost as fraction of position value
            
        Returns:
            Number of shares to trade
        """
        # Calculate Kelly fraction from historical data
        kelly_fraction = self.calculate_kelly_from_strategy_results(historical_returns)
        
        if kelly_fraction == 0:
            return 0
        
        # Adjust Kelly fraction by signal confidence
        adjusted_fraction = kelly_fraction * signal_confidence
        
        # Apply fractional Kelly for safety (quarter Kelly)
        final_fraction = self.fractional_kelly(adjusted_fraction, 0.25)
        
        # Calculate dollar amount to invest
        dollar_amount = portfolio_value * final_fraction
        
        # Account for trading costs
        net_dollar_amount = dollar_amount / (1 + trading_cost_rate)
        
        # Calculate number of shares
        shares = int(net_dollar_amount / price_per_share)
        
        return shares
    
    def get_kelly_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Get comprehensive Kelly statistics for a return series.
        
        Args:
            returns: Series of trading returns
            
        Returns:
            Dictionary with Kelly statistics
        """
        if len(returns) < 5:
            return {}
        
        kelly_fraction = self.calculate_kelly_from_strategy_results(returns)
        optimal_f, geo_mean = self.optimal_f(returns)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        stats = {
            'kelly_fraction': kelly_fraction,
            'optimal_f': optimal_f,
            'geometric_mean': geo_mean,
            'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': abs(losses.mean()) if len(losses) > 0 else 0,
            'profit_factor': abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf'),
            'expectancy': returns.mean(),
            'fractional_kelly_25': self.fractional_kelly(kelly_fraction, 0.25),
            'fractional_kelly_50': self.fractional_kelly(kelly_fraction, 0.50)
        }
        
        return stats