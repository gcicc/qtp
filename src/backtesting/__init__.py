"""
Backtesting Framework

Comprehensive backtesting system for testing trading strategies against historical data
with proper statistical analysis and performance attribution.

Key Features:
- Historical simulation with walk-forward analysis
- Transaction cost modeling and slippage simulation
- Multiple performance metrics and risk analytics
- Monte Carlo simulation for robustness testing
- Bias-free backtesting with proper data handling

Components:
- BacktestEngine: Main backtesting orchestration
- PerformanceAnalyzer: Comprehensive performance metrics calculation
- RiskAnalyzer: Risk metrics and drawdown analysis
- TransactionCostModel: Realistic transaction cost simulation
"""

from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer, RiskAnalyzer  
from .metrics import BacktestMetrics
from .transaction_costs import TransactionCostModel

__all__ = [
    "BacktestEngine",
    "PerformanceAnalyzer", 
    "RiskAnalyzer",
    "BacktestMetrics",
    "TransactionCostModel"
]