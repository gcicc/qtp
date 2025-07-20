"""
Risk Management Module

Comprehensive risk management system for portfolio construction, position sizing,
and real-time risk monitoring. Implements industry-standard risk metrics and
controls to protect capital and optimize risk-adjusted returns.

Key Components:
- Position Sizing: Kelly Criterion, Risk Parity, and custom algorithms
- Portfolio Metrics: VaR, CVaR, Sharpe ratio, maximum drawdown calculations
- Risk Monitoring: Real-time risk assessment and alert systems
- Risk Controls: Automated position limits and stop-loss mechanisms

Submodules:
- position_sizing: Algorithms for determining optimal position sizes
- portfolio_metrics: Risk and performance metric calculations  
- monitoring: Real-time risk monitoring and alerting
"""

from .position_sizing import *
from .portfolio_metrics import *
from .monitoring import *

__all__ = [
    "KellyCriterion",
    "RiskParityOptimizer", 
    "VaRCalculator",
    "PerformanceMetrics",
    "RiskMonitor",
    "PortfolioRiskManager"
]