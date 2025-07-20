"""
Quantitative Trading Platform (QTP)

A comprehensive Python-based quantitative trading platform that emphasizes transparency,
education, and rational decision-making through AI-powered agents.

Core Features:
- Contextual Intelligence: Detailed explanations for every trading recommendation
- Educational Focus: Built-in tutorials and real-time market explanations
- Transparent Methodology: Open-source algorithms with clear documentation
- Statistical Rigor: PhD-level statistical validation for all signals
- AI Agent System: Specialized agents for research, strategy development, risk assessment, and more

Modules:
- data: Market data ingestion, validation, and feature engineering
- strategies: Strategy base classes and built-in trading strategies
- risk: Position sizing, portfolio risk metrics, and real-time monitoring
- backtesting: Historical simulation and performance analytics
- execution: Broker integration and order management
- agents: AI agent system with specialized trading intelligence
- explanations: Signal rationale and market context generation
"""

__version__ = "0.1.0"
__author__ = "QTP Development Team"
__email__ = "contact@qtp.ai"

# Import main modules for convenient access
from . import data
from . import strategies
from . import risk
from . import backtesting
from . import execution
from . import agents
from . import explanations

__all__ = [
    "data",
    "strategies", 
    "risk",
    "backtesting",
    "execution",
    "agents",
    "explanations"
]