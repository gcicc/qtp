"""
Execution Engine

Trade execution system with broker integration, order management, and transaction
cost analysis. Provides intelligent order routing and execution algorithms.

Key Features:
- Multiple broker API integrations with unified interface
- Smart order routing and execution algorithms
- Real-time order management and tracking
- Transaction cost analysis and slippage modeling
- Paper trading simulation for strategy testing

Components:
- ExecutionEngine: Main trade execution orchestration
- BrokerInterface: Unified interface for different brokers
- OrderManager: Order lifecycle management and tracking
- ExecutionAlgorithms: TWAP, VWAP, and other execution strategies
"""

from .engine import ExecutionEngine
from .broker_interface import BrokerInterface
from .order_manager import OrderManager, Order, OrderStatus
from .algorithms import ExecutionAlgorithm, TWAPAlgorithm, VWAPAlgorithm

__all__ = [
    "ExecutionEngine",
    "BrokerInterface",
    "OrderManager",
    "Order", 
    "OrderStatus",
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm"
]