"""
Unit tests for trading signal structures.

Tests all signal-related structures including Signal, Position, Order,
and TradeExecution with comprehensive validation and functionality verification.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategies.signals import (
    Signal, Position, Order, TradeExecution,
    SignalType, SignalStrength, PositionSide, OrderType, OrderStatus
)


class TestSignal:
    """Test cases for Signal data structure."""
    
    def test_valid_signal_creation(self):
        """Test creation of valid trading signal."""
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.85,
            strength=SignalStrength.STRONG,
            strategy_name="MovingAverageCrossover",
            reasoning="RSI oversold + bullish divergence"
        )
        
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.85
        assert signal.strength == SignalStrength.STRONG
        assert signal.strategy_name == "MovingAverageCrossover"
    
    def test_signal_confidence_validation(self):
        """Test signal confidence validation."""
        # Valid confidence
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.75,  # Valid: 0-1
            strength=SignalStrength.STRONG,
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        assert signal.confidence == 0.75
        
        # Invalid confidence > 1
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                confidence=1.5,  # Invalid: > 1
                strength=SignalStrength.STRONG,
                strategy_name="TestStrategy",
                reasoning="Test"
            )
    
    def test_signal_target_price_validation(self):
        """Test signal target price validation."""
        # Valid buy signal - target > current
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            current_price=Decimal("150.00"),
            target_price=Decimal("155.00"),  # Valid: > current for buy
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        assert signal.target_price == Decimal("155.00")
        
        # Invalid buy signal - target <= current
        with pytest.raises(ValueError, match="Buy signal target price should be > current price"):
            Signal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                confidence=0.8,
                strength=SignalStrength.STRONG,
                current_price=Decimal("150.00"),
                target_price=Decimal("145.00"),  # Invalid: < current for buy
                strategy_name="TestStrategy",
                reasoning="Test"
            )
    
    def test_signal_stop_loss_validation(self):
        """Test signal stop loss validation."""
        # Valid buy signal - stop < current
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            current_price=Decimal("150.00"),
            stop_loss=Decimal("145.00"),  # Valid: < current for buy
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        assert signal.stop_loss == Decimal("145.00")
        
        # Invalid buy signal - stop >= current
        with pytest.raises(ValueError, match="Buy signal stop loss should be < current price"):
            Signal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                confidence=0.8,
                strength=SignalStrength.STRONG,
                current_price=Decimal("150.00"),
                stop_loss=Decimal("155.00"),  # Invalid: > current for buy
                strategy_name="TestStrategy",
                reasoning="Test"
            )
    
    def test_signal_expiration(self):
        """Test signal expiration functionality."""
        # Non-expired signal
        future_time = datetime.now() + timedelta(hours=1)
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            valid_until=future_time,
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        assert signal.is_expired() is False
        
        # Expired signal
        past_time = datetime.now() - timedelta(hours=1)
        expired_signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            valid_until=past_time,
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        assert expired_signal.is_expired() is True
    
    def test_signal_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            current_price=Decimal("150.00"),
            target_price=Decimal("160.00"),
            stop_loss=Decimal("145.00"),
            strategy_name="TestStrategy",
            reasoning="Test"
        )
        
        # Risk = current - stop = 150 - 145 = 5
        # Reward = target - current = 160 - 150 = 10
        # Risk/Reward = 5/10 = 0.5
        ratio = signal.risk_reward_ratio()
        assert ratio == 0.5


class TestPosition:
    """Test cases for Position data structure."""
    
    def test_valid_long_position_creation(self):
        """Test creation of valid long position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            entry_timestamp=datetime.now()
        )
        
        assert position.symbol == "AAPL"
        assert position.side == PositionSide.LONG
        assert position.quantity == Decimal("100")
        assert position.avg_entry_price == Decimal("150.00")
    
    def test_valid_short_position_creation(self):
        """Test creation of valid short position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            quantity=Decimal("-100"),  # Negative for short
            avg_entry_price=Decimal("150.00"),
            entry_timestamp=datetime.now()
        )
        
        assert position.side == PositionSide.SHORT
        assert position.quantity == Decimal("-100")
    
    def test_position_quantity_side_validation(self):
        """Test position quantity and side consistency validation."""
        # Invalid: Long position with negative quantity
        with pytest.raises(ValueError, match="Long positions must have positive quantity"):
            Position(
                symbol="AAPL",
                side=PositionSide.LONG,
                quantity=Decimal("-100"),  # Invalid: negative for long
                avg_entry_price=Decimal("150.00"),
                entry_timestamp=datetime.now()
            )
        
        # Invalid: Short position with positive quantity
        with pytest.raises(ValueError, match="Short positions must have negative quantity"):
            Position(
                symbol="AAPL",
                side=PositionSide.SHORT,
                quantity=Decimal("100"),  # Invalid: positive for short
                avg_entry_price=Decimal("150.00"),
                entry_timestamp=datetime.now()
            )
    
    def test_position_pnl_calculations(self):
        """Test position P&L calculations."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            entry_timestamp=datetime.now()
        )
        
        # Unrealized P&L = (current - entry) * quantity = (155 - 150) * 100 = 500
        unrealized_pnl = position.unrealized_pnl()
        assert unrealized_pnl == Decimal("500")
        
        # Market value = quantity * current_price = 100 * 155 = 15500
        market_value = position.market_value()
        assert market_value == Decimal("15500")
        
        # Cost basis = quantity * entry_price = 100 * 150 = 15000
        cost_basis = position.cost_basis()
        assert cost_basis == Decimal("15000")
        
        # Return percentage = (155 - 150) / 150 = 0.0333... ≈ 3.33%
        return_pct = position.return_pct()
        assert abs(return_pct - 0.03333333333333333) < 1e-10
    
    def test_position_profitability(self):
        """Test position profitability detection."""
        # Profitable long position
        profitable_position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),  # Higher than entry
            entry_timestamp=datetime.now()
        )
        assert profitable_position.is_profitable() is True
        
        # Unprofitable long position
        unprofitable_position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("145.00"),  # Lower than entry
            entry_timestamp=datetime.now()
        )
        assert unprofitable_position.is_profitable() is False


class TestOrder:
    """Test cases for Order data structure."""
    
    def test_valid_order_creation(self):
        """Test creation of valid order."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=100,
            price=Decimal("150.00")
        )
        
        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.LIMIT
        assert order.side == "buy"
        assert order.quantity == 100
        assert order.price == Decimal("150.00")
        assert order.status == OrderStatus.PENDING
    
    def test_order_side_validation(self):
        """Test order side validation."""
        # Valid sides
        buy_order = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=100
        )
        assert buy_order.side == "buy"
        
        sell_order = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="sell",
            quantity=100
        )
        assert sell_order.side == "sell"
        
        # Invalid side
        with pytest.raises(ValueError):
            Order(
                symbol="AAPL",
                order_type=OrderType.MARKET,
                side="invalid",
                quantity=100
            )
    
    def test_order_filled_quantity_validation(self):
        """Test filled quantity validation."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=100,
            price=Decimal("150.00")
        )
        
        # Valid filled quantity
        order.filled_quantity = 50
        assert order.filled_quantity == 50
        
        # Invalid: filled > total quantity
        with pytest.raises(ValueError, match="Filled quantity cannot exceed total quantity"):
            order.filled_quantity = 150
    
    def test_order_calculations(self):
        """Test order calculation methods."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=100,
            price=Decimal("150.00"),
            filled_quantity=60
        )
        
        # Remaining quantity
        assert order.remaining_quantity() == 40
        
        # Fill percentage
        assert order.fill_percentage() == 60.0
        
        # Notional value
        assert order.notional_value() == Decimal("15000.00")
        
        # Fully filled check
        assert order.is_fully_filled() is False
        assert order.is_partially_filled() is True
        
        # Mark as fully filled
        order.filled_quantity = 100
        assert order.is_fully_filled() is True
        assert order.is_partially_filled() is False
    
    def test_order_active_status(self):
        """Test order active status detection."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=100,
            price=Decimal("150.00")
        )
        
        # Active statuses
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            order.status = status
            assert order.is_active() is True
        
        # Inactive statuses
        for status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            order.status = status
            assert order.is_active() is False


class TestTradeExecution:
    """Test cases for TradeExecution data structure."""
    
    def test_valid_execution_creation(self):
        """Test creation of valid trade execution."""
        execution = TradeExecution(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            price=Decimal("150.25"),
            timestamp=datetime.now()
        )
        
        assert execution.order_id == "ORD123"
        assert execution.symbol == "AAPL"
        assert execution.side == "buy"
        assert execution.quantity == 50
        assert execution.price == Decimal("150.25")
    
    def test_execution_calculations(self):
        """Test execution calculation methods."""
        execution = TradeExecution(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            price=Decimal("150.25"),
            timestamp=datetime.now(),
            commission=Decimal("5.00")
        )
        
        # Notional value
        expected_notional = Decimal("150.25") * Decimal("50")
        assert execution.notional_value() == expected_notional
        
        # Total cost (including commission)
        expected_total = expected_notional + Decimal("5.00")
        assert execution.total_cost() == expected_total