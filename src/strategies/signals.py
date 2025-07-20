"""
Trading signal and position structures for the QTP platform.

This module defines the core data structures for trading signals, positions, and orders.
All structures include comprehensive metadata for transparency and auditability.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class SignalType(str, Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"


class SignalStrength(str, Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class PositionSide(str, Enum):
    """Position side/direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(str, Enum):
    """Order types for execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Signal(BaseModel):
    """
    Trading signal with comprehensive metadata and reasoning.
    
    Represents a trading recommendation with confidence level, statistical evidence,
    and human-readable explanation for transparency.
    
    Example:
        >>> signal = Signal(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     signal_type=SignalType.BUY,
        ...     confidence=0.85,
        ...     strength=SignalStrength.STRONG,
        ...     target_price=Decimal("155.00"),
        ...     stop_loss=Decimal("148.00"),
        ...     reasoning="RSI oversold + bullish divergence + support at 150",
        ...     metadata={
        ...         "rsi": 28.5,
        ...         "macd_signal": "bullish_crossover",
        ...         "support_level": "150.00"
        ...     }
        ... )
    """
    # Core signal information
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    signal_type: SignalType = Field(..., description="Type of trading signal")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    strength: SignalStrength = Field(..., description="Signal strength category")
    
    # Price targets and risk management
    current_price: Optional[Decimal] = Field(None, gt=0, description="Current market price")
    target_price: Optional[Decimal] = Field(None, gt=0, description="Target price for signal")
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    
    # Position sizing and risk
    position_size: Optional[Decimal] = Field(None, ge=0, description="Recommended position size")
    risk_per_share: Optional[Decimal] = Field(None, ge=0, description="Risk per share")
    max_position_risk: Optional[float] = Field(None, ge=0, le=1, description="Max % of portfolio at risk")
    
    # Metadata and reasoning
    strategy_name: str = Field(..., description="Strategy that generated signal")
    reasoning: str = Field(..., description="Human-readable explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal data")
    statistical_significance: Optional[float] = Field(None, ge=0, le=1, description="Statistical significance")
    
    # Timing and lifecycle
    valid_until: Optional[datetime] = Field(None, description="Signal expiration time")
    priority: int = Field(default=1, ge=1, le=5, description="Signal priority (1-5)")
    signal_id: Optional[str] = Field(None, description="Unique signal identifier")
    
    @validator('target_price')
    def validate_target_price(cls, v, values):
        """Validate target price makes sense relative to signal type."""
        if v is None:
            return v
        
        signal_type = values.get('signal_type')
        current_price = values.get('current_price')
        
        if signal_type and current_price:
            if signal_type == SignalType.BUY and v <= current_price:
                raise ValueError('Buy signal target price should be > current price')
            elif signal_type == SignalType.SELL and v >= current_price:
                raise ValueError('Sell signal target price should be < current price')
        
        return v
    
    @validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        """Validate stop loss makes sense relative to signal type."""
        if v is None:
            return v
            
        signal_type = values.get('signal_type')
        current_price = values.get('current_price')
        
        if signal_type and current_price:
            if signal_type == SignalType.BUY and v >= current_price:
                raise ValueError('Buy signal stop loss should be < current price')
            elif signal_type == SignalType.SELL and v <= current_price:
                raise ValueError('Sell signal stop loss should be > current price')
        
        return v
    
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until
    
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if prices are available."""
        if not all([self.current_price, self.target_price, self.stop_loss]):
            return None
        
        if self.signal_type == SignalType.BUY:
            risk = float(self.current_price - self.stop_loss)
            reward = float(self.target_price - self.current_price)
        else:
            risk = float(self.stop_loss - self.current_price)
            reward = float(self.current_price - self.target_price)
        
        return risk / reward if reward > 0 else None


class Position(BaseModel):
    """
    Trading position data structure.
    
    Represents a current position in a security with P&L tracking,
    risk metrics, and position management data.
    
    Example:
        >>> position = Position(
        ...     symbol="AAPL",
        ...     side=PositionSide.LONG,
        ...     quantity=100,
        ...     avg_entry_price=Decimal("150.00"),
        ...     current_price=Decimal("152.50"),
        ...     entry_timestamp=datetime.now()
        ... )
        >>> print(f"P&L: ${position.unrealized_pnl()}")
    """
    # Position identification
    symbol: str = Field(..., description="Trading symbol")
    side: PositionSide = Field(..., description="Position side (long/short/flat)")
    quantity: Decimal = Field(..., description="Position size (positive for long, negative for short)")
    
    # Entry information
    avg_entry_price: Decimal = Field(..., gt=0, description="Average entry price")
    entry_timestamp: datetime = Field(..., description="Position entry timestamp")
    entry_signals: List[str] = Field(default_factory=list, description="Signal IDs that created position")
    
    # Current market data
    current_price: Optional[Decimal] = Field(None, gt=0, description="Current market price")
    last_update: datetime = Field(default_factory=datetime.now, description="Last price update")
    
    # Risk management
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    max_risk: Optional[Decimal] = Field(None, ge=0, description="Maximum risk amount")
    
    # P&L tracking
    realized_pnl: Decimal = Field(default=Decimal('0'), description="Realized P&L")
    commission_paid: Decimal = Field(default=Decimal('0'), ge=0, description="Total commission paid")
    
    # Position metadata
    strategy_name: Optional[str] = Field(None, description="Strategy that opened position")
    notes: str = Field(default="", description="Position notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional position data")
    
    @validator('quantity')
    def validate_quantity_side_consistency(cls, v, values):
        """Ensure quantity sign matches position side."""
        side = values.get('side')
        if side == PositionSide.LONG and v < 0:
            raise ValueError('Long positions must have positive quantity')
        elif side == PositionSide.SHORT and v > 0:
            raise ValueError('Short positions must have negative quantity')
        elif side == PositionSide.FLAT and v != 0:
            raise ValueError('Flat positions must have zero quantity')
        return v
    
    def unrealized_pnl(self) -> Optional[Decimal]:
        """Calculate unrealized P&L."""
        if self.current_price is None or self.quantity == 0:
            return None
        
        price_diff = self.current_price - self.avg_entry_price
        return self.quantity * price_diff
    
    def total_pnl(self) -> Optional[Decimal]:
        """Calculate total P&L (realized + unrealized)."""
        unrealized = self.unrealized_pnl()
        if unrealized is None:
            return self.realized_pnl
        return self.realized_pnl + unrealized
    
    def market_value(self) -> Optional[Decimal]:
        """Calculate current market value of position."""
        if self.current_price is None:
            return None
        return abs(self.quantity) * self.current_price
    
    def cost_basis(self) -> Decimal:
        """Calculate cost basis of position."""
        return abs(self.quantity) * self.avg_entry_price
    
    def return_pct(self) -> Optional[float]:
        """Calculate percentage return."""
        if self.current_price is None or self.avg_entry_price == 0:
            return None
        
        if self.side == PositionSide.LONG:
            return float((self.current_price - self.avg_entry_price) / self.avg_entry_price)
        elif self.side == PositionSide.SHORT:
            return float((self.avg_entry_price - self.current_price) / self.avg_entry_price)
        return 0.0
    
    def is_profitable(self) -> Optional[bool]:
        """Check if position is currently profitable."""
        pnl = self.unrealized_pnl()
        return pnl > 0 if pnl is not None else None


class Order(BaseModel):
    """
    Order data structure for trade execution.
    
    Represents an order to buy or sell securities with execution tracking,
    timing information, and order management data.
    
    Example:
        >>> order = Order(
        ...     symbol="AAPL",
        ...     order_type=OrderType.LIMIT,
        ...     side="buy",
        ...     quantity=100,
        ...     price=Decimal("150.00"),
        ...     strategy_name="MovingAverageCrossover"
        ... )
    """
    # Order identification
    order_id: Optional[str] = Field(None, description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    order_type: OrderType = Field(..., description="Order type")
    side: str = Field(..., pattern="^(buy|sell)$", description="Order side")
    
    # Order details
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Order price (for limit orders)")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price (for stop orders)")
    
    # Execution tracking
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: int = Field(default=0, ge=0, description="Quantity filled")
    avg_fill_price: Optional[Decimal] = Field(None, gt=0, description="Average fill price")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="Order creation time")
    submitted_at: Optional[datetime] = Field(None, description="Order submission time")
    filled_at: Optional[datetime] = Field(None, description="Order fill time")
    expires_at: Optional[datetime] = Field(None, description="Order expiration time")
    
    # Trade management
    time_in_force: str = Field(default="DAY", description="Time in force (DAY, GTC, IOC, FOK)")
    reduce_only: bool = Field(default=False, description="Reduce-only order flag")
    
    # Strategy and signal tracking
    strategy_name: Optional[str] = Field(None, description="Strategy that generated order")
    signal_id: Optional[str] = Field(None, description="Signal that triggered order")
    parent_order_id: Optional[str] = Field(None, description="Parent order (for bracket orders)")
    
    # Risk and validation
    max_commission: Optional[Decimal] = Field(None, ge=0, description="Maximum acceptable commission")
    notes: str = Field(default="", description="Order notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional order data")
    
    @validator('filled_quantity')
    def filled_quantity_not_exceed_total(cls, v, values):
        """Ensure filled quantity doesn't exceed total quantity."""
        quantity = values.get('quantity')
        if quantity and v > quantity:
            raise ValueError('Filled quantity cannot exceed total quantity')
        return v
    
    def remaining_quantity(self) -> int:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    def is_fully_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_quantity >= self.quantity
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity
    
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        return (self.filled_quantity / self.quantity) * 100.0
    
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value of order."""
        if self.price is None:
            return None
        return self.price * Decimal(str(self.quantity))
    
    def is_active(self) -> bool:
        """Check if order is in an active state."""
        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED, 
            OrderStatus.PARTIALLY_FILLED
        }
        return self.status in active_statuses


class TradeExecution(BaseModel):
    """
    Individual trade execution record.
    
    Represents a single fill of an order with execution details.
    
    Example:
        >>> execution = TradeExecution(
        ...     order_id="ORD123",
        ...     symbol="AAPL",
        ...     side="buy",
        ...     quantity=50,
        ...     price=Decimal("150.25"),
        ...     timestamp=datetime.now()
        ... )
    """
    execution_id: Optional[str] = Field(None, description="Unique execution identifier")
    order_id: str = Field(..., description="Related order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., pattern="^(buy|sell)$", description="Execution side")
    quantity: int = Field(..., gt=0, description="Executed quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    timestamp: datetime = Field(..., description="Execution timestamp")
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Commission paid")
    exchange: Optional[str] = Field(None, description="Execution exchange")
    venue: Optional[str] = Field(None, description="Execution venue")
    
    def notional_value(self) -> Decimal:
        """Calculate notional value of execution."""
        return self.price * Decimal(str(self.quantity))
    
    def total_cost(self) -> Decimal:
        """Calculate total cost including commission."""
        return self.notional_value() + self.commission