"""
Core market data structures for the QTP platform.

This module defines the fundamental data structures used throughout the system
for representing market data, including OHLCV bars, trades, quotes, and market events.
All structures use pydantic for validation and type safety.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator


class MarketEventType(str, Enum):
    """Types of market events that can occur."""
    TRADE = "trade"
    QUOTE = "quote" 
    NEWS = "news"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    HALT = "halt"
    RESUME = "resume"
    OPEN = "open"
    CLOSE = "close"


class OHLCV(BaseModel):
    """
    Open, High, Low, Close, Volume data structure.
    
    Represents price and volume data for a specific time period.
    All price values are stored as Decimal for precision.
    
    Example:
        >>> ohlcv = OHLCV(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     open=Decimal("150.25"),
        ...     high=Decimal("152.50"),
        ...     low=Decimal("149.75"),
        ...     close=Decimal("151.80"),
        ...     volume=1000000,
        ...     timeframe="1m"
        ... )
        >>> print(f"AAPL closed at ${ohlcv.close}")
    """
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL', 'BTC-USD')")
    timestamp: datetime = Field(..., description="Timestamp for the bar")
    open: Decimal = Field(..., gt=0, description="Opening price")
    high: Decimal = Field(..., gt=0, description="Highest price during period")
    low: Decimal = Field(..., gt=0, description="Lowest price during period") 
    close: Decimal = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Volume traded during period")
    timeframe: str = Field(..., description="Time period (e.g., '1m', '5m', '1h', '1d')")
    adjusted_close: Optional[Decimal] = Field(None, gt=0, description="Dividend/split adjusted close")
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        """Validate that high is the highest price."""
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= open price')
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= low price')
        if 'close' in values and v < values['close']:
            raise ValueError('High must be >= close price')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        """Validate that low is the lowest price."""
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= open price')
        if 'close' in values and v > values['close']:
            raise ValueError('Low must be <= close price')
        return v
    
    def is_bullish(self) -> bool:
        """Return True if close > open (bullish candle)."""
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        """Return True if close < open (bearish candle)."""
        return self.close < self.open
    
    def body_size(self) -> Decimal:
        """Return the size of the candle body (abs(close - open))."""
        return abs(self.close - self.open)
    
    def upper_shadow(self) -> Decimal:
        """Return the size of the upper shadow."""
        return self.high - max(self.open, self.close)
    
    def lower_shadow(self) -> Decimal:
        """Return the size of the lower shadow."""
        return min(self.open, self.close) - self.low


class Trade(BaseModel):
    """
    Individual trade data structure.
    
    Represents a single executed trade with price, size, and metadata.
    
    Example:
        >>> trade = Trade(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     price=Decimal("151.50"),
        ...     size=100,
        ...     side="buy"
        ... )
        >>> print(f"Trade: {trade.size} shares at ${trade.price}")
    """
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Trade execution timestamp")
    price: Decimal = Field(..., gt=0, description="Trade execution price")
    size: int = Field(..., gt=0, description="Number of shares/units traded")
    side: str = Field(..., pattern="^(buy|sell)$", description="Trade side: 'buy' or 'sell'")
    trade_id: Optional[str] = Field(None, description="Unique trade identifier")
    exchange: Optional[str] = Field(None, description="Exchange where trade occurred")
    conditions: Optional[str] = Field(None, description="Special trade conditions")
    
    def notional_value(self) -> Decimal:
        """Calculate the notional value of the trade."""
        return self.price * Decimal(str(self.size))


class Quote(BaseModel):
    """
    Bid/Ask quote data structure.
    
    Represents the best bid and ask prices with sizes at a point in time.
    
    Example:
        >>> quote = Quote(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     bid_price=Decimal("151.45"),
        ...     bid_size=500,
        ...     ask_price=Decimal("151.50"),
        ...     ask_size=300
        ... )
        >>> print(f"Spread: ${quote.spread()}")
    """
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Quote timestamp")
    bid_price: Decimal = Field(..., gt=0, description="Best bid price")
    bid_size: int = Field(..., gt=0, description="Size at best bid")
    ask_price: Decimal = Field(..., gt=0, description="Best ask price")
    ask_size: int = Field(..., gt=0, description="Size at best ask")
    exchange: Optional[str] = Field(None, description="Exchange providing quote")
    
    @validator('ask_price')
    def ask_must_be_greater_than_bid(cls, v, values):
        """Validate that ask price >= bid price."""
        if 'bid_price' in values and v < values['bid_price']:
            raise ValueError('Ask price must be >= bid price')
        return v
    
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread."""
        return self.ask_price - self.bid_price
    
    def mid_price(self) -> Decimal:
        """Calculate the mid price between bid and ask."""
        return (self.bid_price + self.ask_price) / Decimal('2')
    
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points relative to mid price."""
        mid = self.mid_price()
        return (self.spread() / mid) * Decimal('10000')


class MarketEvent(BaseModel):
    """
    General market event data structure.
    
    Represents various market events like news, earnings, dividends, etc.
    
    Example:
        >>> event = MarketEvent(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     event_type=MarketEventType.EARNINGS,
        ...     description="Q1 2024 Earnings Release",
        ...     data={"eps": "2.45", "revenue": "119.9B"}
        ... )
    """
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: MarketEventType = Field(..., description="Type of market event")
    description: str = Field(..., description="Human-readable event description")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    source: Optional[str] = Field(None, description="Data source")
    impact_score: Optional[float] = Field(None, ge=0, le=1, description="Expected market impact (0-1)")
    
    def is_price_sensitive(self) -> bool:
        """Determine if event is likely to be price sensitive."""
        price_sensitive_events = {
            MarketEventType.EARNINGS,
            MarketEventType.NEWS,
            MarketEventType.DIVIDEND,
            MarketEventType.SPLIT,
            MarketEventType.HALT
        }
        return self.event_type in price_sensitive_events


class MarketDepth(BaseModel):
    """
    Market depth (Level 2) data structure.
    
    Represents order book depth with multiple bid/ask levels.
    
    Example:
        >>> depth = MarketDepth(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     bids=[
        ...         {"price": Decimal("151.45"), "size": 500},
        ...         {"price": Decimal("151.40"), "size": 1000}
        ...     ],
        ...     asks=[
        ...         {"price": Decimal("151.50"), "size": 300},
        ...         {"price": Decimal("151.55"), "size": 800}
        ...     ]
        ... )
    """
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Depth snapshot timestamp")
    bids: list[Dict[str, Union[Decimal, int]]] = Field(..., description="Bid levels (price, size)")
    asks: list[Dict[str, Union[Decimal, int]]] = Field(..., description="Ask levels (price, size)")
    
    def best_bid(self) -> Optional[Decimal]:
        """Get the best (highest) bid price."""
        if not self.bids:
            return None
        return max(level["price"] for level in self.bids)
    
    def best_ask(self) -> Optional[Decimal]:
        """Get the best (lowest) ask price."""
        if not self.asks:
            return None
        return min(level["price"] for level in self.asks)
    
    def total_bid_size(self) -> int:
        """Calculate total size across all bid levels."""
        return sum(level["size"] for level in self.bids)
    
    def total_ask_size(self) -> int:
        """Calculate total size across all ask levels."""
        return sum(level["size"] for level in self.asks)


class MarketStatus(BaseModel):
    """
    Market status information.
    
    Represents the current status of a market or trading session.
    
    Example:
        >>> status = MarketStatus(
        ...     market="NYSE",
        ...     timestamp=datetime.now(),
        ...     is_open=True,
        ...     session_type="regular",
        ...     next_open=datetime(2024, 1, 2, 9, 30),
        ...     next_close=datetime(2024, 1, 1, 16, 0)
        ... )
    """
    market: str = Field(..., description="Market identifier (e.g., 'NYSE', 'NASDAQ')")
    timestamp: datetime = Field(..., description="Status timestamp")
    is_open: bool = Field(..., description="Whether market is currently open")
    session_type: str = Field(..., description="Type of session (regular, pre, post, closed)")
    next_open: Optional[datetime] = Field(None, description="Next market open time")
    next_close: Optional[datetime] = Field(None, description="Next market close time")
    timezone: str = Field(default="UTC", description="Market timezone")
    holidays: list[datetime] = Field(default_factory=list, description="Upcoming market holidays")