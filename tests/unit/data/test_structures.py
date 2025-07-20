"""
Unit tests for market data structures.

Tests all market data structures including OHLCV, Trade, Quote, and MarketEvent
with comprehensive validation and functionality verification.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.data.structures import (
    OHLCV, Trade, Quote, MarketEvent, MarketDepth, MarketStatus,
    MarketEventType
)


class TestOHLCV:
    """Test cases for OHLCV data structure."""
    
    def test_valid_ohlcv_creation(self):
        """Test creation of valid OHLCV data."""
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=Decimal("150.00"),
            high=Decimal("152.50"),
            low=Decimal("149.00"),
            close=Decimal("151.80"),
            volume=1000000,
            timeframe="1m"
        )
        
        assert ohlcv.symbol == "AAPL"
        assert ohlcv.open == Decimal("150.00")
        assert ohlcv.high == Decimal("152.50")
        assert ohlcv.low == Decimal("149.00")
        assert ohlcv.close == Decimal("151.80")
        assert ohlcv.volume == 1000000
        assert ohlcv.timeframe == "1m"
    
    def test_ohlcv_validation_high_lower_than_low(self):
        """Test validation when high is lower than low."""
        with pytest.raises(ValueError, match="High must be >= low price"):
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=Decimal("150.00"),
                high=Decimal("148.00"),  # High lower than low
                low=Decimal("149.00"),
                close=Decimal("151.80"),
                volume=1000000,
                timeframe="1m"
            )
    
    def test_ohlcv_validation_low_higher_than_close(self):
        """Test validation when low is higher than close."""
        with pytest.raises(ValueError, match="Low must be <= close price"):
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=Decimal("150.00"),
                high=Decimal("152.50"),
                low=Decimal("152.00"),  # Low higher than close
                close=Decimal("151.80"),
                volume=1000000,
                timeframe="1m"
            )
    
    def test_ohlcv_bullish_bearish_detection(self):
        """Test bullish and bearish candle detection."""
        # Bullish candle
        bullish_ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=Decimal("150.00"),
            high=Decimal("152.50"),
            low=Decimal("149.00"),
            close=Decimal("151.80"),  # Close > Open
            volume=1000000,
            timeframe="1m"
        )
        
        assert bullish_ohlcv.is_bullish() is True
        assert bullish_ohlcv.is_bearish() is False
        
        # Bearish candle
        bearish_ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=Decimal("151.80"),
            high=Decimal("152.50"),
            low=Decimal("149.00"),
            close=Decimal("150.00"),  # Close < Open
            volume=1000000,
            timeframe="1m"
        )
        
        assert bearish_ohlcv.is_bullish() is False
        assert bearish_ohlcv.is_bearish() is True
    
    def test_ohlcv_calculations(self):
        """Test OHLCV calculation methods."""
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=Decimal("150.00"),
            high=Decimal("152.50"),
            low=Decimal("149.00"),
            close=Decimal("151.80"),
            volume=1000000,
            timeframe="1m"
        )
        
        # Body size
        expected_body = abs(Decimal("151.80") - Decimal("150.00"))
        assert ohlcv.body_size() == expected_body
        
        # Upper shadow
        expected_upper = Decimal("152.50") - max(Decimal("150.00"), Decimal("151.80"))
        assert ohlcv.upper_shadow() == expected_upper
        
        # Lower shadow
        expected_lower = min(Decimal("150.00"), Decimal("151.80")) - Decimal("149.00")
        assert ohlcv.lower_shadow() == expected_lower


class TestTrade:
    """Test cases for Trade data structure."""
    
    def test_valid_trade_creation(self):
        """Test creation of valid trade data."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=Decimal("151.50"),
            size=100,
            side="buy"
        )
        
        assert trade.symbol == "AAPL"
        assert trade.price == Decimal("151.50")
        assert trade.size == 100
        assert trade.side == "buy"
    
    def test_trade_invalid_side(self):
        """Test validation of invalid trade side."""
        with pytest.raises(ValueError):
            Trade(
                symbol="AAPL",
                timestamp=datetime.now(),
                price=Decimal("151.50"),
                size=100,
                side="invalid_side"
            )
    
    def test_trade_calculations(self):
        """Test trade calculation methods."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=Decimal("151.50"),
            size=100,
            side="buy"
        )
        
        expected_notional = Decimal("151.50") * Decimal("100")
        assert trade.notional_value() == expected_notional


class TestQuote:
    """Test cases for Quote data structure."""
    
    def test_valid_quote_creation(self):
        """Test creation of valid quote data."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=Decimal("151.45"),
            bid_size=500,
            ask_price=Decimal("151.50"),
            ask_size=300
        )
        
        assert quote.symbol == "AAPL"
        assert quote.bid_price == Decimal("151.45")
        assert quote.ask_price == Decimal("151.50")
    
    def test_quote_ask_lower_than_bid(self):
        """Test validation when ask price is lower than bid."""
        with pytest.raises(ValueError, match="Ask price must be >= bid price"):
            Quote(
                symbol="AAPL",
                timestamp=datetime.now(),
                bid_price=Decimal("151.50"),
                bid_size=500,
                ask_price=Decimal("151.45"),  # Ask lower than bid
                ask_size=300
            )
    
    def test_quote_calculations(self):
        """Test quote calculation methods."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=Decimal("151.45"),
            bid_size=500,
            ask_price=Decimal("151.55"),
            ask_size=300
        )
        
        # Spread
        expected_spread = Decimal("151.55") - Decimal("151.45")
        assert quote.spread() == expected_spread
        
        # Mid price
        expected_mid = (Decimal("151.45") + Decimal("151.55")) / Decimal("2")
        assert quote.mid_price() == expected_mid
        
        # Spread in basis points
        spread_bps = quote.spread_bps()
        assert isinstance(spread_bps, Decimal)
        assert spread_bps > 0


class TestMarketEvent:
    """Test cases for MarketEvent data structure."""
    
    def test_valid_market_event_creation(self):
        """Test creation of valid market event."""
        event = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            event_type=MarketEventType.EARNINGS,
            description="Q1 2024 Earnings Release",
            data={"eps": "2.45", "revenue": "119.9B"}
        )
        
        assert event.symbol == "AAPL"
        assert event.event_type == MarketEventType.EARNINGS
        assert event.description == "Q1 2024 Earnings Release"
        assert event.data["eps"] == "2.45"
    
    def test_market_event_price_sensitivity(self):
        """Test price sensitivity detection."""
        # Price sensitive event
        earnings_event = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            event_type=MarketEventType.EARNINGS,
            description="Earnings release"
        )
        assert earnings_event.is_price_sensitive() is True
        
        # Non-price sensitive event
        trade_event = MarketEvent(
            symbol="AAPL",
            timestamp=datetime.now(),
            event_type=MarketEventType.TRADE,
            description="Regular trade"
        )
        assert trade_event.is_price_sensitive() is False


class TestMarketDepth:
    """Test cases for MarketDepth data structure."""
    
    def test_valid_market_depth_creation(self):
        """Test creation of valid market depth."""
        depth = MarketDepth(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[
                {"price": Decimal("151.45"), "size": 500},
                {"price": Decimal("151.40"), "size": 1000}
            ],
            asks=[
                {"price": Decimal("151.50"), "size": 300},
                {"price": Decimal("151.55"), "size": 800}
            ]
        )
        
        assert depth.symbol == "AAPL"
        assert len(depth.bids) == 2
        assert len(depth.asks) == 2
    
    def test_market_depth_calculations(self):
        """Test market depth calculation methods."""
        depth = MarketDepth(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[
                {"price": Decimal("151.45"), "size": 500},
                {"price": Decimal("151.40"), "size": 1000}
            ],
            asks=[
                {"price": Decimal("151.50"), "size": 300},
                {"price": Decimal("151.55"), "size": 800}
            ]
        )
        
        # Best bid/ask
        assert depth.best_bid() == Decimal("151.45")
        assert depth.best_ask() == Decimal("151.50")
        
        # Total sizes
        assert depth.total_bid_size() == 1500
        assert depth.total_ask_size() == 1100


class TestMarketStatus:
    """Test cases for MarketStatus data structure."""
    
    def test_valid_market_status_creation(self):
        """Test creation of valid market status."""
        status = MarketStatus(
            market="NYSE",
            timestamp=datetime.now(),
            is_open=True,
            session_type="regular",
            next_open=datetime(2024, 1, 2, 9, 30),
            next_close=datetime(2024, 1, 1, 16, 0)
        )
        
        assert status.market == "NYSE"
        assert status.is_open is True
        assert status.session_type == "regular"
        assert status.timezone == "UTC"  # Default value