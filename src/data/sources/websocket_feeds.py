"""
WebSocket real-time data feeds for streaming market data.

This module provides WebSocket connections for real-time market data streaming
from various sources including financial data providers and exchanges.
"""

import asyncio
import json
import logging
import time
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from decimal import Decimal
import threading
from queue import Queue, Empty

from ..structures import OHLCV, Trade, Quote, MarketEvent, MarketEventType

logger = logging.getLogger(__name__)


class WebSocketState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    url: str
    subscriptions: List[str]
    heartbeat_interval: int = 30  # seconds
    reconnect_delay: int = 5  # seconds
    max_reconnect_attempts: int = 10
    ping_interval: int = 20  # seconds
    ping_timeout: int = 10  # seconds
    message_queue_size: int = 1000
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class WebSocketDataFeed:
    """
    Real-time WebSocket data feed manager.
    
    Provides a unified interface for connecting to various WebSocket data sources
    with automatic reconnection, message handling, and data parsing.
    
    Features:
    - Automatic connection management and reconnection
    - Message queuing and buffering
    - Multiple subscription management
    - Heartbeat and ping/pong handling
    - Data validation and parsing
    - Event-driven message processing
    - Thread-safe operation
    
    Example:
        >>> config = WebSocketConfig(
        ...     url="wss://api.provider.com/stream",
        ...     subscriptions=["AAPL", "MSFT", "GOOGL"]
        ... )
        >>> feed = WebSocketDataFeed(config)
        >>> 
        >>> def on_trade(trade_data):
        ...     print(f"New trade: {trade_data}")
        >>> 
        >>> feed.set_trade_handler(on_trade)
        >>> await feed.start()
    """
    
    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket data feed.
        
        Args:
            config: WebSocket configuration including URL and subscriptions
        """
        self.config = config
        self.state = WebSocketState.DISCONNECTED
        self.websocket = None
        self.message_queue = Queue(maxsize=config.message_queue_size)
        self.subscriptions = set(config.subscriptions)
        
        # Event handlers
        self.trade_handler: Optional[Callable] = None
        self.quote_handler: Optional[Callable] = None
        self.ohlcv_handler: Optional[Callable] = None
        self.event_handler: Optional[Callable] = None
        self.error_handler: Optional[Callable] = None
        
        # Connection management
        self.reconnect_count = 0
        self.last_heartbeat = None
        self.running = False
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'reconnections': 0,
            'connection_time': None,
            'last_message_time': None
        }
        
        logger.info(f"WebSocket feed initialized for {len(self.subscriptions)} symbols")
    
    async def start(self) -> None:
        """
        Start the WebSocket connection and message processing.
        
        This method establishes the WebSocket connection, subscribes to symbols,
        and begins processing incoming messages. It handles reconnection
        automatically if the connection is lost.
        """
        self.running = True
        self.stats['connection_time'] = datetime.now()
        
        while self.running:
            try:
                await self._connect()
                await self._message_loop()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._handle_error(e)
                
                if self.reconnect_count < self.config.max_reconnect_attempts:
                    await self._reconnect()
                else:
                    logger.error("Max reconnection attempts reached, stopping")
                    break
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and message processing."""
        self.running = False
        self.state = WebSocketState.DISCONNECTED
        
        if self.websocket:
            await self.websocket.close()
            
        logger.info("WebSocket feed stopped")
    
    def set_trade_handler(self, handler: Callable[[Trade], None]) -> None:
        """Set handler for trade messages."""
        self.trade_handler = handler
        
    def set_quote_handler(self, handler: Callable[[Quote], None]) -> None:
        """Set handler for quote messages."""
        self.quote_handler = handler
        
    def set_ohlcv_handler(self, handler: Callable[[OHLCV], None]) -> None:
        """Set handler for OHLCV bar messages."""
        self.ohlcv_handler = handler
        
    def set_event_handler(self, handler: Callable[[MarketEvent], None]) -> None:
        """Set handler for market event messages."""
        self.event_handler = handler
        
    def set_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """Set handler for error events."""
        self.error_handler = handler
    
    def add_subscription(self, symbol: str) -> None:
        """
        Add a new symbol subscription.
        
        Args:
            symbol: Symbol to subscribe to
        """
        if symbol not in self.subscriptions:
            self.subscriptions.add(symbol)
            if self.state == WebSocketState.CONNECTED:
                asyncio.create_task(self._subscribe_symbol(symbol))
                
    def remove_subscription(self, symbol: str) -> None:
        """
        Remove a symbol subscription.
        
        Args:
            symbol: Symbol to unsubscribe from
        """
        if symbol in self.subscriptions:
            self.subscriptions.remove(symbol)
            if self.state == WebSocketState.CONNECTED:
                asyncio.create_task(self._unsubscribe_symbol(symbol))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection and processing statistics."""
        stats = self.stats.copy()
        stats['state'] = self.state.value
        stats['subscriptions'] = len(self.subscriptions)
        stats['queue_size'] = self.message_queue.qsize()
        return stats
    
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        self.state = WebSocketState.CONNECTING
        logger.info(f"Connecting to WebSocket: {self.config.url}")
        
        # Prepare headers
        headers = self.config.headers or {}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        try:
            self.websocket = await websockets.connect(
                self.config.url,
                extra_headers=headers,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout
            )
            
            self.state = WebSocketState.CONNECTED
            self.reconnect_count = 0
            logger.info("WebSocket connected successfully")
            
            # Subscribe to symbols
            await self._subscribe_all()
            
        except Exception as e:
            self.state = WebSocketState.ERROR
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        last_heartbeat = time.time()
        
        while self.running and self.state == WebSocketState.CONNECTED:
            try:
                # Check for heartbeat timeout
                if time.time() - last_heartbeat > self.config.heartbeat_interval:
                    await self._send_heartbeat()
                    last_heartbeat = time.time()
                
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )
                
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = datetime.now()
                
                # Process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # Timeout is expected for heartbeat checking
                continue
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.state = WebSocketState.DISCONNECTED
                break
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                self._handle_error(e)
    
    async def _process_message(self, message: str) -> None:
        """
        Process incoming WebSocket message.
        
        Args:
            message: Raw message string from WebSocket
        """
        try:
            data = json.loads(message)
            message_type = data.get('type', '').lower()
            
            # Route message based on type
            if message_type == 'trade':
                await self._handle_trade_message(data)
            elif message_type == 'quote':
                await self._handle_quote_message(data)
            elif message_type == 'ohlcv' or message_type == 'bar':
                await self._handle_ohlcv_message(data)
            elif message_type == 'event':
                await self._handle_event_message(data)
            elif message_type == 'heartbeat' or message_type == 'ping':
                self.last_heartbeat = datetime.now()
            elif message_type == 'error':
                self._handle_api_error(data)
            else:
                logger.debug(f"Unknown message type: {message_type}")
                
            self.stats['messages_processed'] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
            self.stats['errors'] += 1
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.stats['errors'] += 1
            self._handle_error(e)
    
    async def _handle_trade_message(self, data: Dict[str, Any]) -> None:
        """Handle trade message."""
        try:
            trade = Trade(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                price=Decimal(str(data['price'])),
                size=int(data['size']),
                side=data.get('side', 'unknown'),
                trade_id=data.get('trade_id'),
                exchange=data.get('exchange')
            )
            
            if self.trade_handler:
                self.trade_handler(trade)
                
        except Exception as e:
            logger.error(f"Failed to process trade message: {e}")
    
    async def _handle_quote_message(self, data: Dict[str, Any]) -> None:
        """Handle quote message."""
        try:
            quote = Quote(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                bid_price=Decimal(str(data['bid_price'])),
                bid_size=int(data['bid_size']),
                ask_price=Decimal(str(data['ask_price'])),
                ask_size=int(data['ask_size']),
                exchange=data.get('exchange')
            )
            
            if self.quote_handler:
                self.quote_handler(quote)
                
        except Exception as e:
            logger.error(f"Failed to process quote message: {e}")
    
    async def _handle_ohlcv_message(self, data: Dict[str, Any]) -> None:
        """Handle OHLCV bar message."""
        try:
            ohlcv = OHLCV(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                open=Decimal(str(data['open'])),
                high=Decimal(str(data['high'])),
                low=Decimal(str(data['low'])),
                close=Decimal(str(data['close'])),
                volume=int(data['volume']),
                timeframe=data.get('timeframe', '1m')
            )
            
            if self.ohlcv_handler:
                self.ohlcv_handler(ohlcv)
                
        except Exception as e:
            logger.error(f"Failed to process OHLCV message: {e}")
    
    async def _handle_event_message(self, data: Dict[str, Any]) -> None:
        """Handle market event message."""
        try:
            event = MarketEvent(
                symbol=data['symbol'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                event_type=MarketEventType(data['event_type']),
                description=data['description'],
                data=data.get('data', {}),
                source=data.get('source'),
                impact_score=data.get('impact_score')
            )
            
            if self.event_handler:
                self.event_handler(event)
                
        except Exception as e:
            logger.error(f"Failed to process event message: {e}")
    
    async def _subscribe_all(self) -> None:
        """Subscribe to all configured symbols."""
        for symbol in self.subscriptions:
            await self._subscribe_symbol(symbol)
    
    async def _subscribe_symbol(self, symbol: str) -> None:
        """
        Subscribe to a specific symbol.
        
        Args:
            symbol: Symbol to subscribe to
        """
        try:
            subscribe_message = {
                "action": "subscribe",
                "symbol": symbol,
                "types": ["trade", "quote", "ohlcv"]
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """
        Unsubscribe from a specific symbol.
        
        Args:
            symbol: Symbol to unsubscribe from
        """
        try:
            unsubscribe_message = {
                "action": "unsubscribe",
                "symbol": symbol
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            logger.info(f"Unsubscribed from {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {symbol}: {e}")
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat message."""
        try:
            heartbeat_message = {
                "action": "heartbeat",
                "timestamp": int(time.time() * 1000)
            }
            
            await self.websocket.send(json.dumps(heartbeat_message))
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def _reconnect(self) -> None:
        """Handle reconnection logic."""
        self.state = WebSocketState.RECONNECTING
        self.reconnect_count += 1
        self.stats['reconnections'] += 1
        
        logger.info(f"Reconnecting... (attempt {self.reconnect_count})")
        
        # Close existing connection
        if self.websocket:
            await self.websocket.close()
            
        # Wait before reconnecting
        await asyncio.sleep(self.config.reconnect_delay)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle errors and call error handler if set."""
        self.stats['errors'] += 1
        
        if self.error_handler:
            try:
                self.error_handler(error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _handle_api_error(self, data: Dict[str, Any]) -> None:
        """Handle API error messages."""
        error_msg = data.get('message', 'Unknown API error')
        error_code = data.get('code', 'unknown')
        
        logger.error(f"API Error [{error_code}]: {error_msg}")
        
        # Create exception and handle
        error = Exception(f"API Error [{error_code}]: {error_msg}")
        self._handle_error(error)


class MultiSourceWebSocketManager:
    """
    Manager for multiple WebSocket data feeds.
    
    Coordinates multiple WebSocket connections and provides a unified
    interface for real-time data from different sources.
    """
    
    def __init__(self):
        """Initialize multi-source WebSocket manager."""
        self.feeds: Dict[str, WebSocketDataFeed] = {}
        self.running = False
        
        # Global handlers
        self.trade_handlers: List[Callable] = []
        self.quote_handlers: List[Callable] = []
        self.ohlcv_handlers: List[Callable] = []
        self.event_handlers: List[Callable] = []
        
        logger.info("Multi-source WebSocket manager initialized")
    
    def add_feed(self, name: str, config: WebSocketConfig) -> None:
        """
        Add a WebSocket feed.
        
        Args:
            name: Unique name for the feed
            config: WebSocket configuration
        """
        feed = WebSocketDataFeed(config)
        
        # Set up handlers to forward to global handlers
        feed.set_trade_handler(self._forward_trade)
        feed.set_quote_handler(self._forward_quote)
        feed.set_ohlcv_handler(self._forward_ohlcv)
        feed.set_event_handler(self._forward_event)
        
        self.feeds[name] = feed
        logger.info(f"Added WebSocket feed: {name}")
    
    def add_trade_handler(self, handler: Callable[[Trade], None]) -> None:
        """Add global trade handler."""
        self.trade_handlers.append(handler)
    
    def add_quote_handler(self, handler: Callable[[Quote], None]) -> None:
        """Add global quote handler."""
        self.quote_handlers.append(handler)
    
    def add_ohlcv_handler(self, handler: Callable[[OHLCV], None]) -> None:
        """Add global OHLCV handler."""
        self.ohlcv_handlers.append(handler)
    
    def add_event_handler(self, handler: Callable[[MarketEvent], None]) -> None:
        """Add global event handler."""
        self.event_handlers.append(handler)
    
    async def start_all(self) -> None:
        """Start all WebSocket feeds."""
        self.running = True
        
        tasks = []
        for name, feed in self.feeds.items():
            task = asyncio.create_task(feed.start())
            tasks.append(task)
            logger.info(f"Starting feed: {name}")
        
        # Wait for all feeds to start
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self) -> None:
        """Stop all WebSocket feeds."""
        self.running = False
        
        tasks = []
        for name, feed in self.feeds.items():
            task = asyncio.create_task(feed.stop())
            tasks.append(task)
            logger.info(f"Stopping feed: {name}")
        
        # Wait for all feeds to stop
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_feed_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all feeds."""
        stats = {}
        for name, feed in self.feeds.items():
            stats[name] = feed.get_statistics()
        return stats
    
    def _forward_trade(self, trade: Trade) -> None:
        """Forward trade to all global handlers."""
        for handler in self.trade_handlers:
            try:
                handler(trade)
            except Exception as e:
                logger.error(f"Error in trade handler: {e}")
    
    def _forward_quote(self, quote: Quote) -> None:
        """Forward quote to all global handlers."""
        for handler in self.quote_handlers:
            try:
                handler(quote)
            except Exception as e:
                logger.error(f"Error in quote handler: {e}")
    
    def _forward_ohlcv(self, ohlcv: OHLCV) -> None:
        """Forward OHLCV to all global handlers."""
        for handler in self.ohlcv_handlers:
            try:
                handler(ohlcv)
            except Exception as e:
                logger.error(f"Error in OHLCV handler: {e}")
    
    def _forward_event(self, event: MarketEvent) -> None:
        """Forward event to all global handlers."""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")


# Predefined configurations for popular data sources
POLYGON_CONFIG = WebSocketConfig(
    url="wss://socket.polygon.io/stocks",
    subscriptions=[],
    heartbeat_interval=30
)

ALPACA_CONFIG = WebSocketConfig(
    url="wss://stream.data.alpaca.markets/v2/iex",
    subscriptions=[],
    heartbeat_interval=30
)

FINNHUB_CONFIG = WebSocketConfig(
    url="wss://ws.finnhub.io",
    subscriptions=[],
    heartbeat_interval=30
)