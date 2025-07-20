"""
Real-time data feed implementations.

Provides WebSocket and streaming data connections for live market data feeds.
Handles connection management, reconnection logic, and data buffering.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


class RealTimeFeed(ABC):
    """
    Abstract base class for real-time market data feeds.
    
    Provides a consistent interface for streaming market data regardless
    of the underlying data provider or connection mechanism.
    """
    
    def __init__(self, symbols: List[str], callback: Callable[[Dict], None]):
        self.symbols = symbols
        self.callback = callback
        self.is_connected = False
        self.connection = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data feed."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close the data feed connection."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to additional symbols."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        pass


class WebSocketFeed(RealTimeFeed):
    """
    WebSocket-based real-time data feed implementation.
    
    Handles WebSocket connections with automatic reconnection and
    message parsing for market data streams.
    """
    
    def __init__(
        self,
        symbols: List[str],
        callback: Callable[[Dict], None],
        ws_url: str,
        api_key: Optional[str] = None
    ):
        super().__init__(symbols, callback)
        self.ws_url = ws_url
        self.api_key = api_key
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    async def connect(self) -> bool:
        """
        Connect to the WebSocket feed.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket feed: {self.ws_url}")
            
            # Placeholder WebSocket connection logic
            # In production, would use websockets library:
            # self.connection = await websockets.connect(self.ws_url)
            
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
            logger.info("WebSocket connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.connection:
            logger.info("Closing WebSocket connection")
            # await self.connection.close()
            self.is_connected = False
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to additional symbols via WebSocket."""
        if not self.is_connected:
            logger.error("Cannot subscribe: WebSocket not connected")
            return False
            
        try:
            subscription_message = {
                "action": "subscribe",
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
            
            # In production: await self.connection.send(json.dumps(subscription_message))
            logger.info(f"Subscribed to symbols: {symbols}")
            self.symbols.extend(symbols)
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols via WebSocket."""
        if not self.is_connected:
            logger.error("Cannot unsubscribe: WebSocket not connected")
            return False
            
        try:
            unsubscription_message = {
                "action": "unsubscribe", 
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
            
            # In production: await self.connection.send(json.dumps(unsubscription_message))
            logger.info(f"Unsubscribed from symbols: {symbols}")
            
            for symbol in symbols:
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")
            return False
    
    async def _listen_for_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        while self.is_connected:
            try:
                # In production: message = await self.connection.recv()
                # data = json.loads(message)
                
                # Placeholder message processing
                await asyncio.sleep(1)  # Simulate message reception
                
                # Mock data for demonstration
                mock_data = {
                    "symbol": "AAPL",
                    "price": 150.25,
                    "volume": 1000,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Call the registered callback with the data
                if self.callback:
                    self.callback(mock_data)
                    
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}")
                await self._handle_reconnection()
                break
    
    async def _handle_reconnection(self) -> None:
        """Handle automatic reconnection logic."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
            
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        
        logger.info(f"Attempting reconnection in {wait_time} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(wait_time)
        
        await self.connect()


class PollingFeed(RealTimeFeed):
    """
    Polling-based data feed for REST API sources.
    
    Periodically polls REST endpoints for updated market data.
    Suitable for data sources that don't provide WebSocket streams.
    """
    
    def __init__(
        self,
        symbols: List[str],
        callback: Callable[[Dict], None],
        api_url: str,
        poll_interval: float = 1.0,
        api_key: Optional[str] = None
    ):
        super().__init__(symbols, callback)
        self.api_url = api_url
        self.poll_interval = poll_interval
        self.api_key = api_key
        self._polling_task = None
        
    async def connect(self) -> bool:
        """Start the polling loop."""
        try:
            logger.info(f"Starting polling feed with {self.poll_interval}s interval")
            self.is_connected = True
            self._polling_task = asyncio.create_task(self._polling_loop())
            return True
        except Exception as e:
            logger.error(f"Failed to start polling feed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Stop the polling loop."""
        self.is_connected = False
        if self._polling_task:
            self._polling_task.cancel()
            logger.info("Polling feed stopped")
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Add symbols to the polling list."""
        self.symbols.extend(symbols)
        logger.info(f"Added symbols to polling: {symbols}")
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Remove symbols from the polling list."""
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
        logger.info(f"Removed symbols from polling: {symbols}")
        return True
    
    async def _polling_loop(self) -> None:
        """Main polling loop for fetching data."""
        while self.is_connected:
            try:
                # In production, would make HTTP requests to fetch data
                # data = await self._fetch_market_data(self.symbols)
                
                # Mock data for demonstration
                for symbol in self.symbols:
                    mock_data = {
                        "symbol": symbol,
                        "price": 100.0,  # Would be actual price from API
                        "volume": 500,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if self.callback:
                        self.callback(mock_data)
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch market data from REST API."""
        # Placeholder for actual HTTP requests
        # Would use aiohttp or similar to make API calls
        pass