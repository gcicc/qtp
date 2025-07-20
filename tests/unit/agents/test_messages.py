"""
Unit tests for agent communication structures.

Tests all agent communication structures including AgentMessage, specialized
message types, and MessageBus with comprehensive validation and functionality verification.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.agents.messages import (
    AgentMessage, SignalMessage, MarketDataMessage, AlertMessage,
    QueryMessage, ResponseMessage, StatusMessage, MessageBus, ConversationThread,
    MessageType, MessagePriority, AgentType, MessageStatus
)
from src.strategies.signals import Signal, SignalType, SignalStrength
from src.data.structures import OHLCV, Trade
from src.risk.structures import Alert, AlertType, AlertSeverity


class TestAgentMessage:
    """Test cases for AgentMessage data structure."""
    
    def test_valid_message_creation(self):
        """Test creation of valid agent message."""
        message = AgentMessage(
            sender="market_research_agent",
            receiver="strategy_dev_agent",
            message_type=MessageType.MARKET_ANALYSIS,
            payload={
                "market_regime": "consolidation",
                "confidence": 0.78,
                "key_drivers": ["Fed policy uncertainty"]
            },
            priority=MessagePriority.HIGH
        )
        
        assert message.sender == "market_research_agent"
        assert message.receiver == "strategy_dev_agent"
        assert message.message_type == MessageType.MARKET_ANALYSIS
        assert message.priority == MessagePriority.HIGH
        assert message.payload["market_regime"] == "consolidation"
        assert message.status == MessageStatus.PENDING
    
    def test_message_expiration_validation(self):
        """Test message expiration validation."""
        # Valid future expiration
        future_time = datetime.now() + timedelta(hours=1)
        message = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"},
            expires_at=future_time
        )
        assert message.expires_at == future_time
        
        # Invalid past expiration
        with pytest.raises(ValueError, match="Expiration time must be in the future"):
            past_time = datetime.now() - timedelta(hours=1)
            AgentMessage(
                sender="agent1",
                receiver="agent2",
                message_type=MessageType.QUERY,
                payload={"query": "test"},
                expires_at=past_time
            )
    
    def test_message_broadcast_detection(self):
        """Test broadcast message detection."""
        # Broadcast message
        broadcast_msg = AgentMessage(
            sender="coordinator",
            receiver="broadcast",
            message_type=MessageType.ANNOUNCEMENT,
            payload={"announcement": "System maintenance scheduled"}
        )
        assert broadcast_msg.is_broadcast() is True
        
        # Regular message
        regular_msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"}
        )
        assert regular_msg.is_broadcast() is False
    
    def test_message_expiration_check(self):
        """Test message expiration checking."""
        # Non-expired message
        future_time = datetime.now() + timedelta(minutes=30)
        message = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"},
            expires_at=future_time
        )
        assert message.is_expired() is False
        
        # Expired message
        past_time = datetime.now() - timedelta(minutes=30)
        expired_message = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"},
            expires_at=past_time
        )
        assert expired_message.is_expired() is True
    
    def test_message_retry_logic(self):
        """Test message retry logic."""
        message = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"},
            max_retries=3
        )
        
        # Initially can retry
        assert message.can_retry() is True
        assert message.retry_count == 0
        
        # Mark as failed to increment retry count
        message.mark_failed("Network error")
        assert message.retry_count == 1
        assert message.can_retry() is True
        
        # Exhaust retries
        message.mark_failed("Another error")
        message.mark_failed("Final error")
        assert message.retry_count == 3
        assert message.can_retry() is False
    
    def test_message_status_transitions(self):
        """Test message status transitions."""
        message = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test"}
        )
        
        # Initial status
        assert message.status == MessageStatus.PENDING
        
        # Mark as sent
        message.mark_sent()
        assert message.status == MessageStatus.SENT
        assert message.sent_at is not None
        
        # Mark as delivered
        message.mark_delivered()
        assert message.status == MessageStatus.DELIVERED
        assert message.delivered_at is not None
        
        # Mark as processed
        message.mark_processed()
        assert message.status == MessageStatus.PROCESSED
        assert message.processed_at is not None


class TestSpecializedMessages:
    """Test cases for specialized message types."""
    
    def test_signal_message(self):
        """Test SignalMessage creation and functionality."""
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            confidence=0.85,
            strength=SignalStrength.STRONG,
            strategy_name="TestStrategy",
            reasoning="Test signal"
        )
        
        signal_msg = SignalMessage(
            sender="strategy_agent",
            receiver="execution_agent",
            signal=signal
        )
        
        assert signal_msg.message_type == MessageType.SIGNAL
        assert signal_msg.signal == signal
        assert signal_msg.payload["symbol"] == "AAPL"
        assert signal_msg.payload["signal_type"] == SignalType.BUY
    
    def test_market_data_message(self):
        """Test MarketDataMessage creation and functionality."""
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
        
        data_msg = MarketDataMessage(
            sender="data_ingestion_agent",
            receiver="broadcast",
            data_type="ohlcv",
            market_data=ohlcv
        )
        
        assert data_msg.message_type == MessageType.MARKET_DATA
        assert data_msg.data_type == "ohlcv"
        assert data_msg.market_data == ohlcv
        assert data_msg.payload["symbol"] == "AAPL"
    
    def test_alert_message(self):
        """Test AlertMessage creation and priority setting."""
        alert = Alert(
            alert_type=AlertType.DRAWDOWN,
            severity=AlertSeverity.CRITICAL,
            message="Critical drawdown detected"
        )
        
        alert_msg = AlertMessage(
            sender="risk_monitor_agent",
            receiver="broadcast",
            alert=alert
        )
        
        assert alert_msg.message_type == MessageType.ALERT
        assert alert_msg.alert == alert
        assert alert_msg.priority == MessagePriority.HIGH  # Critical severity -> High priority
    
    def test_query_message(self):
        """Test QueryMessage creation and timeout handling."""
        query_msg = QueryMessage(
            sender="strategy_agent",
            receiver="market_research_agent",
            query="What is the current market regime?",
            parameters={"lookback_days": 30},
            timeout_seconds=60
        )
        
        assert query_msg.message_type == MessageType.QUERY
        assert query_msg.requires_response is True
        assert query_msg.query == "What is the current market regime?"
        assert query_msg.parameters["lookback_days"] == 30
        assert query_msg.expires_at is not None
    
    def test_response_message(self):
        """Test ResponseMessage creation."""
        response_msg = ResponseMessage(
            sender="market_research_agent",
            receiver="strategy_agent",
            response_data={"market_regime": "bullish", "confidence": 0.8},
            correlation_id="query_123"
        )
        
        assert response_msg.message_type == MessageType.RESPONSE
        assert response_msg.response_data["market_regime"] == "bullish"
        assert response_msg.success is True
        assert response_msg.correlation_id == "query_123"
    
    def test_status_message(self):
        """Test StatusMessage creation."""
        status_msg = StatusMessage(
            sender="risk_monitor_agent",
            receiver="coordinator",
            agent_name="risk_monitor_agent",
            agent_type=AgentType.RISK_MONITOR,
            status="healthy",
            health_metrics={"cpu_usage": 0.45, "memory_usage": 0.62},
            active_tasks=3
        )
        
        assert status_msg.message_type == MessageType.STATUS_UPDATE
        assert status_msg.agent_name == "risk_monitor_agent"
        assert status_msg.agent_type == AgentType.RISK_MONITOR
        assert status_msg.health_metrics["cpu_usage"] == 0.45


class TestMessageBus:
    """Test cases for MessageBus functionality."""
    
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        bus = MessageBus()
        
        # Register agents
        bus.register_agent("agent1", AgentType.MARKET_RESEARCH)
        bus.register_agent("agent2", AgentType.STRATEGY_DEV)
        
        assert "agent1" in bus.active_agents
        assert "agent2" in bus.active_agents
        assert bus.active_agents["agent1"] == AgentType.MARKET_RESEARCH
        
        # Unregister agent
        bus.unregister_agent("agent1")
        assert "agent1" not in bus.active_agents
        assert "agent2" in bus.active_agents
    
    def test_message_sending(self):
        """Test message sending through bus."""
        bus = MessageBus()
        bus.register_agent("sender", AgentType.MARKET_RESEARCH)
        bus.register_agent("receiver", AgentType.STRATEGY_DEV)
        
        message = AgentMessage(
            sender="sender",
            receiver="receiver",
            message_type=MessageType.MARKET_ANALYSIS,
            payload={"analysis": "bullish trend"}
        )
        
        # Send message
        success = bus.send_message(message)
        assert success is True
        assert message.status == MessageStatus.DELIVERED
        assert len(bus.messages) == 1
    
    def test_broadcast_message_sending(self):
        """Test broadcast message sending."""
        bus = MessageBus()
        bus.register_agent("sender", AgentType.COORDINATOR)
        bus.register_agent("agent1", AgentType.MARKET_RESEARCH)
        bus.register_agent("agent2", AgentType.STRATEGY_DEV)
        bus.register_agent("agent3", AgentType.RISK_MONITOR)
        
        broadcast_msg = AgentMessage(
            sender="sender",
            receiver="broadcast",
            message_type=MessageType.ANNOUNCEMENT,
            payload={"announcement": "System maintenance"}
        )
        
        # Send broadcast
        success = bus.send_message(broadcast_msg)
        assert success is True
        # Should create messages for all agents except sender (3 total)
        assert len(bus.messages) == 3
    
    def test_message_retrieval(self):
        """Test message retrieval for agents."""
        bus = MessageBus()
        bus.register_agent("agent1", AgentType.MARKET_RESEARCH)
        bus.register_agent("agent2", AgentType.STRATEGY_DEV)
        
        # Send messages
        msg1 = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test1"}
        )
        
        msg2 = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.MARKET_ANALYSIS,
            payload={"analysis": "test2"}
        )
        
        bus.send_message(msg1)
        bus.send_message(msg2)
        
        # Get all messages for agent2
        all_messages = bus.get_messages_for_agent("agent2")
        assert len(all_messages) == 2
        
        # Get specific message type
        query_messages = bus.get_messages_for_agent("agent2", MessageType.QUERY)
        assert len(query_messages) == 1
        assert query_messages[0].message_type == MessageType.QUERY
    
    def test_expired_message_cleanup(self):
        """Test cleanup of expired messages."""
        bus = MessageBus()
        bus.register_agent("agent1", AgentType.MARKET_RESEARCH)
        bus.register_agent("agent2", AgentType.STRATEGY_DEV)
        
        # Create expired message
        past_time = datetime.now() - timedelta(hours=1)
        expired_msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "expired"},
            expires_at=past_time
        )
        
        bus.send_message(expired_msg)
        
        # Cleanup expired messages
        expired_count = bus.cleanup_expired_messages()
        assert expired_count == 1
        assert bus.messages[0].status == MessageStatus.EXPIRED
    
    def test_message_statistics(self):
        """Test message statistics generation."""
        bus = MessageBus()
        bus.register_agent("agent1", AgentType.MARKET_RESEARCH)
        bus.register_agent("agent2", AgentType.STRATEGY_DEV)
        
        # Create messages with different statuses
        msg1 = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test1"}
        )
        
        msg2 = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "test2"}
        )
        
        bus.send_message(msg1)
        bus.send_message(msg2)
        
        # Mark one as processed
        msg1.mark_processed()
        
        stats = bus.get_message_stats()
        assert stats["total_messages"] == 2
        assert stats["delivered"] == 1  # msg2
        assert stats["processed"] == 1  # msg1


class TestConversationThread:
    """Test cases for ConversationThread functionality."""
    
    def test_conversation_thread_creation(self):
        """Test conversation thread creation."""
        thread = ConversationThread(
            thread_id="thread_123",
            participants=["agent1", "agent2"],
            topic="Market Analysis Discussion"
        )
        
        assert thread.thread_id == "thread_123"
        assert "agent1" in thread.participants
        assert "agent2" in thread.participants
        assert thread.topic == "Market Analysis Discussion"
        assert thread.is_active is True
        assert len(thread.messages) == 0
    
    def test_conversation_thread_message_management(self):
        """Test adding and managing messages in thread."""
        thread = ConversationThread(
            thread_id="thread_123",
            participants=["agent1", "agent2"],
            topic="Risk Discussion"
        )
        
        # Add messages to thread
        msg1 = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.QUERY,
            payload={"query": "What's the current risk level?"}
        )
        
        msg2 = AgentMessage(
            sender="agent2",
            receiver="agent1",
            message_type=MessageType.RESPONSE,
            payload={"response": "Risk level is moderate"}
        )
        
        thread.add_message(msg1)
        thread.add_message(msg2)
        
        # Check messages were added with thread ID
        assert len(thread.messages) == 2
        assert msg1.conversation_id == "thread_123"
        assert msg2.conversation_id == "thread_123"
        
        # Get messages by type
        queries = thread.get_messages_by_type(MessageType.QUERY)
        responses = thread.get_messages_by_type(MessageType.RESPONSE)
        
        assert len(queries) == 1
        assert len(responses) == 1
        
        # Get latest message
        latest = thread.get_latest_message()
        assert latest == msg2  # Most recent