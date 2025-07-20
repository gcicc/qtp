"""
Agent communication structures for the QTP platform.

This module defines the core messaging system for inter-agent communication,
including message types, protocols, and coordination mechanisms as specified
in the CLAUDE.md documentation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

from ..data.structures import OHLCV, Trade, Quote, MarketEvent
from ..strategies.signals import Signal, Position, Order
from ..risk.structures import RiskMetrics, Alert


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    # Data updates
    DATA_UPDATE = "data_update"
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    
    # Trading signals and decisions
    SIGNAL = "signal"
    POSITION_UPDATE = "position_update"
    ORDER_UPDATE = "order_update"
    EXECUTION_REPORT = "execution_report"
    
    # Risk and alerts
    ALERT = "alert"
    RISK_UPDATE = "risk_update"
    LIMIT_BREACH = "limit_breach"
    
    # Agent coordination
    QUERY = "query"
    RESPONSE = "response"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    
    # Research and analysis
    RESEARCH_UPDATE = "research_update"
    MARKET_ANALYSIS = "market_analysis"
    STRATEGY_ANALYSIS = "strategy_analysis"
    
    # Educational content
    EXPLANATION = "explanation"
    EDUCATIONAL_CONTENT = "educational_content"
    
    # System events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    ERROR = "error"
    
    # Broadcast messages
    BROADCAST = "broadcast"
    ANNOUNCEMENT = "announcement"


class MessagePriority(int, Enum):
    """Message priority levels (1-5, where 5 is highest)."""
    LOW = 1
    NORMAL = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class AgentType(str, Enum):
    """Types of agents in the system."""
    MARKET_RESEARCH = "market_research"
    STRATEGY_DEV = "strategy_dev"
    RISK_MONITOR = "risk_monitor"
    EXECUTION = "execution"
    EXPLANATION = "explanation"
    EDUCATION = "education"
    COORDINATOR = "coordinator"
    DATA_INGESTION = "data_ingestion"


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class AgentMessage(BaseModel):
    """
    Core agent message structure for inter-agent communication.
    
    Implements the communication protocol specified in CLAUDE.md with
    support for various message types, priorities, and routing.
    
    Example:
        >>> message = AgentMessage(
        ...     sender="market_research_agent",
        ...     receiver="strategy_dev_agent",
        ...     message_type=MessageType.MARKET_ANALYSIS,
        ...     payload={
        ...         "market_regime": "consolidation",
        ...         "confidence": 0.78,
        ...         "key_drivers": ["Fed policy uncertainty"]
        ...     },
        ...     priority=MessagePriority.HIGH
        ... )
    """
    # Message routing
    sender: str = Field(..., description="Sending agent identifier")
    receiver: str = Field(..., description="Receiving agent identifier or 'broadcast'")
    message_type: MessageType = Field(..., description="Type of message")
    
    # Message content
    payload: Dict[str, Any] = Field(..., description="Message payload data")
    subject: Optional[str] = Field(None, description="Message subject/title")
    
    # Message metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Message creation timestamp")
    message_id: Optional[str] = Field(None, description="Unique message identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request/response")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")
    
    # Message properties
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority (1-5)")
    requires_response: bool = Field(default=False, description="Whether message requires a response")
    expires_at: Optional[datetime] = Field(None, description="Message expiration time")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # Processing status
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Message processing status")
    sent_at: Optional[datetime] = Field(None, description="Message send timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Message delivery timestamp")
    processed_at: Optional[datetime] = Field(None, description="Message processing timestamp")
    retry_count: int = Field(default=0, ge=0, description="Current retry count")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    failed_at: Optional[datetime] = Field(None, description="Failure timestamp")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Message tags for filtering")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    @validator('expires_at')
    def expires_at_must_be_future(cls, v):
        """Ensure expiration time is in the future."""
        if v and v <= datetime.now():
            raise ValueError('Expiration time must be in the future')
        return v
    
    def is_broadcast(self) -> bool:
        """Check if message is a broadcast message."""
        return self.receiver.lower() == "broadcast"
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def age_seconds(self) -> float:
        """Calculate message age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()
    
    def mark_sent(self) -> None:
        """Mark message as sent."""
        self.status = MessageStatus.SENT
        self.sent_at = datetime.now()
    
    def mark_delivered(self) -> None:
        """Mark message as delivered."""
        self.status = MessageStatus.DELIVERED
        self.delivered_at = datetime.now()
    
    def mark_processed(self) -> None:
        """Mark message as processed."""
        self.status = MessageStatus.PROCESSED
        self.processed_at = datetime.now()
    
    def mark_failed(self, error: str) -> None:
        """Mark message as failed."""
        self.status = MessageStatus.FAILED
        self.error_message = error
        self.failed_at = datetime.now()
        self.retry_count += 1


class SignalMessage(AgentMessage):
    """
    Specialized message for trading signals.
    
    Extends AgentMessage with signal-specific validation and methods.
    """
    signal: Signal = Field(..., description="Trading signal data")
    
    def __init__(self, **data):
        # Set message type and payload from signal
        if 'signal' in data:
            data['message_type'] = MessageType.SIGNAL
            data['payload'] = data['signal'].dict()
        super().__init__(**data)


class MarketDataMessage(AgentMessage):
    """
    Specialized message for market data updates.
    
    Extends AgentMessage for OHLCV, trade, and quote data.
    """
    data_type: str = Field(..., description="Type of market data (ohlcv, trade, quote)")
    market_data: Union[OHLCV, Trade, Quote, MarketEvent] = Field(..., description="Market data")
    
    def __init__(self, **data):
        if 'market_data' in data:
            data['message_type'] = MessageType.MARKET_DATA
            data['payload'] = data['market_data'].dict()
        super().__init__(**data)


class AlertMessage(AgentMessage):
    """
    Specialized message for risk alerts.
    
    Extends AgentMessage with alert-specific functionality.
    """
    alert: Alert = Field(..., description="Risk alert data")
    
    def __init__(self, **data):
        if 'alert' in data:
            data['message_type'] = MessageType.ALERT
            data['payload'] = data['alert'].dict()
            # Set priority based on alert severity
            if data['alert'].severity.value == "emergency":
                data['priority'] = MessagePriority.CRITICAL
            elif data['alert'].severity.value == "critical":
                data['priority'] = MessagePriority.HIGH
            elif data['alert'].severity.value == "warning":
                data['priority'] = MessagePriority.MEDIUM
        super().__init__(**data)


class QueryMessage(AgentMessage):
    """
    Specialized message for agent queries.
    
    Extends AgentMessage for agent-to-agent queries and responses.
    """
    query: str = Field(..., description="Query string")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    expected_response_type: Optional[str] = Field(None, description="Expected response type")
    timeout_seconds: int = Field(default=30, gt=0, description="Query timeout in seconds")
    
    def __init__(self, **data):
        data['message_type'] = MessageType.QUERY
        data['requires_response'] = True
        data['payload'] = {
            'query': data.get('query'),
            'parameters': data.get('parameters', {}),
            'expected_response_type': data.get('expected_response_type')
        }
        # Set expiration based on timeout
        timeout = data.get('timeout_seconds', 30)
        data['expires_at'] = datetime.now() + timedelta(seconds=timeout)
        super().__init__(**data)


class ResponseMessage(AgentMessage):
    """
    Specialized message for query responses.
    
    Extends AgentMessage for responding to queries.
    """
    response_data: Any = Field(..., description="Response data")
    success: bool = Field(default=True, description="Whether query was successful")
    error_details: Optional[str] = Field(None, description="Error details if unsuccessful")
    
    def __init__(self, **data):
        data['message_type'] = MessageType.RESPONSE
        data['payload'] = {
            'response_data': data.get('response_data'),
            'success': data.get('success', True),
            'error_details': data.get('error_details')
        }
        super().__init__(**data)


class StatusMessage(AgentMessage):
    """
    Specialized message for agent status updates.
    
    Extends AgentMessage for reporting agent health and status.
    """
    agent_name: str = Field(..., description="Agent name")
    agent_type: AgentType = Field(..., description="Agent type")
    status: str = Field(..., description="Agent status")
    health_metrics: Dict[str, Any] = Field(default_factory=dict, description="Agent health metrics")
    active_tasks: int = Field(default=0, ge=0, description="Number of active tasks")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    
    def __init__(self, **data):
        data['message_type'] = MessageType.STATUS_UPDATE
        data['payload'] = {
            'agent_name': data.get('agent_name'),
            'agent_type': data.get('agent_type'),
            'status': data.get('status'),
            'health_metrics': data.get('health_metrics', {}),
            'active_tasks': data.get('active_tasks', 0),
            'last_activity': data.get('last_activity', datetime.now()).isoformat()
        }
        super().__init__(**data)


class MessageBus(BaseModel):
    """
    Message bus for agent communication management.
    
    Manages message routing, delivery, and coordination between agents.
    """
    messages: List[AgentMessage] = Field(default_factory=list, description="All messages")
    active_agents: Dict[str, AgentType] = Field(default_factory=dict, description="Active agents")
    message_handlers: Dict[MessageType, List[str]] = Field(default_factory=dict, description="Message type handlers")
    
    def register_agent(self, agent_id: str, agent_type: AgentType) -> None:
        """Register an agent with the message bus."""
        self.active_agents[agent_id] = agent_type
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message bus."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the bus."""
        if message.is_expired():
            message.mark_failed("Message expired before sending")
            return False
        
        if message.is_broadcast():
            # Send to all active agents except sender
            success = True
            for agent_id in self.active_agents:
                if agent_id != message.sender:
                    agent_message = message.copy()
                    agent_message.receiver = agent_id
                    success &= self._deliver_message(agent_message)
            return success
        else:
            # Send to specific recipient
            return self._deliver_message(message)
    
    def _deliver_message(self, message: AgentMessage) -> bool:
        """Internal method to deliver message to recipient."""
        if message.receiver not in self.active_agents:
            message.mark_failed(f"Recipient '{message.receiver}' not found")
            return False
        
        message.mark_sent()
        self.messages.append(message)
        message.mark_delivered()
        return True
    
    def get_messages_for_agent(self, agent_id: str, 
                              message_type: Optional[MessageType] = None,
                              unprocessed_only: bool = True) -> List[AgentMessage]:
        """Get messages for a specific agent."""
        messages = [
            msg for msg in self.messages
            if msg.receiver == agent_id or msg.is_broadcast()
        ]
        
        if message_type:
            messages = [msg for msg in messages if msg.message_type == message_type]
        
        if unprocessed_only:
            messages = [msg for msg in messages if msg.status != MessageStatus.PROCESSED]
        
        return messages
    
    def get_pending_responses(self, correlation_id: str) -> List[AgentMessage]:
        """Get pending response messages for a correlation ID."""
        return [
            msg for msg in self.messages
            if (msg.correlation_id == correlation_id and 
                msg.message_type == MessageType.RESPONSE and
                msg.status == MessageStatus.DELIVERED)
        ]
    
    def cleanup_expired_messages(self) -> int:
        """Remove expired messages and return count removed."""
        expired_count = 0
        for message in self.messages:
            if message.is_expired() and message.status != MessageStatus.PROCESSED:
                message.status = MessageStatus.EXPIRED
                expired_count += 1
        return expired_count
    
    def get_message_stats(self) -> Dict[str, int]:
        """Get message statistics."""
        stats = {
            'total_messages': len(self.messages),
            'pending': len([m for m in self.messages if m.status == MessageStatus.PENDING]),
            'sent': len([m for m in self.messages if m.status == MessageStatus.SENT]),
            'delivered': len([m for m in self.messages if m.status == MessageStatus.DELIVERED]),
            'processed': len([m for m in self.messages if m.status == MessageStatus.PROCESSED]),
            'failed': len([m for m in self.messages if m.status == MessageStatus.FAILED]),
            'expired': len([m for m in self.messages if m.status == MessageStatus.EXPIRED])
        }
        return stats


class ConversationThread(BaseModel):
    """
    Conversation thread for related messages.
    
    Groups related messages together for context and coordination.
    """
    thread_id: str = Field(..., description="Unique thread identifier")
    participants: List[str] = Field(..., description="Agent participants")
    topic: str = Field(..., description="Conversation topic")
    messages: List[AgentMessage] = Field(default_factory=list, description="Thread messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Thread creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last message time")
    is_active: bool = Field(default=True, description="Whether thread is active")
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the thread."""
        message.conversation_id = self.thread_id
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def get_messages_by_type(self, message_type: MessageType) -> List[AgentMessage]:
        """Get messages of a specific type from the thread."""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_latest_message(self) -> Optional[AgentMessage]:
        """Get the most recent message in the thread."""
        if not self.messages:
            return None
        return max(self.messages, key=lambda m: m.timestamp)