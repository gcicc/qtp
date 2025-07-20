"""
Base Agent Framework

Defines the core interfaces and communication protocols that all AI agents
in the QTP system must implement. Provides standardized messaging, state
management, and coordination mechanisms.

Key Components:
- BaseAgent: Abstract base class for all agents
- AgentMessage: Standardized inter-agent communication protocol
- AgentState: State management for agent lifecycle
- CommunicationBus: Message routing and delivery system
"""

from .agent import BaseAgent, AgentState, AgentConfig
from .communication import AgentMessage, MessageType, CommunicationBus
from .coordination import AgentCoordinator

__all__ = [
    "BaseAgent",
    "AgentState", 
    "AgentConfig",
    "AgentMessage",
    "MessageType",
    "CommunicationBus",
    "AgentCoordinator"
]