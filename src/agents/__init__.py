"""
AI Agent System

Specialized AI agents that work collaboratively to provide comprehensive trading
intelligence, market analysis, risk management, and educational content.

The agent system employs a multi-agent architecture where each agent has defined
responsibilities and communication protocols. Agents can operate independently
or collaborate on complex tasks requiring multiple perspectives.

Core AI Agents:
- MarketResearchAgent: Autonomous market intelligence gathering and analysis
- StrategyDevelopmentAgent: Automated strategy generation and optimization  
- RiskAssessmentAgent: Continuous portfolio risk monitoring and management
- EducationalAgent: Personalized learning content generation
- ExecutionAgent: Intelligent trade timing and order management
- ExplanationAgent: Natural language generation for trading decisions

Architecture Components:
- BaseAgent: Abstract base class for all agents
- AgentCommunication: Inter-agent messaging and coordination system
- AgentOrchestrator: Centralized agent management and task coordination
"""

from .base import BaseAgent, AgentMessage, AgentState
from .market_research import MarketResearchAgent
from .strategy_dev import StrategyDevelopmentAgent  
from .risk_monitor import RiskAssessmentAgent
from .education import EducationalAgent
from .execution import ExecutionAgent
from .explanation import ExplanationAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentState",
    "MarketResearchAgent",
    "StrategyDevelopmentAgent",
    "RiskAssessmentAgent", 
    "EducationalAgent",
    "ExecutionAgent",
    "ExplanationAgent",
    "AgentOrchestrator"
]