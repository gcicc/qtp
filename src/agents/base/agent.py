"""
Base Agent Implementation

Defines the abstract base class and core functionality that all AI agents
in the QTP system inherit from. Provides standardized lifecycle management,
communication interfaces, and state tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """
    Enumeration of possible agent states.
    
    Defines the lifecycle states that agents can be in during operation.
    """
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentConfig:
    """
    Configuration class for AI agents.
    
    Contains all parameters and settings needed to configure an agent,
    including performance parameters, communication settings, and agent-specific options.
    """
    
    # Basic agent information
    name: str
    agent_type: str
    description: str = ""
    version: str = "1.0.0"
    
    # Performance parameters
    max_concurrent_tasks: int = 5
    task_timeout: int = 300  # seconds
    heartbeat_interval: int = 30  # seconds
    
    # Communication settings
    message_queue_size: int = 100
    broadcast_enabled: bool = True
    
    # Agent-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get an agent-specific parameter with optional default."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set an agent-specific parameter."""
        self.parameters[key] = value


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    This class defines the interface that all agents must implement and provides
    common functionality for lifecycle management, communication, and task execution.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.now()
        self.last_heartbeat = None
        
        # Task management
        self.active_tasks = {}
        self.task_history = []
        self.performance_metrics = {}
        
        # Communication
        self.message_handlers = {}
        self.subscribed_topics = set()
        self.communication_bus = None
        
        # Lifecycle hooks
        self.on_state_change_callbacks = []
        
        logger.info(f"Initialized agent: {self.config.name} ({self.agent_id})")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent with any required setup.
        
        This method should be implemented by each agent to perform
        agent-specific initialization tasks.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: 'AgentMessage') -> Optional['AgentMessage']:
        """
        Process an incoming message from another agent.
        
        Args:
            message: Message to process
            
        Returns:
            Optional response message
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific task assigned to this agent.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Task parameters and input data
            
        Returns:
            Task results dictionary
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get a list of capabilities this agent provides.
        
        Returns:
            List of capability names
        """
        pass
    
    async def start(self) -> bool:
        """
        Start the agent and begin operation.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if not await self.initialize():
                logger.error(f"Failed to initialize agent {self.config.name}")
                self._set_state(AgentState.ERROR)
                return False
            
            self._set_state(AgentState.ACTIVE)
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start message processing
            asyncio.create_task(self._message_processing_loop())
            
            logger.info(f"Agent {self.config.name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent {self.config.name}: {e}")
            self._set_state(AgentState.ERROR)
            return False
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        try:
            self._set_state(AgentState.SHUTDOWN)
            
            # Cancel active tasks
            for task_id in list(self.active_tasks.keys()):
                await self.cancel_task(task_id)
            
            # Perform agent-specific cleanup
            await self._cleanup()
            
            logger.info(f"Agent {self.config.name} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping agent {self.config.name}: {e}")
    
    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """
        Submit a task for execution by this agent.
        
        Args:
            task_data: Task parameters and input data
            
        Returns:
            Task ID for tracking
        """
        if self.state != AgentState.ACTIVE:
            raise RuntimeError(f"Agent {self.config.name} is not active")
        
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            raise RuntimeError(f"Agent {self.config.name} at maximum task capacity")
        
        task_id = str(uuid.uuid4())
        task_info = {
            'id': task_id,
            'data': task_data,
            'submitted_at': datetime.now(),
            'status': 'submitted'
        }
        
        self.active_tasks[task_id] = task_info
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task_wrapper(task_id, task_data))
        
        logger.info(f"Task {task_id} submitted to agent {self.config.name}")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an active task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.active_tasks:
            return False
        
        task_info = self.active_tasks[task_id]
        task_info['status'] = 'cancelled'
        task_info['completed_at'] = datetime.now()
        
        # Move to history
        self.task_history.append(task_info)
        del self.active_tasks[task_id]
        
        logger.info(f"Task {task_id} cancelled on agent {self.config.name}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of task to check
            
        Returns:
            Task status string or None if not found
        """
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]['status']
        
        # Check history
        for task in self.task_history:
            if task['id'] == task_id:
                return task['status']
        
        return None
    
    def subscribe_to_topic(self, topic: str) -> None:
        """Subscribe to a message topic."""
        self.subscribed_topics.add(topic)
        logger.debug(f"Agent {self.config.name} subscribed to topic: {topic}")
    
    def unsubscribe_from_topic(self, topic: str) -> None:
        """Unsubscribe from a message topic."""
        self.subscribed_topics.discard(topic)
        logger.debug(f"Agent {self.config.name} unsubscribed from topic: {topic}")
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
    
    def add_state_change_callback(self, callback: Callable[['BaseAgent', AgentState], None]) -> None:
        """Add a callback to be notified of state changes."""
        self.on_state_change_callbacks.append(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.config.name,
            'state': self.state.value,
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len([t for t in self.task_history if t['status'] == 'completed']),
            'failed_tasks': len([t for t in self.task_history if t['status'] == 'failed']),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            **self.performance_metrics
        }
    
    def _set_state(self, new_state: AgentState) -> None:
        """Set agent state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        
        logger.debug(f"Agent {self.config.name} state: {old_state.value} -> {new_state.value}")
        
        # Notify callbacks
        for callback in self.on_state_change_callbacks:
            try:
                callback(self, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    async def _execute_task_wrapper(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Wrapper for task execution with error handling and tracking."""
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            return
        
        try:
            self._set_state(AgentState.BUSY)
            task_info['status'] = 'running'
            task_info['started_at'] = datetime.now()
            
            # Execute the task
            result = await asyncio.wait_for(
                self.execute_task(task_id, task_data),
                timeout=self.config.task_timeout
            )
            
            task_info['status'] = 'completed'
            task_info['result'] = result
            
        except asyncio.TimeoutError:
            task_info['status'] = 'timeout'
            task_info['error'] = 'Task execution timed out'
            logger.warning(f"Task {task_id} timed out on agent {self.config.name}")
            
        except Exception as e:
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            logger.error(f"Task {task_id} failed on agent {self.config.name}: {e}")
            
        finally:
            # Clean up
            task_info['completed_at'] = datetime.now()
            self.task_history.append(task_info)
            del self.active_tasks[task_id]
            
            # Return to idle if no more active tasks
            if not self.active_tasks:
                self._set_state(AgentState.IDLE)
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat signals."""
        while self.state != AgentState.SHUTDOWN:
            try:
                self.last_heartbeat = datetime.now()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error for agent {self.config.name}: {e}")
                break
    
    async def _message_processing_loop(self) -> None:
        """Process incoming messages."""
        while self.state != AgentState.SHUTDOWN:
            try:
                # This would integrate with the communication bus
                # For now, just sleep to prevent busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Message processing error for agent {self.config.name}: {e}")
                break
    
    async def _cleanup(self) -> None:
        """
        Perform agent-specific cleanup.
        
        Subclasses can override this method to perform custom cleanup
        such as closing connections or saving state.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.config.name} ({self.state.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"BaseAgent(name='{self.config.name}', "
                f"type='{self.config.agent_type}', "
                f"id='{self.agent_id}', "
                f"state={self.state.value})")