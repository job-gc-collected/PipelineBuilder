from .ai.base import AIAdapter
from .ai.claude import ClaudeAdapter
from .ai.mock import MockAIAdapter
from .ai.openai_adapter import OpenAIAdapter
from .core.agent import Agent, AgentMessage
from .core.checkpoint import CheckpointResult, checkpoint
from .core.goal_check import GoalCheckResult
from .core.versioning import CompatibilityError
from .core.dag import DAGNode, DAGResult, DAGSpec
from .core.loop import loop
from .core.pipeline import Pipeline
from .core.router import router
from .core.stage import stage
from .core.state import ArtifactStore, State
from .persistence import FileBackend, SQLiteBackend, StorageBackend
from .tracing import BatonTracer, TracedAIAdapter

__version__ = "1.0.0"

__all__ = [
    "Pipeline",
    "State",
    "ArtifactStore",
    "stage",
    "checkpoint",
    "CheckpointResult",
    "loop",
    "router",
    "DAGSpec",
    "DAGNode",
    "DAGResult",
    "Agent",
    "AgentMessage",
    "GoalCheckResult",
    "CompatibilityError",
    "AIAdapter",
    "ClaudeAdapter",
    "MockAIAdapter",
    "OpenAIAdapter",
    # Persistence
    "StorageBackend",
    "FileBackend",
    "SQLiteBackend",
    # Tracing
    "BatonTracer",
    "TracedAIAdapter",
]
