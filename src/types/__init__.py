"""Type definitions for Agent B."""

from .plan import ActionType, PlanAction, Plan, PlanMetadata
from .common import (
    TaskState,
    ExecutionResult,
    SkillDefinition,
    Coordinates,
    ElementSearchResult,
    FailureContext,
    StateSnapshot,
)
from .modules import (
    IOrchestrator,
    ISkillsLibrary,
    IPlanner,
    IPerceptor,
    IExecutor,
    IStateCapturer,
)

__all__ = [
    # Plan types
    "ActionType",
    "PlanAction",
    "Plan",
    "PlanMetadata",
    # Common types
    "TaskState",
    "ExecutionResult",
    "SkillDefinition",
    "Coordinates",
    "ElementSearchResult",
    "FailureContext",
    "StateSnapshot",
    # Module interfaces
    "IOrchestrator",
    "ISkillsLibrary",
    "IPlanner",
    "IPerceptor",
    "IExecutor",
    "IStateCapturer",
]
