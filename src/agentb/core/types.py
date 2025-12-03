"""Core types and data models for Agent B."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions the executor can perform."""

    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    NAVIGATE = "NAVIGATE"


# Adaptive timeout configuration for state changes
STATE_CHANGE_TIMEOUTS = {
    ActionType.NAVIGATE: 15.0,  # Pages take time to load
    ActionType.CLICK: 3.0,      # Most clicks are instant
    ActionType.TYPE: 2.0,       # Typing is fast
    ActionType.SELECT: 2.0,     # Dropdowns are fast
}


class SemanticRole(str, Enum):
    """Semantic roles for plan steps."""

    PRIMARY_ACTION = "primary_action"
    NAVIGATION = "navigation"
    CONFIRMATION = "confirmation"
    FORM_FIELD = "form_field"


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step: int = Field(..., description="Step number in the plan")
    action: ActionType = Field(..., description="Action to perform")
    target_description: str = Field(
        ..., description="Natural language description of the target UI element"
    )
    value: Optional[str] = Field(
        None, description="Value for TYPE/SELECT actions"
    )
    semantic_role: SemanticRole = Field(
        ..., description="Semantic role of this action"
    )
    required_state: str = Field(
        ..., description="Expected UI state before this step can execute"
    )
    cached_coordinates: Optional["Coordinates"] = Field(
        None, description="Cached coordinates from successful execution (for performance)"
    )


class Plan(BaseModel):
    """A complete execution plan for a task."""

    task: str = Field(..., description="Original task description")
    steps: list[PlanStep] = Field(..., description="Ordered list of plan steps")


class Coordinates(BaseModel):
    """Screen coordinates for an element."""

    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")


class FailureInfo(BaseModel):
    """Information about a failed action."""

    step: PlanStep = Field(..., description="The step that failed")
    error_type: str = Field(..., description="Type of error encountered")
    error_message: str = Field(..., description="Human-readable error message")
    screenshot: Optional[bytes] = Field(
        None, description="Screenshot at time of failure"
    )


class ContextBundle(BaseModel):
    """Context bundle for re-planning after failure."""

    goal: str = Field(..., description="Original task goal")
    plan_history: list[PlanStep] = Field(
        ..., description="Steps executed so far"
    )
    failure: FailureInfo = Field(..., description="Details of the failure")
    current_screenshot: Optional[bytes] = Field(
        None, description="Current screenshot"
    )

    class Config:
        arbitrary_types_allowed = True


class Skill(BaseModel):
    """A cached skill in the Skills Library."""

    task: str = Field(..., description="Task description")
    plan: Plan = Field(..., description="Successful plan for this task")
    embedding: Optional[list[float]] = Field(
        None, description="Task embedding vector"
    )


class ElementSearchResult(BaseModel):
    """Result of searching for a UI element."""

    found: bool = Field(..., description="Whether the element was found")
    coordinates: Optional[Coordinates] = Field(
        None, description="Coordinates if found"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1)"
    )
    method: str = Field(
        default="dom", description="Method used: 'dom' or 'vision'"
    )
