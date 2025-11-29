"""Common shared type definitions."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    """State of a task execution."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Coordinates(BaseModel):
    """Screen coordinates for an element."""

    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)


class ElementSearchResult(BaseModel):
    """Result of searching for a UI element."""

    found: bool = Field(..., description="Whether the element was found")
    coordinates: Coordinates | None = Field(
        None, description="Coordinates if found"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1)"
    )
    method: str = Field(
        default="dom", description="Method used: 'dom' or 'vision'"
    )
    selector: str | None = Field(
        None, description="DOM selector if found by DOM"
    )
    alternatives: list[Coordinates] = Field(
        default_factory=list, description="Alternative matches if multiple found"
    )


class StateSnapshot(BaseModel):
    """Snapshot of UI state at a point in time."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When snapshot was taken"
    )
    screenshot: bytes | None = Field(
        None, description="Screenshot as PNG bytes"
    )
    url: str | None = Field(
        None, description="Current page URL"
    )
    title: str | None = Field(
        None, description="Page title"
    )
    step_name: str = Field(
        default="", description="Descriptive name for this state"
    )
    filepath: str | None = Field(
        None, description="Path where screenshot was saved"
    )

    class Config:
        arbitrary_types_allowed = True


class FailureContext(BaseModel):
    """Context about a failure for re-planning."""

    step: int = Field(..., description="Step number that failed")
    action_type: str = Field(..., description="Type of action that failed")
    target: str = Field(..., description="Target description")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    screenshot: bytes | None = Field(
        None, description="Screenshot at failure"
    )
    attempted_coordinates: Coordinates | None = Field(
        None, description="Coordinates that were attempted"
    )
    retry_count: int = Field(
        default=0, description="Number of retries attempted"
    )

    class Config:
        arbitrary_types_allowed = True


class ExecutionResult(BaseModel):
    """Result of executing a task or action."""

    success: bool = Field(..., description="Whether execution succeeded")
    task: str = Field(..., description="Task that was executed")
    state: TaskState = Field(..., description="Final task state")
    steps_completed: int = Field(
        default=0, description="Number of steps completed"
    )
    steps_total: int = Field(
        default=0, description="Total number of steps"
    )
    duration: float = Field(
        default=0.0, description="Execution duration in seconds"
    )
    error: str | None = Field(
        None, description="Error message if failed"
    )
    failure_context: FailureContext | None = Field(
        None, description="Context about the failure"
    )
    final_screenshot: bytes | None = Field(
        None, description="Screenshot after execution"
    )
    skill_saved: bool = Field(
        default=False, description="Whether a new skill was saved"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )

    class Config:
        arbitrary_types_allowed = True


class SkillDefinition(BaseModel):
    """Definition of a cached skill in the Skills Library."""

    id: str = Field(..., description="Unique skill identifier")
    task: str = Field(..., description="Task description")
    plan_json: str = Field(..., description="Serialized plan JSON")
    embedding: list[float] | None = Field(
        None, description="Task embedding vector"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When skill was created"
    )
    last_used: datetime | None = Field(
        None, description="When skill was last used"
    )
    use_count: int = Field(
        default=0, description="Number of times skill was used"
    )
    success_rate: float = Field(
        default=1.0, description="Success rate when used"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional skill metadata"
    )


class ReplanContext(BaseModel):
    """Context bundle for re-planning after failure."""

    goal: str = Field(..., description="Original task goal")
    executed_steps: list[int] = Field(
        default_factory=list, description="Step numbers executed successfully"
    )
    failure: FailureContext = Field(..., description="Failure details")
    current_screenshot: bytes | None = Field(
        None, description="Current UI state"
    )
    previous_plans: int = Field(
        default=0, description="Number of previous plan attempts"
    )

    class Config:
        arbitrary_types_allowed = True
