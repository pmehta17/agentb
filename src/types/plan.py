"""Plan and action type definitions."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions the executor can perform."""

    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    NAVIGATE = "NAVIGATE"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    PRESS_KEY = "PRESS_KEY"
    HOVER = "HOVER"


class ActionParameters(BaseModel):
    """Parameters for a plan action."""

    target_description: str | None = Field(
        None, description="Natural language description of target UI element"
    )
    value: str | None = Field(
        None, description="Value for TYPE/SELECT/NAVIGATE actions"
    )
    coordinates: tuple[int, int] | None = Field(
        None, description="Explicit (x, y) coordinates if known"
    )
    timeout: float = Field(
        default=30.0, description="Timeout in seconds for this action"
    )
    key: str | None = Field(
        None, description="Key to press for PRESS_KEY action"
    )
    scroll_delta: int | None = Field(
        None, description="Scroll amount for SCROLL action"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional action-specific parameters"
    )


class ExpectedOutcome(BaseModel):
    """Expected outcome after an action."""

    description: str = Field(..., description="Description of expected state")
    ui_changes: list[str] = Field(
        default_factory=list, description="Expected UI changes"
    )
    navigation: str | None = Field(
        None, description="Expected URL or page navigation"
    )


class SuccessCriteria(BaseModel):
    """Criteria for determining if an action succeeded."""

    element_visible: str | None = Field(
        None, description="Element that should be visible after action"
    )
    element_hidden: str | None = Field(
        None, description="Element that should be hidden after action"
    )
    url_pattern: str | None = Field(
        None, description="URL pattern to match after navigation"
    )
    state_change_required: bool = Field(
        default=True, description="Whether visual state change is required"
    )
    custom_validation: str | None = Field(
        None, description="Custom validation description for LLM"
    )


class PlanAction(BaseModel):
    """A single action in an execution plan."""

    step: int = Field(..., description="Step number in the plan")
    action_type: ActionType = Field(..., description="Type of action to perform")
    parameters: ActionParameters = Field(
        default_factory=ActionParameters, description="Action parameters"
    )
    expected_outcomes: list[ExpectedOutcome] = Field(
        default_factory=list, description="Expected outcomes after this action"
    )
    success_criteria: SuccessCriteria = Field(
        default_factory=SuccessCriteria, description="Criteria for success"
    )
    description: str = Field(
        default="", description="Human-readable description of this action"
    )
    semantic_role: str = Field(
        default="action", description="Semantic role: action, navigation, confirmation, input"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this action depends on"
    )
    retryable: bool = Field(
        default=True, description="Whether this action can be retried on failure"
    )
    max_retries: int = Field(
        default=3, description="Maximum retry attempts for this action"
    )


class PlanMetadata(BaseModel):
    """Metadata about a plan."""

    task: str = Field(..., description="Original task description")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the plan was created"
    )
    source: str = Field(
        default="planner", description="Source: 'planner', 'cache', or 'manual'"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1) in plan quality"
    )
    estimated_duration: float | None = Field(
        None, description="Estimated execution duration in seconds"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization"
    )
    version: int = Field(
        default=1, description="Plan version (incremented on regeneration)"
    )


class Plan(BaseModel):
    """A complete execution plan for a task."""

    metadata: PlanMetadata = Field(..., description="Plan metadata")
    actions: list[PlanAction] = Field(..., description="Ordered list of actions")

    @property
    def task(self) -> str:
        """Get the task description."""
        return self.metadata.task

    @property
    def step_count(self) -> int:
        """Get the number of steps in the plan."""
        return len(self.actions)

    def get_action(self, step: int) -> PlanAction | None:
        """Get action by step number."""
        for action in self.actions:
            if action.step == step:
                return action
        return None

    def get_remaining_actions(self, from_step: int) -> list[PlanAction]:
        """Get actions from a specific step onwards."""
        return [a for a in self.actions if a.step >= from_step]
