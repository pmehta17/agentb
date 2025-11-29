"""Abstract interfaces for the six core modules."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from .plan import Plan, PlanAction
from .common import (
    Coordinates,
    ElementSearchResult,
    ExecutionResult,
    FailureContext,
    ReplanContext,
    SkillDefinition,
    StateSnapshot,
)


@runtime_checkable
class IStateCapturer(Protocol):
    """Interface for the State Capturer module.

    Handles capturing non-URL UI states through screenshots
    and detecting state changes.
    """

    async def capture_state(self, step_name: str) -> StateSnapshot:
        """Capture current UI state with screenshot.

        Args:
            step_name: Descriptive name for this state

        Returns:
            StateSnapshot with screenshot and metadata
        """
        ...

    async def get_screenshot(self) -> bytes:
        """Get current screenshot as bytes.

        Returns:
            Screenshot as PNG bytes
        """
        ...

    async def wait_for_change(self, timeout: float | None = None) -> bool:
        """Wait for UI state to change.

        Args:
            timeout: Max seconds to wait

        Returns:
            True if state changed, False if timeout reached
        """
        ...

    @property
    def last_screenshot(self) -> bytes | None:
        """Get the last captured screenshot."""
        ...


@runtime_checkable
class IExecutor(Protocol):
    """Interface for the Executor module.

    Browser automation wrapper around Playwright for
    performing low-level actions.
    """

    async def start(self) -> None:
        """Start the browser."""
        ...

    async def stop(self) -> None:
        """Stop the browser and cleanup."""
        ...

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        ...

    async def click(self, x: int, y: int) -> None:
        """Click at coordinates."""
        ...

    async def type_text(self, x: int, y: int, text: str) -> None:
        """Click and type text."""
        ...

    async def select(self, x: int, y: int, value: str) -> None:
        """Select a value from dropdown."""
        ...

    async def get_screenshot(self) -> bytes:
        """Get current screenshot."""
        ...

    async def find_element_by_text(self, text: str) -> Coordinates | None:
        """Find element by text using DOM.

        Returns:
            Coordinates if exactly one match, None otherwise
        """
        ...

    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        ...

    async def scroll(self, x: int, y: int, delta_y: int) -> None:
        """Scroll at position."""
        ...


@runtime_checkable
class IPerceptor(Protocol):
    """Interface for the Perceptor module.

    Vision-based search for UI elements using multimodal LLM.
    """

    async def find_element(
        self, screenshot: bytes, action: PlanAction
    ) -> ElementSearchResult:
        """Find UI element in screenshot using vision.

        Args:
            screenshot: Screenshot as PNG bytes
            action: Plan action with target description

        Returns:
            ElementSearchResult with coordinates or failure info
        """
        ...

    async def explain_failure(
        self, screenshot: bytes, action: PlanAction, error: str
    ) -> str:
        """Get explanation for why an action failed.

        Args:
            screenshot: Screenshot showing failure state
            action: The action that failed
            error: Error message

        Returns:
            Human-readable explanation
        """
        ...


@runtime_checkable
class IPlanner(Protocol):
    """Interface for the Planner module.

    High-level reasoning module using text LLM for
    plan generation and validation.
    """

    async def generate_plan(self, task: str) -> Plan:
        """Generate initial execution plan for a task.

        Args:
            task: Natural language task description

        Returns:
            Structured execution plan
        """
        ...

    async def regenerate_plan(self, context: ReplanContext) -> Plan:
        """Regenerate plan after failure.

        Args:
            context: Context about goal, history, and failure

        Returns:
            Corrected execution plan
        """
        ...

    async def validate_success(self, goal: str, screenshot: bytes) -> bool:
        """Validate if task completed successfully.

        Args:
            goal: Original task goal
            screenshot: Final screenshot

        Returns:
            True if successful, False otherwise
        """
        ...


@runtime_checkable
class ISkillsLibrary(Protocol):
    """Interface for the Skills Library module.

    Long-term semantic cache of successful workflows
    using vector similarity search.
    """

    def add_skill(self, task: str, plan: Plan) -> str:
        """Add a new skill to the library.

        Args:
            task: Task description
            plan: Successful execution plan

        Returns:
            Skill ID
        """
        ...

    def find_skill(self, task: str) -> Plan | None:
        """Find matching skill for a task.

        Args:
            task: Task description to match

        Returns:
            Matching plan if similarity > threshold, None otherwise
        """
        ...

    def get_embedding(self, task: str) -> list[float]:
        """Generate embedding for a task."""
        ...

    def list_skills(self) -> list[SkillDefinition]:
        """List all skills in the library."""
        ...

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill by ID."""
        ...

    @property
    def count(self) -> int:
        """Get number of skills in library."""
        ...


@runtime_checkable
class IOrchestrator(Protocol):
    """Interface for the Orchestrator module.

    Central nervous system coordinating all modules for
    planning, execution, and error recovery.
    """

    async def start(self) -> None:
        """Start the orchestrator and browser."""
        ...

    async def stop(self) -> None:
        """Stop the orchestrator and cleanup."""
        ...

    async def execute_task(self, task: str) -> ExecutionResult:
        """Execute a natural language task.

        Args:
            task: Natural language task description

        Returns:
            ExecutionResult with success/failure details
        """
        ...

    async def navigate_to(self, url: str) -> None:
        """Navigate to a URL."""
        ...

    @property
    def current_plan(self) -> Plan | None:
        """Get the current execution plan."""
        ...

    @property
    def executed_steps(self) -> list[int]:
        """Get list of executed step numbers."""
        ...


class BaseModule(ABC):
    """Abstract base class for all modules."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the module."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup module resources."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if module is ready for use."""
        pass
