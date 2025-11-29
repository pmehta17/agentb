"""Perceptor module - Analyzes tasks and gathers environmental context."""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.types.common import TaskState
from src.types.modules import IPerceptor
from src.modules.state_capturer import StateCapturer


class TaskAnalysis(BaseModel):
    """Result of analyzing a task description."""

    original_task: str = Field(..., description="Original task description")
    intent: str = Field(..., description="Primary intent/goal of the task")
    action_verbs: list[str] = Field(
        default_factory=list, description="Action verbs found in task"
    )
    target_elements: list[str] = Field(
        default_factory=list, description="UI elements or targets mentioned"
    )
    parameters: dict[str, str] = Field(
        default_factory=dict, description="Extracted parameters (values to input)"
    )
    domain: str | None = Field(
        None, description="Detected domain/application (e.g., 'notion', 'linear')"
    )
    complexity: str = Field(
        default="simple", description="Estimated complexity: simple, moderate, complex"
    )
    requires_auth: bool = Field(
        default=False, description="Whether task likely requires authentication"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Important keywords extracted"
    )


class EnvironmentContext(BaseModel):
    """Current environment context."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When context was gathered"
    )
    current_url: str | None = Field(None, description="Current page URL")
    page_title: str | None = Field(None, description="Current page title")
    has_screenshot: bool = Field(
        default=False, description="Whether screenshot is available"
    )
    browser_ready: bool = Field(
        default=False, description="Whether browser is ready"
    )
    previous_task_state: TaskState | None = Field(
        None, description="State of previous task if any"
    )
    active_tasks: int = Field(
        default=0, description="Number of active tasks"
    )
    viewport_size: tuple[int, int] | None = Field(
        None, description="Browser viewport dimensions"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )


class Constraints(BaseModel):
    """Detected constraints and limitations."""

    requires_navigation: bool = Field(
        default=False, description="Whether navigation to URL is needed"
    )
    requires_login: bool = Field(
        default=False, description="Whether login/auth is required"
    )
    has_time_sensitivity: bool = Field(
        default=False, description="Whether task is time-sensitive"
    )
    blocked_actions: list[str] = Field(
        default_factory=list, description="Actions that cannot be performed"
    )
    prerequisites: list[str] = Field(
        default_factory=list, description="Prerequisites that must be met"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about potential issues"
    )


class ResourceAssessment(BaseModel):
    """Assessment of available resources and capabilities."""

    available_actions: list[str] = Field(
        default_factory=list, description="Available action types"
    )
    has_vision: bool = Field(
        default=True, description="Whether vision/screenshot analysis is available"
    )
    has_dom_access: bool = Field(
        default=True, description="Whether DOM access is available"
    )
    has_skills_cache: bool = Field(
        default=False, description="Whether skills library has cached plans"
    )
    cached_skill_available: bool = Field(
        default=False, description="Whether a matching skill exists"
    )
    api_available: bool = Field(
        default=True, description="Whether LLM API is accessible"
    )
    estimated_capabilities: list[str] = Field(
        default_factory=list, description="List of estimated capabilities for this task"
    )


class Perceptor:
    """Analyzes tasks and gathers environmental context for planning.

    The Perceptor module is responsible for:
    - Parsing and understanding task descriptions
    - Gathering current environment context
    - Identifying constraints and limitations
    - Assessing available resources and capabilities
    """

    # Common action verbs for task analysis
    ACTION_VERBS = {
        "click", "tap", "press", "select", "choose",
        "type", "enter", "input", "write", "fill",
        "navigate", "go", "open", "visit", "browse",
        "create", "make", "add", "new",
        "delete", "remove", "clear", "erase",
        "edit", "update", "modify", "change",
        "search", "find", "look", "locate",
        "submit", "send", "save", "confirm",
        "download", "upload", "export", "import",
        "scroll", "drag", "drop", "hover",
        "login", "logout", "sign", "authenticate",
        "filter", "sort", "order", "arrange",
        "copy", "paste", "duplicate", "clone",
    }

    # Domain indicators
    DOMAIN_PATTERNS = {
        "notion": ["notion", "page", "database", "block"],
        "linear": ["linear", "issue", "project", "cycle"],
        "github": ["github", "repository", "repo", "pull request", "pr", "issue"],
        "slack": ["slack", "channel", "message", "dm"],
        "google": ["google", "gmail", "drive", "docs", "sheets"],
        "trello": ["trello", "board", "card", "list"],
        "asana": ["asana", "task", "project"],
        "jira": ["jira", "ticket", "sprint", "epic"],
    }

    def __init__(self, state_capturer: StateCapturer | None = None) -> None:
        """Initialize the perceptor.

        Args:
            state_capturer: State capturer for checking previous states
        """
        self._state_capturer = state_capturer
        self._page: Any = None
        self._last_analysis: TaskAnalysis | None = None
        self._last_context: EnvironmentContext | None = None

    def set_page(self, page: Any) -> None:
        """Set the Playwright page for context gathering.

        Args:
            page: Playwright page instance
        """
        self._page = page

    def set_state_capturer(self, state_capturer: StateCapturer) -> None:
        """Set the state capturer.

        Args:
            state_capturer: State capturer instance
        """
        self._state_capturer = state_capturer

    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Parse and understand a task description.

        Extracts intent, action verbs, targets, parameters, and
        estimates task complexity.

        Args:
            task_description: Natural language task description

        Returns:
            TaskAnalysis with parsed information

        Raises:
            ValueError: If task description is empty
        """
        if not task_description or not task_description.strip():
            raise ValueError("Task description cannot be empty")

        task = task_description.strip().lower()
        original = task_description.strip()

        # Extract action verbs
        action_verbs = self._extract_action_verbs(task)

        # Determine primary intent
        intent = self._determine_intent(task, action_verbs)

        # Extract target elements (quoted strings, UI element names)
        targets = self._extract_targets(original)

        # Extract parameters (values to input)
        parameters = self._extract_parameters(original)

        # Detect domain
        domain = self._detect_domain(task)

        # Estimate complexity
        complexity = self._estimate_complexity(task, action_verbs, targets)

        # Check if auth is likely required
        requires_auth = self._check_auth_requirement(task)

        # Extract important keywords
        keywords = self._extract_keywords(task)

        analysis = TaskAnalysis(
            original_task=original,
            intent=intent,
            action_verbs=action_verbs,
            target_elements=targets,
            parameters=parameters,
            domain=domain,
            complexity=complexity,
            requires_auth=requires_auth,
            keywords=keywords,
        )

        self._last_analysis = analysis
        return analysis

    async def get_context(self) -> EnvironmentContext:
        """Gather current environment context.

        Collects information about browser state, page info,
        and previous task states.

        Returns:
            EnvironmentContext with current state

        Raises:
            RuntimeError: If required components are not set
        """
        context = EnvironmentContext()

        # Browser context
        if self._page is not None:
            try:
                context.current_url = self._page.url
                context.page_title = await self._page.title()
                context.browser_ready = True

                viewport = self._page.viewport_size
                if viewport:
                    context.viewport_size = (viewport["width"], viewport["height"])
            except Exception:
                context.browser_ready = False

        # State capturer context
        if self._state_capturer is not None:
            context.has_screenshot = self._state_capturer.last_screenshot is not None
            context.active_tasks = len(self._state_capturer.get_all_tasks())

            # Check most recent task state
            tasks = self._state_capturer.get_all_tasks()
            if tasks:
                latest_task = tasks[-1]
                state_data = self._state_capturer.get_state(latest_task)
                if state_data and "state" in state_data:
                    try:
                        context.previous_task_state = TaskState(state_data["state"])
                    except ValueError:
                        pass

        self._last_context = context
        return context

    def identify_constraints(
        self, task_analysis: TaskAnalysis | None = None
    ) -> Constraints:
        """Detect limitations or requirements for a task.

        Args:
            task_analysis: Task analysis to check (uses last if None)

        Returns:
            Constraints with detected limitations

        Raises:
            ValueError: If no task analysis available
        """
        analysis = task_analysis or self._last_analysis
        if analysis is None:
            raise ValueError("No task analysis available. Call analyze_task() first.")

        constraints = Constraints()

        # Check for navigation requirement
        navigation_verbs = {"navigate", "go", "open", "visit", "browse"}
        if any(verb in analysis.action_verbs for verb in navigation_verbs):
            constraints.requires_navigation = True

        # Check for login requirement
        if analysis.requires_auth:
            constraints.requires_login = True
            constraints.prerequisites.append("User must be logged in")

        # Check for time sensitivity
        time_words = {"now", "immediately", "urgent", "asap", "today", "deadline"}
        if any(word in analysis.original_task.lower() for word in time_words):
            constraints.has_time_sensitivity = True
            constraints.warnings.append("Task appears to be time-sensitive")

        # Check for potentially destructive actions
        destructive_verbs = {"delete", "remove", "clear", "erase"}
        if any(verb in analysis.action_verbs for verb in destructive_verbs):
            constraints.warnings.append(
                "Task involves destructive action - confirm before executing"
            )

        # Check for multi-step complexity
        if analysis.complexity == "complex":
            constraints.warnings.append(
                "Complex task may require multiple re-planning attempts"
            )

        # Add prerequisites based on domain
        if analysis.domain:
            constraints.prerequisites.append(
                f"Must have access to {analysis.domain}"
            )

        return constraints

    def assess_resources(
        self,
        task_analysis: TaskAnalysis | None = None,
        skills_available: bool = False,
    ) -> ResourceAssessment:
        """Check available tools and capabilities.

        Args:
            task_analysis: Task analysis to assess (uses last if None)
            skills_available: Whether skills library has matching skills

        Returns:
            ResourceAssessment with capability information
        """
        assessment = ResourceAssessment()

        # Standard available actions
        assessment.available_actions = [
            "CLICK", "TYPE", "SELECT", "NAVIGATE",
            "SCROLL", "WAIT", "PRESS_KEY", "HOVER"
        ]

        # Check vision capability
        assessment.has_vision = True  # Always available via Perceptor

        # Check DOM access
        assessment.has_dom_access = self._page is not None

        # Check skills cache
        assessment.has_skills_cache = self._state_capturer is not None
        assessment.cached_skill_available = skills_available

        # Estimate capabilities based on task
        analysis = task_analysis or self._last_analysis
        if analysis:
            capabilities = []

            # Map action verbs to capabilities
            verb_capabilities = {
                "click": "Button/link interaction",
                "type": "Text input",
                "select": "Dropdown selection",
                "navigate": "Page navigation",
                "scroll": "Page scrolling",
                "search": "Search functionality",
                "create": "Content creation",
                "delete": "Content deletion",
                "upload": "File upload",
                "download": "File download",
            }

            for verb in analysis.action_verbs:
                if verb in verb_capabilities:
                    capabilities.append(verb_capabilities[verb])

            assessment.estimated_capabilities = list(set(capabilities))

        return assessment

    def _extract_action_verbs(self, task: str) -> list[str]:
        """Extract action verbs from task description."""
        words = re.findall(r'\b\w+\b', task)
        return [word for word in words if word in self.ACTION_VERBS]

    def _determine_intent(self, task: str, action_verbs: list[str]) -> str:
        """Determine primary intent from task and verbs."""
        if not action_verbs:
            return "unknown"

        # Priority order for intent
        intent_priority = [
            ("create", ["create", "make", "add", "new"]),
            ("delete", ["delete", "remove", "clear", "erase"]),
            ("update", ["edit", "update", "modify", "change"]),
            ("navigate", ["navigate", "go", "open", "visit"]),
            ("search", ["search", "find", "look", "locate"]),
            ("input", ["type", "enter", "input", "fill"]),
            ("interact", ["click", "tap", "press", "select"]),
        ]

        for intent, verbs in intent_priority:
            if any(v in action_verbs for v in verbs):
                return intent

        return action_verbs[0]

    def _extract_targets(self, task: str) -> list[str]:
        """Extract target UI elements from task."""
        targets = []

        # Extract quoted strings
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", task)
        targets.extend(quoted)

        # Common UI element patterns
        ui_patterns = [
            r"\b(button|link|input|field|dropdown|menu|tab|checkbox|radio)\b",
            r"\b(sidebar|header|footer|modal|dialog|popup)\b",
        ]

        for pattern in ui_patterns:
            matches = re.findall(pattern, task, re.IGNORECASE)
            targets.extend(matches)

        return list(set(targets))

    def _extract_parameters(self, task: str) -> dict[str, str]:
        """Extract input parameters from task."""
        params = {}

        # Pattern: "type X in Y" or "enter X into Y"
        type_match = re.search(
            r"(?:type|enter|input|write)\s+['\"]?([^'\"]+)['\"]?\s+(?:in|into|to)\s+",
            task, re.IGNORECASE
        )
        if type_match:
            params["input_value"] = type_match.group(1).strip()

        # Pattern: "named X" or "called X" or "titled X"
        name_match = re.search(
            r"(?:named|called|titled)\s+['\"]?([^'\"]+)['\"]?",
            task, re.IGNORECASE
        )
        if name_match:
            params["name"] = name_match.group(1).strip()

        # Extract URLs
        url_match = re.search(r"https?://\S+", task)
        if url_match:
            params["url"] = url_match.group(0)

        return params

    def _detect_domain(self, task: str) -> str | None:
        """Detect application domain from task."""
        for domain, keywords in self.DOMAIN_PATTERNS.items():
            if any(keyword in task for keyword in keywords):
                return domain
        return None

    def _estimate_complexity(
        self, task: str, verbs: list[str], targets: list[str]
    ) -> str:
        """Estimate task complexity."""
        # Count factors
        factors = 0

        # Multiple actions
        if len(verbs) > 2:
            factors += 1

        # Multiple targets
        if len(targets) > 2:
            factors += 1

        # Long description
        if len(task.split()) > 15:
            factors += 1

        # Contains conditionals
        if any(word in task for word in ["if", "then", "when", "unless"]):
            factors += 1

        # Contains sequences
        if any(word in task for word in ["first", "then", "after", "finally"]):
            factors += 1

        if factors >= 3:
            return "complex"
        elif factors >= 1:
            return "moderate"
        return "simple"

    def _check_auth_requirement(self, task: str) -> bool:
        """Check if task likely requires authentication."""
        auth_words = {
            "login", "logout", "sign in", "sign out", "authenticate",
            "my account", "my profile", "dashboard", "settings",
            "private", "personal"
        }
        return any(word in task for word in auth_words)

    def _extract_keywords(self, task: str) -> list[str]:
        """Extract important keywords from task."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "is", "are", "was", "were", "be",
            "this", "that", "these", "those", "it", "its"
        }

        words = re.findall(r'\b\w+\b', task)
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]

        return list(set(keywords))

    async def initialize(self) -> None:
        """Initialize the module."""
        pass

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        self._last_analysis = None
        self._last_context = None

    @property
    def is_ready(self) -> bool:
        """Check if module is ready for use."""
        return True

    @property
    def last_analysis(self) -> TaskAnalysis | None:
        """Get the last task analysis."""
        return self._last_analysis

    @property
    def last_context(self) -> EnvironmentContext | None:
        """Get the last environment context."""
        return self._last_context
