"""Orchestrator - Central nervous system for planning, execution, and error recovery."""

import structlog

from agentb.core.config import Config
from agentb.core.types import (
    ActionType,
    ContextBundle,
    Coordinates,
    FailureInfo,
    Plan,
    PlanStep,
)
from agentb.executor import Executor
from agentb.perceptor import Perceptor
from agentb.planner import Planner
from agentb.skills_library import SkillsLibrary
from agentb.state_capturer import StateCapturer


logger = structlog.get_logger()


class Orchestrator:
    """Central orchestrator coordinating all agent modules."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the orchestrator.

        Args:
            config: Application configuration
        """
        self.config = config or Config()

        # Initialize modules
        self.executor = Executor(config=self.config)
        self.planner = Planner(config=self.config)
        self.perceptor = Perceptor(config=self.config)
        self.skills_library = SkillsLibrary(config=self.config)
        self.state_capturer: StateCapturer | None = None

        # Execution state
        self._current_plan: Plan | None = None
        self._executed_steps: list[PlanStep] = []
        self._replan_count = 0

    async def start(self) -> None:
        """Start the orchestrator and browser."""
        page = await self.executor.start()
        self.state_capturer = StateCapturer(page, config=self.config)
        logger.info("orchestrator_started")

    async def stop(self) -> None:
        """Stop the orchestrator and cleanup resources."""
        await self.executor.stop()
        logger.info("orchestrator_stopped")

    async def execute_task(self, task: str) -> bool:
        """Execute a natural language task.

        Args:
            task: Natural language task description

        Returns:
            True if task completed successfully, False otherwise
        """
        logger.info("task_received", task=task)

        # Reset execution state
        self._executed_steps = []
        self._replan_count = 0

        # Step 1: Query Skills Library
        plan = self.skills_library.find_skill(task)

        if plan:
            logger.info("skill_cache_hit", task=task)
        else:
            # Step 2: Detect authentication status if storage_state is provided
            is_authenticated = False
            if self.config.storage_state:
                logger.info("checking_authentication_status")
                is_authenticated = await self._detect_authenticated_state()
                if is_authenticated:
                    logger.info("authenticated_state_detected",
                               reason="workspace_ui_detected")

            # Step 3: Generate initial plan with current URL context and auth status
            logger.info("skill_cache_miss", task=task)
            current_url = self.executor.get_current_url()
            plan = await self.planner.generate_initial_plan(
                task,
                current_url=current_url,
                is_authenticated=is_authenticated
            )

        self._current_plan = plan

        # Step 3: Execute plan
        success = await self._execute_plan(plan)

        if not success:
            return False

        # Step 4: Validate success
        screenshot = await self.state_capturer.get_screenshot()
        is_valid = await self.planner.validate_success(task, screenshot)

        if is_valid:
            # Step 5: Save new skill (if not from cache)
            if not self.skills_library.find_skill(task):
                self.skills_library.add_skill(task, plan)
                logger.info("skill_saved", task=task)

            logger.info("task_completed", task=task, success=True)
            return True
        else:
            logger.warning("task_validation_failed", task=task)
            return False

    async def _execute_plan(self, plan: Plan) -> bool:
        """Execute all steps in a plan.

        Args:
            plan: Plan to execute

        Returns:
            True if all steps succeeded, False otherwise
        """
        for step in plan.steps:
            success = await self._execute_step(step)

            if not success:
                # Attempt re-planning
                if self._replan_count >= self.config.max_retries:
                    logger.error(
                        "max_replans_reached",
                        count=self._replan_count,
                    )
                    return False

                # Re-plan and retry
                new_plan = await self._replan(step)
                if new_plan:
                    self._replan_count += 1
                    return await self._execute_plan(new_plan)
                else:
                    return False

            self._executed_steps.append(step)

        return True

    async def _execute_step(self, step: PlanStep) -> bool:
        """Execute a single plan step.

        Args:
            step: Step to execute

        Returns:
            True if step succeeded, False otherwise
        """
        # Safety check: Skip login-related steps if storage_state was provided
        if self.config.storage_state:
            login_keywords = [
                "login", "log in", "sign in", "signin",
                "enter credentials", "username", "password",
                "authenticate", "enter email"
            ]
            target_lower = step.target_description.lower()
            if any(keyword in target_lower for keyword in login_keywords):
                logger.warning(
                    "login_step_skipped",
                    reason="storage_state_provided",
                    step=step.step,
                    target=step.target_description,
                )
                # Return True to continue execution
                return True

        logger.info(
            "executing_step",
            step=step.step,
            action=step.action.value,
            target=step.target_description,
        )

        # Capture state before action
        await self.state_capturer.capture_state(f"before_step_{step.step}")

        # NAVIGATE actions don't need element coordinates
        if step.action == ActionType.NAVIGATE:
            try:
                if step.value is None:
                    raise ValueError("NAVIGATE action requires a URL in the value field")

                # Check if we're already on the target domain
                current_url = self.executor.get_current_url()
                from urllib.parse import urlparse
                current_domain = urlparse(current_url).netloc
                target_domain = urlparse(step.value).netloc

                if current_domain == target_domain:
                    logger.info(
                        "navigation_skipped",
                        reason="already_on_domain",
                        current_url=current_url,
                        target_url=step.value,
                    )
                    # Still mark as successful, just skip the actual navigation
                else:
                    await self.executor.navigate(step.value)
            except Exception as e:
                logger.error(
                    "action_failed",
                    step=step.step,
                    error=str(e),
                )
                return False
        else:
            # Find element coordinates for other actions
            coords = await self._find_element(step)

            if coords is None:
                return False

            # Execute action
            try:
                await self._perform_action(step, coords)
            except Exception as e:
                logger.error(
                    "action_failed",
                    step=step.step,
                    error=str(e),
                )
                return False

        # Wait for state change
        await self.state_capturer.wait_for_change()

        # Capture state after action
        await self.state_capturer.capture_state(f"after_step_{step.step}")

        logger.info("step_completed", step=step.step)
        return True

    async def _find_element(self, step: PlanStep) -> Coordinates | None:
        """Find element using DOM-first, vision-fallback strategy.

        Args:
            step: Step containing target description

        Returns:
            Coordinates if found, None otherwise
        """
        # DOM-first: Try to find by text
        coords = await self.executor.find_element_by_text(step.target_description)

        if coords:
            logger.debug(
                "element_found_by_dom",
                target=step.target_description,
            )
            return coords

        # Vision fallback
        logger.debug(
            "dom_search_failed_trying_vision",
            target=step.target_description,
        )

        screenshot = await self.executor.get_screenshot()
        result = await self.perceptor.find_element(screenshot, step)

        if isinstance(result, Coordinates):
            return result
        else:
            # FailureInfo returned
            logger.warning(
                "vision_search_failed",
                target=step.target_description,
                error=result.error_message,
            )
            return None

    async def _perform_action(self, step: PlanStep, coords: Coordinates) -> None:
        """Perform the action specified in a step.

        Args:
            step: Step with action to perform
            coords: Target coordinates
        """
        match step.action:
            case ActionType.CLICK:
                await self.executor.click(coords.x, coords.y)

            case ActionType.TYPE:
                if step.value is None:
                    raise ValueError("TYPE action requires a value")
                await self.executor.type_text(coords.x, coords.y, step.value)

            case ActionType.SELECT:
                if step.value is None:
                    raise ValueError("SELECT action requires a value")
                await self.executor.select(coords.x, coords.y, step.value)

            case ActionType.NAVIGATE:
                if step.value is None:
                    raise ValueError("NAVIGATE action requires a URL")
                await self.executor.navigate(step.value)

    async def _replan(self, failed_step: PlanStep) -> Plan | None:
        """Generate a new plan after failure.

        Args:
            failed_step: The step that failed

        Returns:
            New plan or None if re-planning failed
        """
        logger.info(
            "replanning",
            failed_step=failed_step.step,
            attempt=self._replan_count + 1,
        )

        screenshot = await self.executor.get_screenshot()

        # Build context bundle
        context = ContextBundle(
            goal=self._current_plan.task,
            plan_history=self._executed_steps.copy(),
            failure=FailureInfo(
                step=failed_step,
                error_type="element_not_found",
                error_message=f"Could not find element: {failed_step.target_description}",
                screenshot=screenshot,
            ),
            current_screenshot=screenshot,
        )

        try:
            new_plan = await self.planner.regenerate_plan(context)
            logger.info(
                "replan_generated",
                new_steps=len(new_plan.steps),
            )
            return new_plan
        except Exception as e:
            logger.error("replan_failed", error=str(e))
            return None

    async def navigate_to(self, url: str) -> None:
        """Navigate to a URL (convenience method).

        Args:
            url: URL to navigate to
        """
        await self.executor.navigate(url)

    async def _detect_authenticated_state(self) -> bool:
        """Detect if user is already authenticated by checking for workspace UI.

        Returns:
            True if authenticated workspace UI is detected, False otherwise
        """
        try:
            screenshot = await self.state_capturer.get_screenshot()

            # Use perceptor to check for authenticated workspace indicators
            from agentb.core.types import PlanStep

            # Create a dummy step to check for workspace UI elements
            check_step = PlanStep(
                step=0,
                action=ActionType.CLICK,  # Dummy action, won't be used
                target_description="workspace sidebar or user profile menu or workspace name",
                value=None,
                semantic_role="primary_action",
                required_state="authenticated_workspace",
            )

            result = await self.perceptor.find_element(screenshot, check_step)

            # If coordinates returned, workspace UI was found
            if isinstance(result, Coordinates):
                return True
            else:
                # Check the failure message for authenticated state indicators
                if isinstance(result, FailureInfo):
                    message = result.error_message.lower()
                    # Look for indicators that user is already logged in
                    authenticated_indicators = [
                        "already logged in",
                        "workspace",
                        "authenticated",
                        "user is logged in",
                        "dashboard",
                    ]
                    if any(indicator in message for indicator in authenticated_indicators):
                        return True
                return False
        except Exception as e:
            logger.debug("authentication_check_failed", error=str(e))
            # If check fails, assume not authenticated to be safe
            return False

    @property
    def current_plan(self) -> Plan | None:
        """Get the current execution plan."""
        return self._current_plan

    @property
    def executed_steps(self) -> list[PlanStep]:
        """Get the list of successfully executed steps."""
        return self._executed_steps.copy()
