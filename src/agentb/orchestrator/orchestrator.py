"""Orchestrator - Central nervous system for planning, execution, and error recovery."""

import asyncio
import structlog

from agentb.core.config import Config
from agentb.core.types import (
    ActionType,
    ContextBundle,
    Coordinates,
    FailureInfo,
    Plan,
    PlanStep,
    SemanticRole,
    STATE_CHANGE_TIMEOUTS,
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
            logger.info(
                "skill_cache_hit",
                task=task,
                steps=len(plan.steps),
                step_list=[
                    f"Step {s.step}: {s.action.value} - {s.target_description}"
                    for s in plan.steps
                ]
            )

            # Check if we're on a blank page with a cached plan
            current_url = self.executor.get_current_url()
            if current_url == "about:blank":
                # Infer target URL from task
                target_url = self._infer_url_from_task(task)
                if target_url:
                    logger.info(
                        "auto_navigating_cached_skill",
                        target_url=target_url,
                        reason="cached_plan_on_blank_page"
                    )
                    await self.executor.navigate(target_url)
                    # Smarter wait for authenticated sessions
                    if self.config.storage_state:
                        # Authenticated session - just wait for DOM ready, not visual change
                        # This is much faster since the session loads instantly
                        await self.executor.page.wait_for_load_state("domcontentloaded")
                        logger.debug(
                            "navigation_fast_wait_authenticated",
                            reason="storage_state_provided"
                        )
                    else:
                        # Wait for full page load with visual change detection
                        await self.state_capturer.wait_for_change(timeout=5.0)
                else:
                    logger.warning(
                        "cannot_infer_url",
                        task=task,
                        message="Cached plan on blank page but couldn't infer URL"
                    )
        else:
            # Step 2: Generate initial plan with current URL context
            # Authentication is handled externally via storage_state from login.py
            logger.info("skill_cache_miss", task=task)
            current_url = self.executor.get_current_url()

            # Auto-navigate if on blank page (same as cache hit logic)
            if current_url == "about:blank":
                target_url = self._infer_url_from_task(task)
                if target_url:
                    logger.info(
                        "auto_navigating_cache_miss",
                        target_url=target_url,
                        reason="blank_page_before_planning"
                    )
                    await self.executor.navigate(target_url)
                    # Smarter wait for authenticated sessions
                    if self.config.storage_state:
                        await self.executor.page.wait_for_load_state("domcontentloaded")
                        logger.debug(
                            "navigation_fast_wait_authenticated",
                            reason="storage_state_provided"
                        )
                    else:
                        await self.state_capturer.wait_for_change(timeout=5.0)
                    # Update current_url after navigation
                    current_url = self.executor.get_current_url()
                else:
                    logger.warning(
                        "cannot_infer_url",
                        task=task,
                        message="Cache miss on blank page but couldn't infer URL"
                    )

            # Assume authenticated if storage_state is provided
            is_authenticated = self.config.storage_state is not None

            # Capture screenshot for vision-based planning (if not on blank page)
            screenshot = None
            if current_url != "about:blank":
                screenshot = await self.state_capturer.get_screenshot()
                logger.info(
                    "vision_planning_enabled",
                    reason="screenshot_captured_for_planning"
                )

            plan = await self.planner.generate_initial_plan(
                task,
                screenshot=screenshot,
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
            # Step 5: Save/update skill with ACTUAL successful execution path
            # Build a plan from the steps that actually succeeded
            if self._executed_steps:
                # Create a plan from successful steps, renumbering them sequentially
                # Include cached coordinates for performance optimization on future runs
                successful_plan = Plan(
                    task=task,
                    steps=[
                        PlanStep(
                            step=i + 1,
                            action=step.action,
                            target_description=step.target_description,
                            value=step.value,
                            semantic_role=step.semantic_role,
                            required_state=step.required_state,
                            cached_coordinates=step.cached_coordinates,  # Cache coords for speed
                        )
                        for i, step in enumerate(self._executed_steps)
                    ]
                )

                # Check if we already have a cached skill
                existing_skill_result = self.skills_library.find_skill_with_id(task)

                if existing_skill_result:
                    skill_id, existing_plan = existing_skill_result
                    # Update: Delete old skill and add new successful path
                    logger.info(
                        "updating_cached_skill",
                        task=task,
                        old_steps=len(existing_plan.steps),
                        new_steps=len(successful_plan.steps)
                    )
                    # Delete the specific old skill
                    self.skills_library.delete_skill(skill_id)
                    # Add the new successful path
                    self.skills_library.add_skill(task, successful_plan)
                    logger.info("skill_updated", task=task)
                else:
                    # New skill: Just add it
                    self.skills_library.add_skill(task, successful_plan)
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
        # Guard: Skip all actions on blank page - will trigger re-plan with navigation
        current_url = self.executor.get_current_url()
        if current_url == "about:blank":
            logger.warning(
                "step_skipped_blank_page",
                step=step.step,
                action=step.action.value,
                reason="cannot_execute_on_blank_page"
            )
            return False

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

            # Cache the coordinates for future reuse (if not already cached)
            if not step.cached_coordinates:
                step.cached_coordinates = coords
                logger.debug(
                    "caching_coordinates_for_step",
                    step=step.step,
                    x=coords.x,
                    y=coords.y,
                )

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

        # Adaptive timeout and optional wait based on action type
        # Navigation actions need longer waits, modal interactions are faster
        if step.action == ActionType.NAVIGATE:
            # Full page navigation - wait with navigation timeout
            timeout = STATE_CHANGE_TIMEOUTS.get(step.action, 10.0)
            await self.state_capturer.wait_for_change(timeout=timeout)
        elif step.semantic_role == SemanticRole.NAVIGATION:
            # Links/buttons that navigate to new pages
            timeout = 5.0
            await self.state_capturer.wait_for_change(timeout=timeout)
        else:
            # Modal/dialog/form interactions - use action-specific short timeout
            timeout = STATE_CHANGE_TIMEOUTS.get(step.action, 3.0)
            # Try waiting for change, but don't fail if timeout
            changed = await self.state_capturer.wait_for_change(timeout=timeout)
            if not changed:
                # No visual change detected, just wait brief moment for UI update
                logger.debug(
                    "no_visual_change_brief_pause",
                    step=step.step,
                    action=step.action.value
                )
                await asyncio.sleep(0.5)

        # Capture state after action
        await self.state_capturer.capture_state(f"after_step_{step.step}")

        logger.info("step_completed", step=step.step)
        return True

    async def _find_element(self, step: PlanStep) -> Coordinates | None:
        """Find element using cached-coords-first, DOM, then vision-fallback strategy.

        Args:
            step: Step containing target description

        Returns:
            Coordinates if found, None otherwise
        """
        # Guard: Skip element search on blank pages
        current_url = self.executor.get_current_url()
        if current_url == "about:blank":
            logger.warning(
                "element_search_skipped_blank_page",
                target=step.target_description,
                message="Cannot search for elements on blank page"
            )
            return None

        # OPTIMIZATION: Try cached coordinates first (from previous successful execution)
        if step.cached_coordinates:
            logger.debug(
                "trying_cached_coordinates",
                target=step.target_description,
                x=step.cached_coordinates.x,
                y=step.cached_coordinates.y,
            )
            # Verify element still exists at cached location with quick check
            is_valid = await self.executor.verify_element_at(step.cached_coordinates)
            if is_valid:
                logger.info(
                    "element_found_by_cached_coords",
                    target=step.target_description,
                    x=step.cached_coordinates.x,
                    y=step.cached_coordinates.y,
                )
                return step.cached_coordinates
            else:
                logger.debug(
                    "cached_coords_invalid_falling_back",
                    target=step.target_description,
                )

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
            # Planner already logged plan_regenerated_with_vision
            # Just add step_list for CLI display
            logger.info(
                "replan_ready",
                new_steps=len(new_plan.steps),
                step_list=[
                    f"Step {s.step}: {s.action.value} - {s.target_description}"
                    for s in new_plan.steps
                ]
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

    def _infer_url_from_task(self, task: str) -> str | None:
        """Infer starting URL from task description.

        Args:
            task: Task description

        Returns:
            URL string or None if cannot infer
        """
        task_lower = task.lower()

        # Common service mappings
        url_mappings = {
            "notion": "https://www.notion.so",
            "github": "https://github.com",
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "twitter": "https://twitter.com",
            "linkedin": "https://www.linkedin.com",
            "slack": "https://slack.com",
            "trello": "https://trello.com",
            "asana": "https://app.asana.com",
            "linear": "https://linear.app",
        }

        # Check for each service name in task
        for service, url in url_mappings.items():
            if service in task_lower:
                return url

        # If no match found, return None
        return None

    @property
    def current_plan(self) -> Plan | None:
        """Get the current execution plan."""
        return self._current_plan

    @property
    def executed_steps(self) -> list[PlanStep]:
        """Get the list of successfully executed steps."""
        return self._executed_steps.copy()
