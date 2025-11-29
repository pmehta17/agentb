"""Unit tests for Orchestrator module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentb.core.config import Config
from agentb.core.types import (
    ActionType,
    Coordinates,
    FailureInfo,
    Plan,
    PlanStep,
    SemanticRole,
)
from agentb.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def sample_plan() -> Plan:
    """Create a sample Plan for testing.

    Returns:
        Plan with two steps
    """
    return Plan(
        task="Complete form",
        steps=[
            PlanStep(
                step=1,
                action=ActionType.TYPE,
                target_description="Email input",
                value="test@example.com",
                semantic_role=SemanticRole.FORM_FIELD,
                required_state="Form visible"
            ),
            PlanStep(
                step=2,
                action=ActionType.CLICK,
                target_description="Submit button",
                value=None,
                semantic_role=SemanticRole.PRIMARY_ACTION,
                required_state="Form filled"
            )
        ]
    )


class TestOrchestrator:
    """Test suite for Orchestrator class."""

    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    def test_init_creates_all_modules(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that __init__ initializes all modules."""
        orchestrator = Orchestrator(test_config)

        # Verify all modules initialized
        mock_executor_class.assert_called_once_with(config=test_config)
        mock_planner_class.assert_called_once_with(config=test_config)
        mock_perceptor_class.assert_called_once_with(config=test_config)
        mock_skills_library_class.assert_called_once_with(config=test_config)

        # Verify state initialized
        assert orchestrator.config == test_config
        assert orchestrator._current_plan is None
        assert orchestrator._executed_steps == []
        assert orchestrator._replan_count == 0

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_start_initializes_browser_and_state_capturer(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that start() initializes browser and StateCapturer."""
        mock_executor = AsyncMock()
        mock_page = MagicMock()
        mock_executor.start = AsyncMock(return_value=mock_page)
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        # Start orchestrator
        await orchestrator.start()

        # Verify browser started
        mock_executor.start.assert_called_once()

        # Verify StateCapturer created
        mock_state_capturer_class.assert_called_once_with(mock_page, config=test_config)

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_stop_closes_browser(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that stop() closes browser."""
        mock_executor = AsyncMock()
        mock_executor.stop = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        # Stop orchestrator
        await orchestrator.stop()

        # Verify browser stopped
        mock_executor.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_execute_task_with_skill_cache_hit(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan,
        sample_image_bytes: bytes
    ) -> None:
        """Test that execute_task() uses cached skill when available."""
        # Setup mocks
        mock_skills_library = MagicMock()
        mock_skills_library.find_skill = MagicMock(return_value=sample_plan)
        mock_skills_library_class.return_value = mock_skills_library

        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=Coordinates(x=100, y=200))
        mock_executor.type_text = AsyncMock()
        mock_executor.click = AsyncMock()
        mock_executor_class.return_value = mock_executor

        mock_planner = AsyncMock()
        mock_planner.validate_success = AsyncMock(return_value=True)
        mock_planner_class.return_value = mock_planner

        mock_state_capturer = AsyncMock()
        mock_state_capturer.capture_state = AsyncMock()
        mock_state_capturer.wait_for_change = AsyncMock(return_value=True)
        mock_state_capturer.get_screenshot = AsyncMock(return_value=sample_image_bytes)

        orchestrator = Orchestrator(test_config)
        orchestrator.state_capturer = mock_state_capturer

        # Execute task
        success = await orchestrator.execute_task("Complete form")

        # Verify cache was checked
        mock_skills_library.find_skill.assert_called_with("Complete form")

        # Verify planner was NOT called (cache hit)
        mock_planner.generate_initial_plan.assert_not_called()

        # Verify task succeeded
        assert success is True

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_execute_task_with_skill_cache_miss(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan,
        sample_image_bytes: bytes
    ) -> None:
        """Test that execute_task() generates plan on cache miss."""
        # Setup mocks
        mock_skills_library = MagicMock()
        mock_skills_library.find_skill = MagicMock(return_value=None)  # Cache miss
        mock_skills_library.add_skill = MagicMock()
        mock_skills_library_class.return_value = mock_skills_library

        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=Coordinates(x=100, y=200))
        mock_executor.type_text = AsyncMock()
        mock_executor.click = AsyncMock()
        mock_executor.get_current_url = MagicMock(return_value="https://example.com")
        mock_executor_class.return_value = mock_executor

        mock_planner = AsyncMock()
        mock_planner.generate_initial_plan = AsyncMock(return_value=sample_plan)
        mock_planner.validate_success = AsyncMock(return_value=True)
        mock_planner_class.return_value = mock_planner

        mock_state_capturer = AsyncMock()
        mock_state_capturer.capture_state = AsyncMock()
        mock_state_capturer.wait_for_change = AsyncMock(return_value=True)
        mock_state_capturer.get_screenshot = AsyncMock(return_value=sample_image_bytes)

        orchestrator = Orchestrator(test_config)
        orchestrator.state_capturer = mock_state_capturer

        # Execute task
        success = await orchestrator.execute_task("Complete form")

        # Verify planner was called with current_url and is_authenticated (cache miss)
        mock_planner.generate_initial_plan.assert_called_once_with(
            "Complete form", current_url="https://example.com", is_authenticated=False
        )

        # Verify skill was saved
        mock_skills_library.add_skill.assert_called_once_with("Complete form", sample_plan)

        # Verify task succeeded
        assert success is True

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_execute_task_saves_skill_only_on_success(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan,
        sample_image_bytes: bytes
    ) -> None:
        """Test that execute_task() only saves skill when task validates successfully."""
        # Setup mocks
        mock_skills_library = MagicMock()
        mock_skills_library.find_skill = MagicMock(return_value=None)
        mock_skills_library.add_skill = MagicMock()
        mock_skills_library_class.return_value = mock_skills_library

        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=Coordinates(x=100, y=200))
        mock_executor.type_text = AsyncMock()
        mock_executor.click = AsyncMock()
        mock_executor_class.return_value = mock_executor

        mock_planner = AsyncMock()
        mock_planner.generate_initial_plan = AsyncMock(return_value=sample_plan)
        mock_planner.validate_success = AsyncMock(return_value=False)  # Validation fails
        mock_planner_class.return_value = mock_planner

        mock_state_capturer = AsyncMock()
        mock_state_capturer.capture_state = AsyncMock()
        mock_state_capturer.wait_for_change = AsyncMock(return_value=True)
        mock_state_capturer.get_screenshot = AsyncMock(return_value=sample_image_bytes)

        orchestrator = Orchestrator(test_config)
        orchestrator.state_capturer = mock_state_capturer

        # Execute task
        success = await orchestrator.execute_task("Complete form")

        # Verify skill was NOT saved (validation failed)
        mock_skills_library.add_skill.assert_not_called()

        # Verify task failed
        assert success is False

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_find_element_uses_dom_first(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _find_element() tries DOM search first."""
        # Setup mocks
        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=Coordinates(x=150, y=250))
        mock_executor_class.return_value = mock_executor

        mock_perceptor = AsyncMock()
        mock_perceptor_class.return_value = mock_perceptor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.CLICK,
            target_description="Button",
            value=None,
            semantic_role=SemanticRole.PRIMARY_ACTION,
            required_state="Ready"
        )

        # Find element
        coords = await orchestrator._find_element(step)

        # Verify DOM search was called
        mock_executor.find_element_by_text.assert_called_once_with("Button")

        # Verify perceptor was NOT called (DOM succeeded)
        mock_perceptor.find_element.assert_not_called()

        # Verify correct coordinates returned
        assert coords.x == 150
        assert coords.y == 250

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_find_element_falls_back_to_vision(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that _find_element() falls back to vision when DOM fails."""
        # Setup mocks
        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=None)  # DOM fails
        mock_executor.get_screenshot = AsyncMock(return_value=sample_image_bytes)
        mock_executor_class.return_value = mock_executor

        mock_perceptor = AsyncMock()
        mock_perceptor.find_element = AsyncMock(return_value=Coordinates(x=300, y=400))
        mock_perceptor_class.return_value = mock_perceptor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.CLICK,
            target_description="Button",
            value=None,
            semantic_role=SemanticRole.PRIMARY_ACTION,
            required_state="Ready"
        )

        # Find element
        coords = await orchestrator._find_element(step)

        # Verify DOM search was tried
        mock_executor.find_element_by_text.assert_called_once_with("Button")

        # Verify vision search was called (fallback)
        mock_perceptor.find_element.assert_called_once_with(sample_image_bytes, step)

        # Verify correct coordinates returned
        assert coords.x == 300
        assert coords.y == 400

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_find_element_returns_none_when_both_fail(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that _find_element() returns None when both DOM and vision fail."""
        # Setup mocks
        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=None)
        mock_executor.get_screenshot = AsyncMock(return_value=sample_image_bytes)
        mock_executor_class.return_value = mock_executor

        step = PlanStep(
            step=1,
            action=ActionType.CLICK,
            target_description="Button",
            value=None,
            semantic_role=SemanticRole.PRIMARY_ACTION,
            required_state="Ready"
        )

        mock_perceptor = AsyncMock()
        failure_info = FailureInfo(
            step=step,
            error_type="not_found",
            error_message="Element not visible",
            screenshot=sample_image_bytes
        )
        mock_perceptor.find_element = AsyncMock(return_value=failure_info)
        mock_perceptor_class.return_value = mock_perceptor

        orchestrator = Orchestrator(test_config)

        # Find element
        coords = await orchestrator._find_element(step)

        # Verify None returned
        assert coords is None

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_perform_action_click(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _perform_action() executes CLICK action."""
        mock_executor = AsyncMock()
        mock_executor.click = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.CLICK,
            target_description="Button",
            value=None,
            semantic_role=SemanticRole.PRIMARY_ACTION,
            required_state="Ready"
        )

        coords = Coordinates(x=100, y=200)

        # Perform action
        await orchestrator._perform_action(step, coords)

        # Verify click was called
        mock_executor.click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_perform_action_type(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _perform_action() executes TYPE action."""
        mock_executor = AsyncMock()
        mock_executor.type_text = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.TYPE,
            target_description="Input",
            value="Hello",
            semantic_role=SemanticRole.FORM_FIELD,
            required_state="Ready"
        )

        coords = Coordinates(x=50, y=100)

        # Perform action
        await orchestrator._perform_action(step, coords)

        # Verify type_text was called
        mock_executor.type_text.assert_called_once_with(50, 100, "Hello")

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_perform_action_select(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _perform_action() executes SELECT action."""
        mock_executor = AsyncMock()
        mock_executor.select = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.SELECT,
            target_description="Dropdown",
            value="Option A",
            semantic_role=SemanticRole.FORM_FIELD,
            required_state="Ready"
        )

        coords = Coordinates(x=150, y=200)

        # Perform action
        await orchestrator._perform_action(step, coords)

        # Verify select was called
        mock_executor.select.assert_called_once_with(150, 200, "Option A")

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_perform_action_navigate(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _perform_action() executes NAVIGATE action."""
        mock_executor = AsyncMock()
        mock_executor.navigate = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        step = PlanStep(
            step=1,
            action=ActionType.NAVIGATE,
            target_description="URL",
            value="https://example.com",
            semantic_role=SemanticRole.NAVIGATION,
            required_state="Ready"
        )

        coords = Coordinates(x=0, y=0)

        # Perform action
        await orchestrator._perform_action(step, coords)

        # Verify navigate was called
        mock_executor.navigate.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_perform_action_raises_on_missing_value(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _perform_action() raises ValueError when value is missing."""
        mock_executor = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        # TYPE without value
        step = PlanStep(
            step=1,
            action=ActionType.TYPE,
            target_description="Input",
            value=None,  # Missing!
            semantic_role=SemanticRole.FORM_FIELD,
            required_state="Ready"
        )

        coords = Coordinates(x=0, y=0)

        # Should raise ValueError
        with pytest.raises(ValueError, match="TYPE action requires a value"):
            await orchestrator._perform_action(step, coords)

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_execute_step_captures_state_before_and_after(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _execute_step() captures state before and after action."""
        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=Coordinates(x=100, y=200))
        mock_executor.click = AsyncMock()
        mock_executor_class.return_value = mock_executor

        mock_state_capturer = AsyncMock()
        mock_state_capturer.capture_state = AsyncMock()
        mock_state_capturer.wait_for_change = AsyncMock(return_value=True)

        orchestrator = Orchestrator(test_config)
        orchestrator.state_capturer = mock_state_capturer

        step = PlanStep(
            step=1,
            action=ActionType.CLICK,
            target_description="Button",
            value=None,
            semantic_role=SemanticRole.PRIMARY_ACTION,
            required_state="Ready"
        )

        # Execute step
        success = await orchestrator._execute_step(step)

        # Verify state captured before and after
        assert mock_state_capturer.capture_state.call_count == 2
        mock_state_capturer.capture_state.assert_any_call("before_step_1")
        mock_state_capturer.capture_state.assert_any_call("after_step_1")

        # Verify wait for change was called
        mock_state_capturer.wait_for_change.assert_called_once()

        assert success is True

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_replan_creates_context_bundle(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan,
        sample_image_bytes: bytes
    ) -> None:
        """Test that _replan() creates proper ContextBundle."""
        mock_executor = AsyncMock()
        mock_executor.get_screenshot = AsyncMock(return_value=sample_image_bytes)
        mock_executor_class.return_value = mock_executor

        mock_planner = AsyncMock()
        new_plan = Plan(task="Complete form", steps=[])
        mock_planner.regenerate_plan = AsyncMock(return_value=new_plan)
        mock_planner_class.return_value = mock_planner

        orchestrator = Orchestrator(test_config)
        orchestrator._current_plan = sample_plan
        orchestrator._executed_steps = [sample_plan.steps[0]]

        failed_step = sample_plan.steps[1]

        # Replan
        result_plan = await orchestrator._replan(failed_step)

        # Verify regenerate_plan was called
        mock_planner.regenerate_plan.assert_called_once()

        # Verify ContextBundle structure
        context = mock_planner.regenerate_plan.call_args[0][0]
        assert context.goal == "Complete form"
        assert len(context.plan_history) == 1
        assert context.failure.step == failed_step
        assert context.failure.error_type == "element_not_found"

        assert result_plan == new_plan

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_execute_plan_respects_max_retries(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that _execute_plan() respects max_retries limit."""
        # Set max_retries to 2
        test_config.max_retries = 2

        plan = Plan(
            task="Test",
            steps=[
                PlanStep(
                    step=1,
                    action=ActionType.CLICK,
                    target_description="Button",
                    value=None,
                    semantic_role=SemanticRole.PRIMARY_ACTION,
                    required_state="Ready"
                )
            ]
        )

        mock_executor = AsyncMock()
        mock_executor.find_element_by_text = AsyncMock(return_value=None)  # Always fails
        mock_executor.get_screenshot = AsyncMock(return_value=b"screenshot")
        mock_executor_class.return_value = mock_executor

        mock_perceptor = AsyncMock()
        mock_perceptor.find_element = AsyncMock(return_value=FailureInfo(
            step=plan.steps[0],
            error_type="not_found",
            error_message="Not found",
            screenshot=b"screenshot"
        ))
        mock_perceptor_class.return_value = mock_perceptor

        mock_state_capturer = AsyncMock()
        mock_state_capturer.capture_state = AsyncMock()

        orchestrator = Orchestrator(test_config)
        orchestrator.state_capturer = mock_state_capturer
        orchestrator._current_plan = plan  # Set current plan so _replan can access it

        # Execute plan (will fail and hit retry limit)
        success = await orchestrator._execute_plan(plan)

        # Verify failed
        assert success is False
        assert orchestrator._replan_count == 0  # Never actually replanned (failed at first step)

    @pytest.mark.asyncio
    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    async def test_navigate_to_convenience_method(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config
    ) -> None:
        """Test that navigate_to() calls executor.navigate()."""
        mock_executor = AsyncMock()
        mock_executor.navigate = AsyncMock()
        mock_executor_class.return_value = mock_executor

        orchestrator = Orchestrator(test_config)

        # Navigate
        await orchestrator.navigate_to("https://example.com")

        # Verify executor.navigate was called
        mock_executor.navigate.assert_called_once_with("https://example.com")

    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    def test_current_plan_property(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan
    ) -> None:
        """Test that current_plan property returns current plan."""
        orchestrator = Orchestrator(test_config)

        # Initially None
        assert orchestrator.current_plan is None

        # Set plan
        orchestrator._current_plan = sample_plan

        # Verify property returns plan
        assert orchestrator.current_plan == sample_plan

    @patch("agentb.orchestrator.orchestrator.SkillsLibrary")
    @patch("agentb.orchestrator.orchestrator.StateCapturer")
    @patch("agentb.orchestrator.orchestrator.Perceptor")
    @patch("agentb.orchestrator.orchestrator.Planner")
    @patch("agentb.orchestrator.orchestrator.Executor")
    def test_executed_steps_property(
        self,
        mock_executor_class,
        mock_planner_class,
        mock_perceptor_class,
        mock_state_capturer_class,
        mock_skills_library_class,
        test_config: Config,
        sample_plan: Plan
    ) -> None:
        """Test that executed_steps property returns copy of executed steps."""
        orchestrator = Orchestrator(test_config)

        # Initially empty
        assert orchestrator.executed_steps == []

        # Add steps
        orchestrator._executed_steps = [sample_plan.steps[0]]

        # Verify property returns copy
        executed = orchestrator.executed_steps
        assert len(executed) == 1
        assert executed[0] == sample_plan.steps[0]

        # Verify it's a copy (modifying doesn't affect internal state)
        executed.append(sample_plan.steps[1])
        assert len(orchestrator._executed_steps) == 1
