"""Unit tests for Planner module."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from agentb.core.config import Config
from agentb.core.types import (
    ActionType,
    ContextBundle,
    FailureInfo,
    Plan,
    PlanStep,
    SemanticRole,
)
from agentb.planner.planner import Planner


@pytest.fixture
def sample_plan_json() -> str:
    """Create sample plan JSON response.

    Returns:
        JSON string representing a plan
    """
    return json.dumps([
        {
            "step": 1,
            "action": "NAVIGATE",
            "target_description": "Google homepage",
            "value": "https://google.com",
            "semantic_role": "navigation",
            "required_state": "Browser ready"
        },
        {
            "step": 2,
            "action": "TYPE",
            "target_description": "Search input box",
            "value": "Python tutorials",
            "semantic_role": "form_field",
            "required_state": "Homepage loaded"
        }
    ])


@pytest.fixture
def sample_context_bundle() -> ContextBundle:
    """Create a sample ContextBundle for testing.

    Returns:
        ContextBundle with failure context
    """
    failed_step = PlanStep(
        step=2,
        action=ActionType.CLICK,
        target_description="Submit button",
        value=None,
        semantic_role=SemanticRole.PRIMARY_ACTION,
        required_state="Form visible"
    )

    executed_step = PlanStep(
        step=1,
        action=ActionType.NAVIGATE,
        target_description="Homepage",
        value="https://example.com",
        semantic_role=SemanticRole.NAVIGATION,
        required_state="Browser ready"
    )

    failure = FailureInfo(
        step=failed_step,
        error_type="element_not_found",
        error_message="Submit button not visible",
        screenshot=b"fake screenshot"
    )

    return ContextBundle(
        goal="Complete the registration form",
        plan_history=[executed_step],
        failure=failure,
        current_screenshot=b"current screenshot"
    )


class TestPlanner:
    """Test suite for Planner class."""

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_init_creates_client(
        self, mock_anthropic, test_config: Config
    ) -> None:
        """Test that __init__ creates Anthropic client."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # Verify client created
        mock_anthropic.assert_called_once_with(api_key=test_config.anthropic_api_key)
        assert planner.client == mock_client
        assert planner.config == test_config

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_init_with_default_config(self, mock_anthropic) -> None:
        """Test that Planner initializes with default config."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner()

        # Verify default config created
        assert planner.config is not None
        assert isinstance(planner.config, Config)

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_returns_plan(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str
    ) -> None:
        """Test that generate_initial_plan() returns a Plan."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Generate plan
        task = "Search for Python tutorials"
        plan = await planner.generate_initial_plan(task)

        # Verify Plan returned
        assert isinstance(plan, Plan)
        assert plan.task == task
        assert len(plan.steps) == 2

        # Verify first step
        assert plan.steps[0].step == 1
        assert plan.steps[0].action == ActionType.NAVIGATE
        assert plan.steps[0].target_description == "Google homepage"
        assert plan.steps[0].value == "https://google.com"
        assert plan.steps[0].semantic_role == SemanticRole.NAVIGATION

        # Verify API was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == test_config.planner_model
        assert call_kwargs["max_tokens"] == 2048
        assert task in call_kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_handles_json_in_markdown(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str
    ) -> None:
        """Test that generate_initial_plan() extracts JSON from markdown."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response with markdown
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = f"Here is the plan:\n```json\n{sample_plan_json}\n```"
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Generate plan
        plan = await planner.generate_initial_plan("Test task")

        # Verify plan extracted
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_handles_exception(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that generate_initial_plan() raises exception on API error."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        planner = Planner(test_config)

        # Should raise exception
        with pytest.raises(Exception, match="API error"):
            await planner.generate_initial_plan("Test task")

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_regenerate_plan_returns_plan(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str,
        sample_context_bundle: ContextBundle
    ) -> None:
        """Test that regenerate_plan() returns a corrected Plan."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Regenerate plan
        plan = await planner.regenerate_plan(sample_context_bundle)

        # Verify Plan returned
        assert isinstance(plan, Plan)
        assert plan.task == sample_context_bundle.goal
        assert len(plan.steps) == 2

        # Verify API was called with context
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]

        # Verify prompt includes failure context
        assert sample_context_bundle.goal in prompt
        assert "element_not_found" in prompt
        assert "Submit button not visible" in prompt

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_regenerate_plan_includes_executed_steps(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str,
        sample_context_bundle: ContextBundle
    ) -> None:
        """Test that regenerate_plan() includes executed steps in prompt."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Regenerate plan
        await planner.regenerate_plan(sample_context_bundle)

        # Verify executed steps in prompt
        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]

        assert "NAVIGATE" in prompt
        assert "Homepage" in prompt

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_regenerate_plan_handles_empty_history(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str,
        sample_context_bundle: ContextBundle
    ) -> None:
        """Test that regenerate_plan() handles empty plan history."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Empty history
        sample_context_bundle.plan_history = []

        # Regenerate plan
        plan = await planner.regenerate_plan(sample_context_bundle)

        # Should still work
        assert isinstance(plan, Plan)

        # Verify prompt indicates no steps executed
        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "(none)" in prompt

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_validate_success_returns_true_when_successful(
        self,
        mock_anthropic,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that validate_success() returns True for successful task."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock success response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"success": true, "reason": "Task completed successfully"}'
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Validate success
        result = await planner.validate_success("Complete form", sample_image_bytes)

        # Verify True returned
        assert result is True

        # Verify API called with image and text
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]

        message = call_kwargs["messages"][0]
        assert len(message["content"]) == 2

        # Verify image content
        image_content = message["content"][0]
        assert image_content["type"] == "image"
        expected_base64 = base64.standard_b64encode(sample_image_bytes).decode("utf-8")
        assert image_content["source"]["data"] == expected_base64

        # Verify text content
        text_content = message["content"][1]
        assert "Complete form" in text_content["text"]

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_validate_success_returns_false_when_failed(
        self,
        mock_anthropic,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that validate_success() returns False for failed task."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock failure response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"success": false, "reason": "Error message visible"}'
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Validate success
        result = await planner.validate_success("Submit form", sample_image_bytes)

        # Verify False returned
        assert result is False

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_validate_success_returns_false_on_exception(
        self,
        mock_anthropic,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that validate_success() returns False on API exception."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        planner = Planner(test_config)

        # Validate success (should not raise)
        result = await planner.validate_success("Test task", sample_image_bytes)

        # Verify False returned
        assert result is False

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_parse_plan_response_with_valid_json(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str
    ) -> None:
        """Test that _parse_plan_response() parses valid JSON."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # Parse plan
        steps = planner._parse_plan_response(sample_plan_json)

        # Verify steps
        assert len(steps) == 2
        assert all(isinstance(s, PlanStep) for s in steps)
        assert steps[0].action == ActionType.NAVIGATE
        assert steps[1].action == ActionType.TYPE

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_extract_json_array_from_plain_text(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str
    ) -> None:
        """Test that _extract_json_array() extracts array from text."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # Extract from text with surrounding content
        text = f"Here is the plan:\n{sample_plan_json}\nThat's all."
        extracted = planner._extract_json_array(text)

        # Verify array extracted
        assert extracted == sample_plan_json

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_extract_json_array_returns_text_if_no_array(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that _extract_json_array() returns stripped text if no array."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # No array in text
        text = "   No JSON here   "
        extracted = planner._extract_json_array(text)

        # Verify stripped text returned
        assert extracted == "No JSON here"

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_parse_json_response_with_valid_json(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that _parse_json_response() parses valid JSON."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # Parse valid JSON
        result = planner._parse_json_response('{"success": true, "reason": "Done"}')

        # Verify parsed
        assert result["success"] is True
        assert result["reason"] == "Done"

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_parse_json_response_extracts_from_text(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that _parse_json_response() extracts JSON from surrounding text."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # JSON embedded in text
        text = 'Result: {"success": false, "reason": "Failed"} - end'
        result = planner._parse_json_response(text)

        # Verify extracted
        assert result["success"] is False
        assert result["reason"] == "Failed"

    @patch("agentb.planner.planner.anthropic.Anthropic")
    def test_parse_json_response_returns_failure_on_invalid_json(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that _parse_json_response() returns failure dict on invalid JSON."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        planner = Planner(test_config)

        # Invalid JSON
        result = planner._parse_json_response("Not valid JSON at all")

        # Verify fallback
        assert result["success"] is False
        assert "Failed to parse" in result["reason"]

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_with_single_step(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that generate_initial_plan() handles single-step plans."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Single step plan
        single_step_json = json.dumps([{
            "step": 1,
            "action": "CLICK",
            "target_description": "Button",
            "value": None,
            "semantic_role": "primary_action",
            "required_state": "Ready"
        }])

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = single_step_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Generate plan
        plan = await planner.generate_initial_plan("Click button")

        # Verify single step
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.CLICK

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_with_all_action_types(
        self,
        mock_anthropic,
        test_config: Config
    ) -> None:
        """Test that generate_initial_plan() handles all action types."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Plan with all action types
        all_actions_json = json.dumps([
            {
                "step": 1,
                "action": "NAVIGATE",
                "target_description": "URL",
                "value": "https://example.com",
                "semantic_role": "navigation",
                "required_state": "Ready"
            },
            {
                "step": 2,
                "action": "CLICK",
                "target_description": "Button",
                "value": None,
                "semantic_role": "primary_action",
                "required_state": "Page loaded"
            },
            {
                "step": 3,
                "action": "TYPE",
                "target_description": "Input",
                "value": "Text",
                "semantic_role": "form_field",
                "required_state": "Input visible"
            },
            {
                "step": 4,
                "action": "SELECT",
                "target_description": "Dropdown",
                "value": "Option",
                "semantic_role": "form_field",
                "required_state": "Dropdown visible"
            }
        ])

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = all_actions_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Generate plan
        plan = await planner.generate_initial_plan("Test all actions")

        # Verify all action types
        assert len(plan.steps) == 4
        assert plan.steps[0].action == ActionType.NAVIGATE
        assert plan.steps[1].action == ActionType.CLICK
        assert plan.steps[2].action == ActionType.TYPE
        assert plan.steps[3].action == ActionType.SELECT

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_validate_success_handles_missing_fields(
        self,
        mock_anthropic,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that validate_success() handles missing JSON fields."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Response missing success field
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"reason": "Unknown"}'
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        planner = Planner(test_config)

        # Validate (should default to False)
        result = await planner.validate_success("Test", sample_image_bytes)

        # Verify False (missing success defaults to False)
        assert result is False

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_generate_initial_plan_uses_correct_model(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str
    ) -> None:
        """Test that generate_initial_plan() uses configured model."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        # Set specific model
        test_config.planner_model = "claude-opus-4-20250514"

        planner = Planner(test_config)

        # Generate plan
        await planner.generate_initial_plan("Test")

        # Verify correct model used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_regenerate_plan_uses_correct_model(
        self,
        mock_anthropic,
        test_config: Config,
        sample_plan_json: str,
        sample_context_bundle: ContextBundle
    ) -> None:
        """Test that regenerate_plan() uses configured model."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = sample_plan_json
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        # Set specific model
        test_config.planner_model = "claude-opus-4-20250514"

        planner = Planner(test_config)

        # Regenerate plan
        await planner.regenerate_plan(sample_context_bundle)

        # Verify correct model used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    @patch("agentb.planner.planner.anthropic.Anthropic")
    async def test_validate_success_uses_correct_model(
        self,
        mock_anthropic,
        test_config: Config,
        sample_image_bytes: bytes
    ) -> None:
        """Test that validate_success() uses configured model."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"success": true, "reason": "OK"}'
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        # Set specific model
        test_config.planner_model = "claude-opus-4-20250514"

        planner = Planner(test_config)

        # Validate
        await planner.validate_success("Test", sample_image_bytes)

        # Verify correct model used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"
