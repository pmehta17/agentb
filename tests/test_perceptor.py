"""Unit tests for Perceptor module."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from agentb.core.config import Config
from agentb.core.types import ActionType, Coordinates, FailureInfo, PlanStep, SemanticRole
from agentb.perceptor.perceptor import Perceptor


@pytest.fixture
def sample_step() -> PlanStep:
    """Create a sample PlanStep for testing.

    Returns:
        PlanStep for clicking a button
    """
    return PlanStep(
        step=1,
        action=ActionType.CLICK,
        target_description="Submit button",
        value=None,
        semantic_role=SemanticRole.PRIMARY_ACTION,
        required_state="Form filled",
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client.

    Returns:
        MagicMock configured to simulate Anthropic client
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = MagicMock()
    return client


@pytest.fixture
def mock_response_found():
    """Create a mock Anthropic response for element found.

    Returns:
        Mock response with found=true JSON
    """
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = '{"found": true, "x": 500, "y": 300, "confidence": 0.95}'
    response.content = [content_block]
    return response


@pytest.fixture
def mock_response_not_found():
    """Create a mock Anthropic response for element not found.

    Returns:
        Mock response with found=false JSON
    """
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = '{"found": false, "reason": "Element not visible in screenshot"}'
    response.content = [content_block]
    return response


class TestPerceptor:
    """Test suite for Perceptor class."""

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_init_creates_client(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that __init__ creates Anthropic client."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Verify client created
        mock_anthropic.assert_called_once_with(api_key=test_config.anthropic_api_key)
        assert perceptor.client == mock_anthropic_client
        assert perceptor.config == test_config

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_init_with_default_config(self, mock_anthropic, mock_anthropic_client) -> None:
        """Test that Perceptor initializes with default config."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor()

        # Verify default config created
        assert perceptor.config is not None
        assert isinstance(perceptor.config, Config)

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_returns_coordinates_when_found(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        mock_response_found,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() returns Coordinates when element found."""
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_response_found

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify Coordinates returned
        assert isinstance(result, Coordinates)
        assert result.x == 500
        assert result.y == 300

        # Verify API was called with correct parameters
        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]

        assert call_kwargs["model"] == test_config.perceptor_model
        assert call_kwargs["max_tokens"] == 256
        assert len(call_kwargs["messages"]) == 1

        # Verify message structure
        message = call_kwargs["messages"][0]
        assert message["role"] == "user"
        assert len(message["content"]) == 2

        # Verify image content
        image_content = message["content"][0]
        assert image_content["type"] == "image"
        assert image_content["source"]["type"] == "base64"
        assert image_content["source"]["media_type"] == "image/png"

        # Verify image is base64 encoded
        expected_base64 = base64.standard_b64encode(sample_image_bytes).decode("utf-8")
        assert image_content["source"]["data"] == expected_base64

        # Verify text content includes step details
        text_content = message["content"][1]
        assert text_content["type"] == "text"
        assert "Submit button" in text_content["text"]
        assert "primary_action" in text_content["text"]  # Enum value is lowercase
        assert "Form filled" in text_content["text"]

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_returns_failure_info_when_not_found(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        mock_response_not_found,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() returns FailureInfo when element not found."""
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_response_not_found

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify FailureInfo returned
        assert isinstance(result, FailureInfo)
        assert result.step == sample_step
        assert result.error_type == "element_not_found"
        assert result.error_message == "Element not visible in screenshot"
        assert result.screenshot == sample_image_bytes

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_handles_api_exception(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() handles API exceptions gracefully."""
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.side_effect = Exception("API error")

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify FailureInfo returned with exception details
        assert isinstance(result, FailureInfo)
        assert result.step == sample_step
        assert result.error_type == "perceptor_error"
        assert "API error" in result.error_message
        assert result.screenshot == sample_image_bytes

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_parse_response_with_valid_json(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that _parse_response() parses valid JSON."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Parse valid JSON
        result = perceptor._parse_response('{"found": true, "x": 100, "y": 200}')

        # Verify parsed correctly
        assert result["found"] is True
        assert result["x"] == 100
        assert result["y"] == 200

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_parse_response_with_json_in_text(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that _parse_response() extracts JSON from surrounding text."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Parse JSON embedded in text
        response_text = 'Here is the result: {"found": false, "reason": "Not visible"} as you can see.'
        result = perceptor._parse_response(response_text)

        # Verify JSON extracted
        assert result["found"] is False
        assert result["reason"] == "Not visible"

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_parse_response_with_invalid_json(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that _parse_response() handles invalid JSON."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Parse invalid JSON
        result = perceptor._parse_response("This is not valid JSON at all")

        # Verify fallback response
        assert result["found"] is False
        assert "Failed to parse model response" in result["reason"]

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_parse_response_with_malformed_json(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that _parse_response() handles malformed JSON."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Parse malformed JSON
        result = perceptor._parse_response('{"found": true, "x": 100')

        # Verify fallback response
        assert result["found"] is False
        assert "Failed to parse model response" in result["reason"]

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_with_json_in_markdown(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() handles JSON wrapped in markdown."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response with markdown formatting
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = 'Sure! Here is the result:\n```json\n{"found": true, "x": 250, "y": 150, "confidence": 0.9}\n```'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify Coordinates returned (JSON extracted from markdown)
        assert isinstance(result, Coordinates)
        assert result.x == 250
        assert result.y == 150

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_converts_to_integers(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() converts coordinates to integers."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response with float coordinates
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"found": true, "x": 123.7, "y": 456.2}'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify coordinates are integers
        assert isinstance(result, Coordinates)
        assert isinstance(result.x, int)
        assert isinstance(result.y, int)
        assert result.x == 123
        assert result.y == 456

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_with_missing_confidence(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() handles missing confidence field."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response without confidence
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"found": true, "x": 400, "y": 600}'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify Coordinates returned (confidence is optional)
        assert isinstance(result, Coordinates)
        assert result.x == 400
        assert result.y == 600

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_with_default_reason(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() uses default reason when missing."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response without reason
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"found": false}'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Find element
        result = await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify default reason used
        assert isinstance(result, FailureInfo)
        assert result.error_message == "Element not found in screenshot"

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_explain_failure_returns_explanation(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that explain_failure() returns explanation text."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response with explanation
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = "The submit button is disabled because the form validation failed. You need to fill in the required email field first."
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Explain failure
        explanation = await perceptor.explain_failure(
            sample_image_bytes, sample_step, "Element not clickable"
        )

        # Verify explanation returned
        assert "submit button is disabled" in explanation
        assert "form validation failed" in explanation

        # Verify API was called
        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]

        # Verify message includes failure details
        message = call_kwargs["messages"][0]
        text_content = message["content"][1]
        assert "CLICK" in text_content["text"]
        assert "Submit button" in text_content["text"]
        assert "Element not clickable" in text_content["text"]

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_explain_failure_handles_exception(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that explain_failure() handles API exceptions."""
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.side_effect = Exception("API timeout")

        perceptor = Perceptor(test_config)

        # Explain failure
        explanation = await perceptor.explain_failure(
            sample_image_bytes, sample_step, "Error"
        )

        # Verify error message returned
        assert "Failed to analyze screenshot" in explanation
        assert "API timeout" in explanation

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_with_different_step_types(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() works with different action types."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"found": true, "x": 100, "y": 200}'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Test with TYPE action
        type_step = PlanStep(
            step=1,
            action=ActionType.TYPE,
            target_description="Email input field",
            value="test@example.com",
            semantic_role=SemanticRole.FORM_FIELD,
            required_state="Login form visible",
        )

        result = await perceptor.find_element(sample_image_bytes, type_step)

        # Verify coordinates returned
        assert isinstance(result, Coordinates)

        # Verify prompt includes TYPE action details
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        text_content = call_kwargs["messages"][0]["content"][1]["text"]
        assert "Email input field" in text_content
        assert "form_field" in text_content  # Enum value is lowercase

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_encodes_different_image_formats(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
    ) -> None:
        """Test that find_element() properly encodes image bytes."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = '{"found": true, "x": 100, "y": 200}'
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        perceptor = Perceptor(test_config)

        # Test with custom bytes
        custom_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        result = await perceptor.find_element(custom_bytes, sample_step)

        # Verify image was base64 encoded correctly
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        image_data = call_kwargs["messages"][0]["content"][0]["source"]["data"]

        expected_base64 = base64.standard_b64encode(custom_bytes).decode("utf-8")
        assert image_data == expected_base64

    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    def test_parse_response_with_nested_json(
        self, mock_anthropic, test_config: Config, mock_anthropic_client
    ) -> None:
        """Test that _parse_response() handles simple nested structures."""
        mock_anthropic.return_value = mock_anthropic_client

        perceptor = Perceptor(test_config)

        # Parse JSON with nested structure (though not deeply nested)
        result = perceptor._parse_response('{"found": true, "x": 100, "y": 200, "metadata": "info"}')

        # Verify parsed correctly
        assert result["found"] is True
        assert result["x"] == 100

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_find_element_uses_correct_model(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        mock_response_found,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that find_element() uses the configured model."""
        mock_anthropic.return_value = mock_anthropic_client
        mock_anthropic_client.messages.create.return_value = mock_response_found

        # Set specific model in config
        test_config.perceptor_model = "claude-opus-4-20250514"

        perceptor = Perceptor(test_config)

        # Find element
        await perceptor.find_element(sample_image_bytes, sample_step)

        # Verify correct model used
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    @patch("agentb.perceptor.perceptor.anthropic.Anthropic")
    async def test_explain_failure_uses_correct_model(
        self,
        mock_anthropic,
        test_config: Config,
        mock_anthropic_client,
        sample_step: PlanStep,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that explain_failure() uses the configured model."""
        mock_anthropic.return_value = mock_anthropic_client

        # Mock response
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = "Explanation text"
        response.content = [content_block]
        mock_anthropic_client.messages.create.return_value = response

        # Set specific model
        test_config.perceptor_model = "claude-opus-4-20250514"

        perceptor = Perceptor(test_config)

        # Explain failure
        await perceptor.explain_failure(sample_image_bytes, sample_step, "Error")

        # Verify correct model used
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"
