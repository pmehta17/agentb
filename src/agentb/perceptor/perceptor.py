"""Perceptor - Vision-based search for UI elements."""

import base64
import json
import re

import anthropic
import structlog

from agentb.core.config import Config
from agentb.core.types import Coordinates, FailureInfo, PlanStep


logger = structlog.get_logger()


class Perceptor:
    """Vision-based UI element detection using multimodal LLM."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the perceptor.

        Args:
            config: Application configuration
        """
        self.config = config or Config()
        self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

    async def find_element(
        self, screenshot: bytes, step: PlanStep
    ) -> Coordinates | FailureInfo:
        """Find UI element in screenshot using vision.

        Args:
            screenshot: Screenshot as PNG bytes
            step: Plan step containing target description

        Returns:
            Coordinates if found, FailureInfo if not found
        """
        # Encode screenshot as base64
        image_base64 = base64.standard_b64encode(screenshot).decode("utf-8")

        prompt = f"""Analyze this screenshot and find the UI element described below.

Target Element: {step.target_description}
Semantic Role: {step.semantic_role.value}
Required State: {step.required_state}

Instructions:
1. Carefully examine the screenshot
2. Locate the UI element that matches the target description
3. Return the CENTER coordinates (x, y) of the element

If found, respond with ONLY a JSON object:
{{"found": true, "x": <number>, "y": <number>, "confidence": <0.0-1.0>}}

If NOT found, respond with:
{{"found": false, "reason": "<explanation of why element was not found>"}}

Important:
- Coordinates should be the CENTER of the element
- Be precise - wrong coordinates will cause click failures
- Consider the semantic role when identifying the correct element
- If multiple similar elements exist, choose the most likely one based on context
"""

        try:
            response = self.client.messages.create(
                model=self.config.perceptor_model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            # Parse response
            response_text = response.content[0].text
            result = self._parse_response(response_text)

            if result.get("found"):
                coords = Coordinates(
                    x=int(result["x"]),
                    y=int(result["y"]),
                )
                logger.info(
                    "element_found_by_vision",
                    target=step.target_description,
                    x=coords.x,
                    y=coords.y,
                    confidence=result.get("confidence", 1.0),
                )
                return coords
            else:
                reason = result.get("reason", "Element not found in screenshot")
                logger.warning(
                    "element_not_found_by_vision",
                    target=step.target_description,
                    reason=reason,
                )
                return FailureInfo(
                    step=step,
                    error_type="element_not_found",
                    error_message=reason,
                    screenshot=screenshot,
                )

        except Exception as e:
            logger.error(
                "perceptor_error",
                target=step.target_description,
                error=str(e),
            )
            return FailureInfo(
                step=step,
                error_type="perceptor_error",
                error_message=str(e),
                screenshot=screenshot,
            )

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON response from the model.

        Args:
            response_text: Raw response text

        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        try:
            # First try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return failure if no JSON found
        return {"found": False, "reason": "Failed to parse model response"}

    async def explain_failure(
        self, screenshot: bytes, step: PlanStep, error: str
    ) -> str:
        """Get explanation for why an action failed.

        Args:
            screenshot: Screenshot showing the failure state
            step: The step that failed
            error: Error message from the failure

        Returns:
            Human-readable explanation of the failure
        """
        image_base64 = base64.standard_b64encode(screenshot).decode("utf-8")

        prompt = f"""Analyze this screenshot and explain why the following action failed.

Action: {step.action.value}
Target: {step.target_description}
Error: {error}

Provide a brief explanation of:
1. What the current UI state shows
2. Why the target element might not be available
3. What state or action might be needed first

Keep the explanation concise (2-3 sentences).
"""

        try:
            response = self.client.messages.create(
                model=self.config.perceptor_model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            return response.content[0].text

        except Exception as e:
            logger.error("explain_failure_error", error=str(e))
            return f"Failed to analyze screenshot: {e}"
