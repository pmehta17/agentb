"""Planner - High-level reasoning with LLM."""

import base64
import json
import re

import anthropic
import structlog

from agentb.core.config import Config
from agentb.core.types import (
    ActionType,
    ContextBundle,
    Plan,
    PlanStep,
    SemanticRole,
)


logger = structlog.get_logger()


PLAN_SCHEMA = """
[
  {
    "step": 1,
    "action": "CLICK", // CLICK, TYPE, SELECT, NAVIGATE
    "target_description": "the 'New Project' button in the left sidebar",
    "value": null, // Required for TYPE (text), SELECT (option), and NAVIGATE (URL). null for CLICK.
    "semantic_role": "primary_action", // primary_action, navigation, confirmation, form_field
    "required_state": "projects_list_visible"
  }
]
"""


class Planner:
    """High-level reasoning module using text-only LLM."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the planner.

        Args:
            config: Application configuration
        """
        self.config = config or Config()
        self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

    async def generate_initial_plan(
        self,
        task: str,
        current_url: str | None = None,
        is_authenticated: bool = False
    ) -> Plan:
        """Generate an initial execution plan for a task.

        Args:
            task: Natural language task description
            current_url: Current browser URL (if available)
            is_authenticated: Whether user is already authenticated (via storage_state)

        Returns:
            Structured execution plan
        """
        context_info = ""
        if current_url:
            context_info = f"\nCurrent Browser State:\n- Currently on: {current_url}\n- You may already be on the target site, so check if navigation is needed\n"

        auth_info = ""
        if is_authenticated:
            auth_info = """
⚠️ CRITICAL - USER IS ALREADY AUTHENTICATED:
- Storage state was loaded, user has valid login session
- Workspace UI elements (sidebar, profile, workspace name) are already visible
- DO NOT generate ANY login-related steps (sign in, log in, enter credentials, etc.)
- START DIRECTLY with the actual task - skip all authentication steps
- The user is already on their authenticated workspace/dashboard
"""

        prompt = f"""You are an expert at planning browser automation tasks. Generate a step-by-step plan to accomplish the following task.

Task: {task}
{context_info}
{auth_info}
Requirements:
1. Break down the task into discrete, atomic actions
2. Use ONLY these action types: CLICK, TYPE, SELECT, NAVIGATE
3. Provide clear target descriptions that can be found in a UI
4. Consider the logical order of operations (e.g., navigate before click)
5. Include necessary waits/confirmations
6. If already on the target website (check current URL), skip navigation and start with the actual task
7. If user is authenticated (see auth info above), NEVER include login/sign-in steps

Output Format:
Return ONLY a JSON array of steps following this schema:
{PLAN_SCHEMA}

Important:
- Each step must be independently executable
- target_description should be specific enough to uniquely identify the element
- value is required for TYPE (text to type), SELECT (option to select), and NAVIGATE (URL to navigate to). null for CLICK actions.
- For NAVIGATE actions, value must be a complete URL (e.g., "https://notion.so" or "https://google.com")
- If current URL shows you're already on the target site, DO NOT add a NAVIGATE step
- If user is authenticated, DO NOT add any login/sign-in steps - start with the actual task
- semantic_role helps prioritize element matching
- required_state describes what must be visible before this step

Generate the plan:
"""

        try:
            response = self.client.messages.create(
                model=self.config.planner_model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            response_text = response.content[0].text
            steps = self._parse_plan_response(response_text)

            plan = Plan(task=task, steps=steps)
            logger.info(
                "plan_generated",
                task=task,
                steps=len(steps),
                step_list=[
                    f"Step {s.step}: {s.action.value} - {s.target_description}"
                    for s in steps
                ]
            )

            return plan

        except Exception as e:
            logger.error("plan_generation_failed", task=task, error=str(e))
            raise

    async def regenerate_plan(self, context: ContextBundle) -> Plan:
        """Regenerate plan using failure context with visual awareness.

        Args:
            context: Bundle containing goal, history, failure info, and screenshot

        Returns:
            Corrected execution plan
        """
        # Format executed steps
        executed_steps = "\n".join(
            f"  Step {s.step}: {s.action.value} on '{s.target_description}'"
            for s in context.plan_history
        )

        # Determine if we should continue from current state or restart
        has_progress = len(context.plan_history) > 0

        if has_progress:
            continuation_guidance = f"""
CRITICAL: The steps listed above were ALREADY COMPLETED SUCCESSFULLY.
DO NOT repeat these steps. DO NOT navigate away from the current page/state.
The screenshot shows the CURRENT state AFTER those successful steps.

Your job is to CONTINUE from this point forward to complete the goal.
- If a modal/dialog is open, work within it
- If you're already on the target page, don't navigate again
- Start your plan from where the previous plan left off"""
        else:
            continuation_guidance = """
This is the first attempt. Generate a complete plan from the beginning."""

        prompt = f"""You are an expert at planning browser automation tasks. A previous plan failed and needs correction.

Original Goal: {context.goal}

Steps Executed Successfully:
{executed_steps if context.plan_history else "  (none)"}
{continuation_guidance}

Failed Step:
  Step {context.failure.step.step}: {context.failure.step.action.value} on '{context.failure.step.target_description}'
  Error Type: {context.failure.error_type}
  Error Message: {context.failure.error_message}

IMPORTANT: You are provided with a screenshot showing the CURRENT state of the browser.
Analyze this screenshot carefully to understand:
1. What UI elements are currently visible
2. What state the application is in (modal open, page loaded, etc.)
3. Why the failed step couldn't find its target element
4. What the next logical steps should be given what's ACTUALLY on screen

Requirements:
1. Consider why the original step failed based on what you see
2. The plan should CONTINUE from the current state (preserve progress!)
3. Do NOT restart or re-navigate unless absolutely necessary
4. Use ONLY these action types: CLICK, TYPE, SELECT, NAVIGATE
5. Base your plan on what's VISIBLE in the screenshot, not assumptions
6. If elements from completed steps are visible (open modals, etc), work within them

Output Format:
Return ONLY a JSON array of steps following this schema:
{PLAN_SCHEMA}

Important:
- value is required for TYPE (text to type), SELECT (option to select), and NAVIGATE (URL to navigate to). null for CLICK actions.
- For NAVIGATE actions, value must be a complete URL (e.g., "https://notion.so" or "https://google.com")
- target_description should be specific enough to uniquely identify the element
- PRESERVE PROGRESS: Only use NAVIGATE if the current screen shows you're not on the right page
- If a modal/dialog is open (visible in screenshot), continue working in it

Generate the corrected continuation plan based on what you SEE in the screenshot:
"""

        try:
            # Encode screenshot for vision model
            image_base64 = base64.standard_b64encode(context.current_screenshot).decode("utf-8")

            # Use vision-capable model for re-planning
            response = self.client.messages.create(
                model=self.config.planner_model,
                max_tokens=2048,
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

            response_text = response.content[0].text
            steps = self._parse_plan_response(response_text)

            plan = Plan(task=context.goal, steps=steps)
            logger.info(
                "plan_regenerated_with_vision",
                task=context.goal,
                steps=len(steps),
                failed_step=context.failure.step.step,
            )

            return plan

        except Exception as e:
            logger.error(
                "plan_regeneration_failed",
                task=context.goal,
                error=str(e),
            )
            raise

    async def validate_success(self, goal: str, screenshot: bytes) -> bool:
        """Validate if the task was completed successfully.

        Args:
            goal: Original task goal
            screenshot: Final screenshot after execution

        Returns:
            True if task appears successful, False otherwise
        """
        image_base64 = base64.standard_b64encode(screenshot).decode("utf-8")

        prompt = f"""Analyze this screenshot and determine if the following task was completed successfully.

Task: {goal}

Examine the screenshot for:
1. Visual indicators of success (confirmation messages, new content, etc.)
2. The expected end state for this type of task
3. Any error messages or unexpected states

Respond with ONLY a JSON object:
{{"success": true/false, "reason": "<brief explanation>"}}
"""

        try:
            response = self.client.messages.create(
                model=self.config.planner_model,
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

            response_text = response.content[0].text
            result = self._parse_json_response(response_text)

            success = result.get("success", False)
            reason = result.get("reason", "Unknown")

            logger.info(
                "validation_complete",
                goal=goal,
                success=success,
                reason=reason,
            )

            return success

        except Exception as e:
            logger.error("validation_failed", goal=goal, error=str(e))
            return False

    def _parse_plan_response(self, response_text: str) -> list[PlanStep]:
        """Parse plan steps from model response.

        Args:
            response_text: Raw response text

        Returns:
            List of PlanStep objects
        """
        # Try to extract JSON array from response
        json_str = self._extract_json_array(response_text)
        steps_data = json.loads(json_str)

        steps = []
        for step_data in steps_data:
            step = PlanStep(
                step=step_data["step"],
                action=ActionType(step_data["action"]),
                target_description=step_data["target_description"],
                value=step_data.get("value"),
                semantic_role=SemanticRole(step_data["semantic_role"]),
                required_state=step_data["required_state"],
            )
            steps.append(step)

        return steps

    def _extract_json_array(self, text: str) -> str:
        """Extract JSON array from text.

        Args:
            text: Text potentially containing JSON

        Returns:
            JSON array string
        """
        # Try to find JSON array in response
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group()

        # If no array found, try the whole text
        return text.strip()

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON object from response.

        Args:
            response_text: Raw response text

        Returns:
            Parsed dictionary
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        match = re.search(r"\{[^}]+\}", response_text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"success": False, "reason": "Failed to parse response"}
