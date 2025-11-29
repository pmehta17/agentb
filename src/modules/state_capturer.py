"""State Capturer module - Captures and manages task execution state."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict

from PIL import Image

from src.types.common import StateSnapshot, TaskState
from src.types.modules import IStateCapturer


class StateCapturer:
    """Captures and manages task execution state with history tracking.

    This module handles:
    - Capturing UI state snapshots (screenshots)
    - Tracking task execution state
    - Maintaining state history for each task
    - Detecting UI state changes through pixel comparison
    """

    def __init__(
        self,
        screenshots_dir: str | Path = "./data/screenshots",
        pixelmatch_threshold: float = 0.02,
        state_change_timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Initialize the state capturer.

        Args:
            screenshots_dir: Directory to save screenshots
            pixelmatch_threshold: Minimum diff ratio to detect state change
            state_change_timeout: Max seconds to wait for state change
            poll_interval: Seconds between state change polls
        """
        self._screenshots_dir = Path(screenshots_dir)
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._pixelmatch_threshold = pixelmatch_threshold
        self._state_change_timeout = state_change_timeout
        self._poll_interval = poll_interval

        # In-memory storage for task states
        self._states: dict[str, dict[str, Any]] = {}
        self._state_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Last captured screenshot
        self._last_screenshot: bytes | None = None

        # Page reference (set externally)
        self._page: Any = None

    def set_page(self, page: Any) -> None:
        """Set the Playwright page for screenshot capture.

        Args:
            page: Playwright page instance
        """
        self._page = page

    async def capture_state(self, step_name: str) -> StateSnapshot:
        """Capture current UI state with screenshot.

        Args:
            step_name: Descriptive name for this state

        Returns:
            StateSnapshot with screenshot and metadata

        Raises:
            RuntimeError: If page is not set
        """
        if self._page is None:
            raise RuntimeError("Page not set. Call set_page() first.")

        timestamp = datetime.now()
        screenshot = await self._page.screenshot()

        # Generate filename
        time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in step_name)
        filename = f"{time_str}_{safe_name}.png"
        filepath = self._screenshots_dir / filename

        # Save screenshot
        filepath.write_bytes(screenshot)
        self._last_screenshot = screenshot

        # Get page info
        url = self._page.url
        title = await self._page.title()

        snapshot = StateSnapshot(
            timestamp=timestamp,
            screenshot=screenshot,
            url=url,
            title=title,
            step_name=step_name,
            filepath=str(filepath),
        )

        return snapshot

    async def get_screenshot(self) -> bytes:
        """Get current screenshot as bytes.

        Returns:
            Screenshot as PNG bytes

        Raises:
            RuntimeError: If page is not set
        """
        if self._page is None:
            raise RuntimeError("Page not set. Call set_page() first.")

        screenshot = await self._page.screenshot()
        self._last_screenshot = screenshot
        return screenshot

    async def wait_for_change(self, timeout: float | None = None) -> bool:
        """Wait for UI state to change.

        Captures a 'before' screenshot, waits for network idle,
        then polls for visual changes using pixel comparison.

        Args:
            timeout: Max seconds to wait (uses default if None)

        Returns:
            True if state changed, False if timeout reached

        Raises:
            RuntimeError: If page is not set
        """
        if self._page is None:
            raise RuntimeError("Page not set. Call set_page() first.")

        timeout = timeout or self._state_change_timeout

        # Capture 'before' screenshot
        before = await self._page.screenshot()

        # Wait for network idle
        try:
            await self._page.wait_for_load_state("networkidle", timeout=timeout * 1000)
        except Exception:
            pass  # Network idle timeout is not fatal

        # Poll for visual change
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

            after = await self._page.screenshot()
            diff_ratio = self._calculate_diff(before, after)

            if diff_ratio > self._pixelmatch_threshold:
                self._last_screenshot = after
                return True

        return False

    @property
    def last_screenshot(self) -> bytes | None:
        """Get the last captured screenshot."""
        return self._last_screenshot

    # Task state management methods

    def save_state(self, task_id: str, state_data: dict[str, Any]) -> None:
        """Save state data for a task.

        Args:
            task_id: Unique task identifier
            state_data: State data to save

        Raises:
            ValueError: If task_id is empty
        """
        if not task_id:
            raise ValueError("task_id cannot be empty")

        # Add timestamp if not present
        if "timestamp" not in state_data:
            state_data["timestamp"] = datetime.now().isoformat()

        # Save current state
        self._states[task_id] = state_data.copy()

        # Append to history
        self._state_history[task_id].append(state_data.copy())

    def get_state(self, task_id: str) -> dict[str, Any] | None:
        """Get current state for a task.

        Args:
            task_id: Unique task identifier

        Returns:
            Current state data or None if not found
        """
        return self._states.get(task_id)

    def get_state_history(self, task_id: str) -> list[dict[str, Any]]:
        """Get state history for a task.

        Args:
            task_id: Unique task identifier

        Returns:
            List of historical state snapshots (oldest first)
        """
        return self._state_history.get(task_id, []).copy()

    def clear_state(self, task_id: str) -> bool:
        """Clear state and history for a task.

        Args:
            task_id: Unique task identifier

        Returns:
            True if state was cleared, False if task not found
        """
        found = False

        if task_id in self._states:
            del self._states[task_id]
            found = True

        if task_id in self._state_history:
            del self._state_history[task_id]
            found = True

        return found

    def update_task_state(
        self,
        task_id: str,
        state: TaskState,
        step: int | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update task execution state.

        Convenience method for updating common task state fields.

        Args:
            task_id: Unique task identifier
            state: New task state
            step: Current step number
            error: Error message if failed
            metadata: Additional metadata
        """
        state_data: dict[str, Any] = {
            "state": state.value,
            "timestamp": datetime.now().isoformat(),
        }

        if step is not None:
            state_data["step"] = step

        if error is not None:
            state_data["error"] = error

        if metadata:
            state_data["metadata"] = metadata

        self.save_state(task_id, state_data)

    def get_all_tasks(self) -> list[str]:
        """Get all task IDs with saved state.

        Returns:
            List of task IDs
        """
        return list(self._states.keys())

    def clear_all(self) -> int:
        """Clear all states and history.

        Returns:
            Number of tasks cleared
        """
        count = len(self._states)
        self._states.clear()
        self._state_history.clear()
        return count

    def _calculate_diff(self, img1_bytes: bytes, img2_bytes: bytes) -> float:
        """Calculate pixel difference ratio between two images.

        Args:
            img1_bytes: First image as PNG bytes
            img2_bytes: Second image as PNG bytes

        Returns:
            Ratio of different pixels (0.0 to 1.0)
        """
        import io

        img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")

        # Different sizes = completely different
        if img1.size != img2.size:
            return 1.0

        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())

        total_pixels = len(pixels1)
        if total_pixels == 0:
            return 0.0

        # Count different pixels with tolerance for anti-aliasing
        diff_count = 0
        tolerance = 10

        for p1, p2 in zip(pixels1, pixels2):
            if (
                abs(p1[0] - p2[0]) > tolerance
                or abs(p1[1] - p2[1]) > tolerance
                or abs(p1[2] - p2[2]) > tolerance
            ):
                diff_count += 1

        return diff_count / total_pixels

    async def initialize(self) -> None:
        """Initialize the module."""
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

    async def cleanup(self) -> None:
        """Cleanup module resources."""
        self._last_screenshot = None

    @property
    def is_ready(self) -> bool:
        """Check if module is ready for use."""
        return self._page is not None
