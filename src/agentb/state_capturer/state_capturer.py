"""State Capturer - Handles capturing non-URL UI states."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from PIL import Image

from agentb.core.config import Config

if TYPE_CHECKING:
    from playwright.async_api import Page


logger = structlog.get_logger()


class StateCapturer:
    """Captures UI state transitions visually."""

    def __init__(self, page: "Page", config: Config | None = None) -> None:
        """Initialize the state capturer.

        Args:
            page: Playwright page instance
            config: Application configuration
        """
        self.page = page
        self.config = config or Config()
        self._screenshots_dir = Path(self.config.screenshots_dir)
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._last_screenshot: bytes | None = None

    async def capture_state(self, step_name: str) -> Path | None:
        """Save current screenshot with descriptive name.

        Args:
            step_name: Descriptive name for this state

        Returns:
            Path to the saved screenshot, or None if page is blank
        """
        # Skip screenshot if page is blank
        if self.page.url == "about:blank":
            logger.warning(
                "screenshot_skipped_blank_page",
                step_name=step_name,
                url=self.page.url
            )
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in step_name)
        filename = f"{timestamp}_{safe_name}.png"
        filepath = self._screenshots_dir / filename

        screenshot = await self.page.screenshot()
        filepath.write_bytes(screenshot)
        self._last_screenshot = screenshot

        logger.info("state_captured", step_name=step_name, filepath=str(filepath))
        return filepath

    async def get_screenshot(self) -> bytes:
        """Get current screenshot as bytes.

        Returns:
            Screenshot as PNG bytes
        """
        # Warn if page is blank (but still return the screenshot)
        if self.page.url == "about:blank":
            logger.warning(
                "screenshot_from_blank_page",
                url=self.page.url,
                message="Taking screenshot of blank page - may cause vision model issues"
            )

        screenshot = await self.page.screenshot()
        self._last_screenshot = screenshot
        return screenshot

    async def wait_for_change(self, timeout: float | None = None) -> bool:
        """Wait for UI state to change.

        Captures a 'before' screenshot, waits for network idle,
        then polls screenshots and compares using pixel difference.

        Features early exit optimization: exits immediately when significant
        change detected instead of waiting full timeout.

        Args:
            timeout: Max seconds to wait (uses config default if None)

        Returns:
            True if state changed, False if timeout reached
        """
        timeout = timeout or self.config.state_change_timeout
        threshold = self.config.pixelmatch_threshold
        poll_interval = self.config.state_change_poll_interval

        # Capture 'before' screenshot
        before = await self.page.screenshot()

        # Wait for network idle first (with shorter timeout to avoid hanging)
        try:
            await self.page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 3000))
        except Exception:
            # Network idle timeout is not fatal, continue polling
            pass

        # Poll for visual change with adaptive polling interval
        elapsed = 0.0
        current_poll_interval = poll_interval

        while elapsed < timeout:
            await asyncio.sleep(current_poll_interval)
            elapsed += current_poll_interval

            after = await self.page.screenshot()
            diff_ratio = self._calculate_diff(before, after)

            if diff_ratio > threshold:
                # SUCCESS - State changed! Exit immediately
                logger.info(
                    "state_change_detected",
                    diff_ratio=diff_ratio,
                    elapsed_seconds=elapsed,
                )
                self._last_screenshot = after
                return True

            # Adaptive polling: if very similar, slow down polling to reduce CPU
            # This helps when waiting for slow changes
            if diff_ratio < threshold / 10:
                current_poll_interval = min(current_poll_interval * 1.2, 1.0)
            else:
                # Some change happening, poll faster
                current_poll_interval = poll_interval

        logger.warning(
            "state_change_timeout",
            timeout_seconds=timeout,
            threshold=threshold,
        )
        return False

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

        # Ensure same size
        if img1.size != img2.size:
            return 1.0  # Different sizes = completely different

        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())

        total_pixels = len(pixels1)
        if total_pixels == 0:
            return 0.0

        # Count different pixels (with small tolerance for anti-aliasing)
        diff_count = 0
        tolerance = 10  # Allow small color differences

        for p1, p2 in zip(pixels1, pixels2):
            if (
                abs(p1[0] - p2[0]) > tolerance
                or abs(p1[1] - p2[1]) > tolerance
                or abs(p1[2] - p2[2]) > tolerance
            ):
                diff_count += 1

        return diff_count / total_pixels

    @property
    def last_screenshot(self) -> bytes | None:
        """Get the last captured screenshot."""
        return self._last_screenshot
