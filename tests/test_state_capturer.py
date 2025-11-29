"""Unit tests for State Capturer module."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentb.core.config import Config
from agentb.state_capturer.state_capturer import StateCapturer


class TestStateCapturer:
    """Test suite for StateCapturer class."""

    @pytest.mark.asyncio
    async def test_init_creates_screenshots_directory(
        self, mock_page: AsyncMock, test_config: Config
    ) -> None:
        """Test that __init__ creates screenshots directory if it doesn't exist."""
        # Create config with non-existent directory
        non_existent_dir = test_config.screenshots_dir.parent / "new_screenshots"
        config = Config(
            screenshots_dir=non_existent_dir,
            chroma_persist_dir=test_config.chroma_persist_dir,
        )

        # Initialize state capturer
        capturer = StateCapturer(mock_page, config)

        # Verify directory was created
        assert capturer._screenshots_dir.exists()
        assert capturer._screenshots_dir.is_dir()

        # Cleanup
        non_existent_dir.rmdir()

    @pytest.mark.asyncio
    async def test_capture_state_creates_screenshot_with_timestamp(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that capture_state() creates screenshot with correct timestamp format."""
        # Setup mock to return sample screenshot
        mock_page.screenshot.return_value = sample_image_bytes

        # Create state capturer
        capturer = StateCapturer(mock_page, test_config)

        # Capture state with timestamp before and after
        before_time = datetime.now()
        filepath = await capturer.capture_state("test_step")
        after_time = datetime.now()

        # Verify screenshot was taken
        mock_page.screenshot.assert_called_once()

        # Verify file was created
        assert filepath.exists()
        assert filepath.suffix == ".png"

        # Verify filename contains timestamp
        filename = filepath.name
        assert "test_step" in filename

        # Extract timestamp from filename (format: YYYYMMDD_HHMMSS_ffffff_stepname.png)
        timestamp_str = "_".join(filename.split("_")[:3])
        file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")

        # Verify timestamp is within expected range
        assert before_time <= file_time <= after_time

        # Verify screenshot bytes were saved
        assert filepath.read_bytes() == sample_image_bytes

        # Verify last_screenshot was updated
        assert capturer.last_screenshot == sample_image_bytes

    @pytest.mark.asyncio
    async def test_capture_state_sanitizes_step_name(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that capture_state() sanitizes step names with special characters."""
        mock_page.screenshot.return_value = sample_image_bytes

        capturer = StateCapturer(mock_page, test_config)

        # Capture state with special characters in name
        filepath = await capturer.capture_state("test/step:with*special?chars")

        # Verify special characters were replaced with underscores
        assert filepath.exists()
        assert "test_step_with_special_chars" in filepath.name

    @pytest.mark.asyncio
    async def test_get_screenshot_returns_bytes(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that get_screenshot() returns screenshot as bytes."""
        mock_page.screenshot.return_value = sample_image_bytes

        capturer = StateCapturer(mock_page, test_config)

        # Get screenshot
        screenshot = await capturer.get_screenshot()

        # Verify screenshot bytes returned
        assert screenshot == sample_image_bytes

        # Verify last_screenshot was updated
        assert capturer.last_screenshot == sample_image_bytes

    @pytest.mark.asyncio
    async def test_wait_for_change_detects_change_above_threshold(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        different_image_bytes: bytes,
    ) -> None:
        """Test that wait_for_change() detects visual changes above threshold."""
        # Setup mock to return different images on consecutive calls
        mock_page.screenshot.side_effect = [
            sample_image_bytes,  # Initial 'before' screenshot
            sample_image_bytes,  # First poll (no change)
            different_image_bytes,  # Second poll (change detected)
        ]

        capturer = StateCapturer(mock_page, test_config)

        # Wait for change with short timeout
        result = await capturer.wait_for_change(timeout=2.0)

        # Verify change was detected
        assert result is True

        # Verify page.screenshot was called multiple times
        assert mock_page.screenshot.call_count >= 2

        # Verify last_screenshot was updated
        assert capturer.last_screenshot == different_image_bytes

    @pytest.mark.asyncio
    async def test_wait_for_change_returns_false_on_timeout_no_changes(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that wait_for_change() returns False on timeout with no changes."""
        # Setup mock to always return same image
        mock_page.screenshot.return_value = sample_image_bytes

        # Create config with very short timeout and poll interval
        config = Config(
            screenshots_dir=test_config.screenshots_dir,
            chroma_persist_dir=test_config.chroma_persist_dir,
            state_change_timeout=1.0,
            state_change_poll_interval=0.2,
            pixelmatch_threshold=0.02,
        )

        capturer = StateCapturer(mock_page, config)

        # Wait for change (should timeout)
        result = await capturer.wait_for_change()

        # Verify timeout occurred
        assert result is False

        # Verify screenshot was called multiple times during polling
        assert mock_page.screenshot.call_count >= 2

    @pytest.mark.asyncio
    async def test_wait_for_change_handles_network_idle_timeout(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        different_image_bytes: bytes,
    ) -> None:
        """Test that wait_for_change() continues polling even if network idle times out."""
        # Setup mock to timeout on wait_for_load_state
        mock_page.wait_for_load_state.side_effect = asyncio.TimeoutError(
            "Network idle timeout"
        )

        # Setup screenshot to change after timeout
        mock_page.screenshot.side_effect = [
            sample_image_bytes,  # Before
            sample_image_bytes,  # First poll
            different_image_bytes,  # Second poll (change)
        ]

        capturer = StateCapturer(mock_page, test_config)

        # Wait for change (should detect change despite network idle timeout)
        result = await capturer.wait_for_change(timeout=2.0)

        # Verify change was detected
        assert result is True

        # Verify wait_for_load_state was called and raised timeout
        mock_page.wait_for_load_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_change_detects_small_changes_above_threshold(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        slightly_different_image_bytes: bytes,
    ) -> None:
        """Test that wait_for_change() detects small changes above threshold."""
        # Setup mock to return slightly different image
        mock_page.screenshot.side_effect = [
            sample_image_bytes,  # Before
            slightly_different_image_bytes,  # After (1% change)
        ]

        # Reduce threshold to detect small changes
        config = Config(
            screenshots_dir=test_config.screenshots_dir,
            chroma_persist_dir=test_config.chroma_persist_dir,
            pixelmatch_threshold=0.005,  # 0.5% threshold (1% change should trigger)
            state_change_poll_interval=0.1,
        )

        capturer = StateCapturer(mock_page, config)

        # Wait for change
        result = await capturer.wait_for_change(timeout=1.0)

        # Verify change was detected
        assert result is True

    def test_calculate_diff_identical_images(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that _calculate_diff() returns 0.0 for identical images."""
        capturer = StateCapturer(mock_page, test_config)

        # Calculate diff between identical images
        diff_ratio = capturer._calculate_diff(sample_image_bytes, sample_image_bytes)

        # Verify no difference
        assert diff_ratio == 0.0

    def test_calculate_diff_completely_different_images(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        different_image_bytes: bytes,
    ) -> None:
        """Test that _calculate_diff() returns high value for completely different images."""
        capturer = StateCapturer(mock_page, test_config)

        # Calculate diff between white and black images
        diff_ratio = capturer._calculate_diff(
            sample_image_bytes, different_image_bytes
        )

        # Verify high difference (should be close to 1.0)
        assert diff_ratio > 0.9  # Most pixels are different

    def test_calculate_diff_slightly_different_images(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        slightly_different_image_bytes: bytes,
    ) -> None:
        """Test that _calculate_diff() correctly measures small differences."""
        capturer = StateCapturer(mock_page, test_config)

        # Calculate diff (1% of pixels changed)
        diff_ratio = capturer._calculate_diff(
            sample_image_bytes, slightly_different_image_bytes
        )

        # Verify small difference (should be around 0.01)
        assert 0.005 < diff_ratio < 0.02

    def test_calculate_diff_different_size_images(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        different_size_image_bytes: bytes,
    ) -> None:
        """Test that _calculate_diff() returns 1.0 for different sized images."""
        capturer = StateCapturer(mock_page, test_config)

        # Calculate diff between different sized images
        diff_ratio = capturer._calculate_diff(
            sample_image_bytes, different_size_image_bytes
        )

        # Verify images are considered completely different
        assert diff_ratio == 1.0

    def test_calculate_diff_with_anti_aliasing_tolerance(
        self,
        mock_page: AsyncMock,
        test_config: Config,
    ) -> None:
        """Test that _calculate_diff() tolerates small color differences (anti-aliasing)."""
        import io

        from PIL import Image

        # Create two images with very small color differences (within tolerance)
        img1 = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img2 = Image.new("RGB", (100, 100), color=(250, 250, 250))  # 5 point difference

        buffer1 = io.BytesIO()
        img1.save(buffer1, format="PNG")
        bytes1 = buffer1.getvalue()

        buffer2 = io.BytesIO()
        img2.save(buffer2, format="PNG")
        bytes2 = buffer2.getvalue()

        capturer = StateCapturer(mock_page, test_config)

        # Calculate diff (should be 0 due to tolerance)
        diff_ratio = capturer._calculate_diff(bytes1, bytes2)

        # Verify no difference detected (within tolerance of 10)
        assert diff_ratio == 0.0

    def test_last_screenshot_property(
        self,
        mock_page: AsyncMock,
        test_config: Config,
    ) -> None:
        """Test that last_screenshot property returns last captured screenshot."""
        capturer = StateCapturer(mock_page, test_config)

        # Initially None
        assert capturer.last_screenshot is None

        # Set last screenshot
        test_bytes = b"test screenshot data"
        capturer._last_screenshot = test_bytes

        # Verify property returns correct value
        assert capturer.last_screenshot == test_bytes

    @pytest.mark.asyncio
    async def test_multiple_captures_update_last_screenshot(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
        different_image_bytes: bytes,
    ) -> None:
        """Test that multiple captures correctly update last_screenshot."""
        mock_page.screenshot.side_effect = [sample_image_bytes, different_image_bytes]

        capturer = StateCapturer(mock_page, test_config)

        # First capture
        await capturer.capture_state("step1")
        assert capturer.last_screenshot == sample_image_bytes

        # Second capture
        await capturer.capture_state("step2")
        assert capturer.last_screenshot == different_image_bytes

    @pytest.mark.asyncio
    async def test_wait_for_change_with_custom_timeout(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that wait_for_change() respects custom timeout parameter."""
        mock_page.screenshot.return_value = sample_image_bytes

        # Set short poll interval for faster testing
        config = Config(
            screenshots_dir=test_config.screenshots_dir,
            chroma_persist_dir=test_config.chroma_persist_dir,
            state_change_poll_interval=0.1,
        )

        capturer = StateCapturer(mock_page, config)

        # Measure time taken with custom timeout
        import time

        start_time = time.time()
        result = await capturer.wait_for_change(timeout=0.5)
        elapsed_time = time.time() - start_time

        # Verify timeout was respected
        assert result is False
        assert 0.4 < elapsed_time < 0.8  # Allow some margin

    @pytest.mark.asyncio
    async def test_capture_state_with_empty_step_name(
        self,
        mock_page: AsyncMock,
        test_config: Config,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that capture_state() handles empty step names gracefully."""
        mock_page.screenshot.return_value = sample_image_bytes

        capturer = StateCapturer(mock_page, test_config)

        # Capture with empty name
        filepath = await capturer.capture_state("")

        # Verify file was created with just timestamp
        assert filepath.exists()
        assert filepath.suffix == ".png"

        # Filename should contain timestamp and .png
        parts = filepath.stem.split("_")
        assert len(parts) >= 3  # YYYYMMDD, HHMMSS, ffffff

    @pytest.mark.asyncio
    async def test_state_capturer_with_default_config(
        self,
        mock_page: AsyncMock,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that StateCapturer works with default config (None)."""
        mock_page.screenshot.return_value = sample_image_bytes

        # Create capturer with no config (should use default)
        capturer = StateCapturer(mock_page)

        # Verify default config was created
        assert capturer.config is not None
        assert isinstance(capturer.config, Config)

        # Verify screenshots directory was created
        assert capturer._screenshots_dir.exists()

        # Test basic functionality
        screenshot = await capturer.get_screenshot()
        assert screenshot == sample_image_bytes

        # Cleanup default directory if it was created
        default_dir = Path("./data/screenshots")
        if default_dir.exists() and default_dir != capturer._screenshots_dir:
            import shutil

            shutil.rmtree(default_dir.parent)
