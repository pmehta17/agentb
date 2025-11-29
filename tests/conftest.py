"""Shared pytest fixtures for Agent B tests."""

import asyncio
import io
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from agentb.core.config import Config


@pytest.fixture
def temp_screenshots_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for screenshots.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to temporary screenshots directory
    """
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    return screenshots_dir


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for ChromaDB.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to temporary ChromaDB directory
    """
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


@pytest.fixture
def test_config(temp_screenshots_dir: Path, temp_chroma_dir: Path) -> Config:
    """Create a test configuration with temporary directories.

    Args:
        temp_screenshots_dir: Temporary screenshots directory
        temp_chroma_dir: Temporary ChromaDB directory

    Returns:
        Config instance for testing
    """
    return Config(
        anthropic_api_key="test-api-key",
        screenshots_dir=temp_screenshots_dir,
        chroma_persist_dir=temp_chroma_dir,
        pixelmatch_threshold=0.02,
        state_change_timeout=10.0,
        state_change_poll_interval=0.5,
        headless=True,
        viewport_width=1280,
        viewport_height=720,
    )


@pytest.fixture
def mock_page() -> AsyncMock:
    """Create a mock Playwright Page object.

    Returns:
        AsyncMock configured to simulate Playwright Page
    """
    page = AsyncMock()

    # Mock screenshot method
    page.screenshot = AsyncMock()

    # Mock wait_for_load_state method
    page.wait_for_load_state = AsyncMock()

    return page


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a sample PNG image as bytes.

    Returns:
        PNG image bytes (100x100 white image)
    """
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def different_image_bytes() -> bytes:
    """Create a different PNG image as bytes.

    Returns:
        PNG image bytes (100x100 black image)
    """
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def slightly_different_image_bytes() -> bytes:
    """Create a slightly different PNG image as bytes.

    Returns:
        PNG image bytes (100x100 mostly white, with small black region)
    """
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # Change a small region (1% of pixels) to black
    pixels = img.load()
    for x in range(10):
        for y in range(10):
            pixels[x, y] = (0, 0, 0)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def different_size_image_bytes() -> bytes:
    """Create an image with different dimensions.

    Returns:
        PNG image bytes (200x200 white image)
    """
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_executor() -> MagicMock:
    """Create a mock Executor object.

    Returns:
        MagicMock configured to simulate Executor
    """
    executor = MagicMock()

    # Mock async methods
    executor.click = AsyncMock()
    executor.type_text = AsyncMock()
    executor.select = AsyncMock()
    executor.navigate = AsyncMock()
    executor.find_element_by_text = AsyncMock()
    executor.take_screenshot = AsyncMock()

    return executor


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for Windows compatibility."""
    if asyncio.get_event_loop_policy().__class__.__name__ == "WindowsProactorEventLoopPolicy":
        # Use WindowsSelectorEventLoopPolicy for better async support
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
async def async_test_context() -> AsyncGenerator[None, None]:
    """Provide async test context with proper cleanup.

    Yields:
        None (context manager for async tests)
    """
    yield
    # Cleanup any pending tasks
    pending = [task for task in asyncio.all_tasks() if not task.done()]
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
