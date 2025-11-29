"""Unit tests for Executor module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentb.core.config import Config
from agentb.core.types import Coordinates
from agentb.executor.executor import Executor


class TestExecutor:
    """Test suite for Executor class."""

    @pytest.mark.asyncio
    async def test_init_with_default_config(self) -> None:
        """Test that Executor initializes with default config."""
        executor = Executor()

        assert executor.config is not None
        assert isinstance(executor.config, Config)
        assert executor._playwright is None
        assert executor._browser is None
        assert executor._context is None
        assert executor._page is None

    @pytest.mark.asyncio
    async def test_init_with_custom_config(self, test_config: Config) -> None:
        """Test that Executor initializes with custom config."""
        executor = Executor(test_config)

        assert executor.config == test_config
        assert executor.config.headless is True
        assert executor.config.viewport_width == 1280
        assert executor.config.viewport_height == 720

    @pytest.mark.asyncio
    async def test_start_launches_browser(self, test_config: Config) -> None:
        """Test that start() launches browser and returns page."""
        # Mock Playwright components
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        executor = Executor(test_config)

        # Patch async_playwright
        with patch(
            "agentb.executor.executor.async_playwright"
        ) as mock_async_playwright:
            mock_async_playwright.return_value.start = AsyncMock(
                return_value=mock_playwright
            )

            # Start browser
            page = await executor.start()

            # Verify browser was launched
            mock_playwright.chromium.launch.assert_called_once_with(headless=True)

            # Verify context was created with viewport
            mock_browser.new_context.assert_called_once_with(
                viewport={"width": 1280, "height": 720}
            )

            # Verify page was created
            mock_context.new_page.assert_called_once()

            # Verify page returned
            assert page == mock_page
            assert executor._page == mock_page
            assert executor._browser == mock_browser
            assert executor._context == mock_context

    @pytest.mark.asyncio
    async def test_stop_closes_browser(self) -> None:
        """Test that stop() closes browser and cleans up resources."""
        executor = Executor()

        # Mock browser and playwright
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        executor._browser = mock_browser
        executor._playwright = mock_playwright

        # Stop browser
        await executor.stop()

        # Verify cleanup
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_none_browser(self) -> None:
        """Test that stop() handles None browser gracefully."""
        executor = Executor()

        # Stop without starting (no browser)
        await executor.stop()

        # Should not raise exception

    @pytest.mark.asyncio
    async def test_page_property_returns_page(self) -> None:
        """Test that page property returns the page instance."""
        executor = Executor()
        mock_page = AsyncMock()
        executor._page = mock_page

        assert executor.page == mock_page

    @pytest.mark.asyncio
    async def test_page_property_raises_if_not_started(self) -> None:
        """Test that page property raises RuntimeError if browser not started."""
        executor = Executor()

        with pytest.raises(RuntimeError, match="Browser not started"):
            _ = executor.page

    @pytest.mark.asyncio
    async def test_navigate_goes_to_url(self) -> None:
        """Test that navigate() goes to URL with networkidle wait."""
        executor = Executor()
        mock_page = AsyncMock()
        executor._page = mock_page

        # Navigate to URL
        await executor.navigate("https://example.com")

        # Verify goto was called with updated parameters
        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="load", timeout=60000
        )

    @pytest.mark.asyncio
    async def test_click_at_coordinates(self) -> None:
        """Test that click() clicks at specified coordinates."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        executor._page = mock_page

        # Click at coordinates
        await executor.click(100, 200)

        # Verify mouse click
        mock_page.mouse.click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_type_text_clicks_and_types(self) -> None:
        """Test that type_text() clicks and types text."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        mock_page.keyboard = AsyncMock()
        executor._page = mock_page

        # Type text at coordinates
        await executor.type_text(50, 75, "Hello World")

        # Verify mouse click
        mock_page.mouse.click.assert_called_once_with(50, 75)

        # Verify keyboard type
        mock_page.keyboard.type.assert_called_once_with("Hello World")

    @pytest.mark.asyncio
    async def test_select_clicks_types_and_enters(self) -> None:
        """Test that select() clicks, types value, and presses Enter."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        executor._page = mock_page

        # Select value
        await executor.select(100, 150, "Option A")

        # Verify mouse click
        mock_page.mouse.click.assert_called_once_with(100, 150)

        # Verify wait for dropdown
        mock_page.wait_for_timeout.assert_called_once_with(200)

        # Verify keyboard type
        mock_page.keyboard.type.assert_called_once_with("Option A")

        # Verify Enter press
        mock_page.keyboard.press.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_get_screenshot_returns_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """Test that get_screenshot() returns screenshot bytes."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=sample_image_bytes)
        executor._page = mock_page

        # Get screenshot
        screenshot = await executor.get_screenshot()

        # Verify screenshot returned
        assert screenshot == sample_image_bytes
        mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_press_key_presses_key(self) -> None:
        """Test that press_key() presses the specified key."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        executor._page = mock_page

        # Press Enter key
        await executor.press_key("Enter")

        # Verify key press
        mock_page.keyboard.press.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_scroll_moves_and_scrolls(self) -> None:
        """Test that scroll() moves mouse and scrolls."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        executor._page = mock_page

        # Scroll down
        await executor.scroll(200, 300, 100)

        # Verify mouse move
        mock_page.mouse.move.assert_called_once_with(200, 300)

        # Verify scroll
        mock_page.mouse.wheel.assert_called_once_with(0, 100)

    @pytest.mark.asyncio
    async def test_find_element_by_text_single_match(self) -> None:
        """Test that find_element_by_text() returns coordinates for single match."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock element with bounding box
        mock_element = AsyncMock()
        mock_element.is_visible = AsyncMock(return_value=True)
        mock_element.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 200, "width": 50, "height": 30}
        )

        # Mock locator that returns one element
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_element])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Click Me")

        # Verify coordinates (center of bounding box)
        assert coords is not None
        assert coords.x == 125  # 100 + 50/2
        assert coords.y == 215  # 200 + 30/2

    @pytest.mark.asyncio
    async def test_find_element_by_text_no_match(self) -> None:
        """Test that find_element_by_text() returns None when no matches."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock locator with no elements
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Nonexistent")

        # Verify None returned
        assert coords is None

    @pytest.mark.asyncio
    async def test_find_element_by_text_multiple_matches(self) -> None:
        """Test that find_element_by_text() returns None for multiple matches."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock two different elements
        mock_element1 = AsyncMock()
        mock_element1.is_visible = AsyncMock(return_value=True)
        mock_element1.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 200, "width": 50, "height": 30}
        )

        mock_element2 = AsyncMock()
        mock_element2.is_visible = AsyncMock(return_value=True)
        mock_element2.bounding_box = AsyncMock(
            return_value={"x": 300, "y": 400, "width": 50, "height": 30}
        )

        # Mock locator that returns two elements
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_element1, mock_element2])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Submit")

        # Verify None returned (ambiguous)
        assert coords is None

    @pytest.mark.asyncio
    async def test_find_element_by_text_skips_invisible_elements(self) -> None:
        """Test that find_element_by_text() skips invisible elements."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock visible element
        mock_visible = AsyncMock()
        mock_visible.is_visible = AsyncMock(return_value=True)
        mock_visible.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 200, "width": 50, "height": 30}
        )

        # Mock invisible element
        mock_invisible = AsyncMock()
        mock_invisible.is_visible = AsyncMock(return_value=False)

        # Mock locator that returns both
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_invisible, mock_visible])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Visible Button")

        # Verify only visible element found
        assert coords is not None
        assert coords.x == 125
        assert coords.y == 215

    @pytest.mark.asyncio
    async def test_find_element_by_text_skips_elements_without_bounding_box(
        self,
    ) -> None:
        """Test that find_element_by_text() skips elements without bounding box."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock element without bounding box (returns None)
        mock_no_box = AsyncMock()
        mock_no_box.is_visible = AsyncMock(return_value=True)
        mock_no_box.bounding_box = AsyncMock(return_value=None)

        # Mock element with bounding box
        mock_with_box = AsyncMock()
        mock_with_box.is_visible = AsyncMock(return_value=True)
        mock_with_box.bounding_box = AsyncMock(
            return_value={"x": 50, "y": 100, "width": 40, "height": 20}
        )

        # Mock locator
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_no_box, mock_with_box])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Link")

        # Verify only element with box found
        assert coords is not None
        assert coords.x == 70  # 50 + 40/2
        assert coords.y == 110  # 100 + 20/2

    @pytest.mark.asyncio
    async def test_find_element_by_text_deduplicates_same_coordinates(self) -> None:
        """Test that find_element_by_text() deduplicates elements at same coordinates."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock two elements at same position (e.g., nested elements)
        mock_element1 = AsyncMock()
        mock_element1.is_visible = AsyncMock(return_value=True)
        mock_element1.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 200, "width": 50, "height": 30}
        )

        mock_element2 = AsyncMock()
        mock_element2.is_visible = AsyncMock(return_value=True)
        mock_element2.bounding_box = AsyncMock(
            return_value={"x": 100, "y": 200, "width": 50, "height": 30}
        )

        # Mock locator that returns both
        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_element1, mock_element2])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Button")

        # Verify single match after deduplication
        assert coords is not None
        assert coords.x == 125
        assert coords.y == 215

    @pytest.mark.asyncio
    async def test_find_element_by_text_tries_multiple_selectors(self) -> None:
        """Test that find_element_by_text() tries multiple selector strategies."""
        executor = Executor()
        mock_page = AsyncMock()

        call_count = 0
        expected_selectors = [
            "text=Submit",
            "button:has-text('Submit')",
            "a:has-text('Submit')",
            "[aria-label='Submit']",
            "[placeholder='Submit']",
            "label:has-text('Submit')",
        ]

        def mock_locator_func(selector: str):
            nonlocal call_count
            # Verify selector is expected
            assert selector in expected_selectors

            # Return element only on button selector
            if selector == "button:has-text('Submit')":
                mock_element = AsyncMock()
                mock_element.is_visible = AsyncMock(return_value=True)
                mock_element.bounding_box = AsyncMock(
                    return_value={"x": 10, "y": 20, "width": 30, "height": 40}
                )
                mock_locator = AsyncMock()
                mock_locator.all = AsyncMock(return_value=[mock_element])
            else:
                mock_locator = AsyncMock()
                mock_locator.all = AsyncMock(return_value=[])

            call_count += 1
            return mock_locator

        mock_page.locator = MagicMock(side_effect=mock_locator_func)
        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Submit")

        # Verify element found
        assert coords is not None
        assert coords.x == 25  # 10 + 30/2
        assert coords.y == 40  # 20 + 40/2

        # Verify multiple selectors tried
        assert call_count >= 2  # At least text= and button:has-text

    @pytest.mark.asyncio
    async def test_find_element_by_text_handles_selector_exceptions(self) -> None:
        """Test that find_element_by_text() handles selector exceptions gracefully."""
        executor = Executor()
        mock_page = AsyncMock()

        # First selector raises exception, second succeeds
        call_count = 0

        def mock_locator_func(selector: str):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call raises exception
                mock_locator = AsyncMock()
                mock_locator.all = AsyncMock(side_effect=Exception("Selector error"))
                return mock_locator
            else:
                # Second call succeeds
                mock_element = AsyncMock()
                mock_element.is_visible = AsyncMock(return_value=True)
                mock_element.bounding_box = AsyncMock(
                    return_value={"x": 0, "y": 0, "width": 100, "height": 50}
                )
                mock_locator = AsyncMock()
                mock_locator.all = AsyncMock(return_value=[mock_element])
                return mock_locator

        mock_page.locator = MagicMock(side_effect=mock_locator_func)
        executor._page = mock_page

        # Find element (should not raise exception)
        coords = await executor.find_element_by_text("Search")

        # Verify element found despite first selector failing
        assert coords is not None
        assert coords.x == 50
        assert coords.y == 25

    @pytest.mark.asyncio
    async def test_press_key_with_special_keys(self) -> None:
        """Test that press_key() handles special keys (Escape, Tab, etc.)."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.keyboard = AsyncMock()
        executor._page = mock_page

        # Test various special keys
        special_keys = ["Escape", "Tab", "ArrowDown", "PageDown", "Control"]

        for key in special_keys:
            await executor.press_key(key)

        # Verify all keys pressed
        assert mock_page.keyboard.press.call_count == len(special_keys)

    @pytest.mark.asyncio
    async def test_scroll_negative_delta(self) -> None:
        """Test that scroll() handles negative delta (scroll up)."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        executor._page = mock_page

        # Scroll up
        await executor.scroll(100, 200, -150)

        # Verify scroll with negative delta
        mock_page.mouse.wheel.assert_called_once_with(0, -150)

    @pytest.mark.asyncio
    async def test_type_text_with_empty_string(self) -> None:
        """Test that type_text() handles empty string gracefully."""
        executor = Executor()
        mock_page = AsyncMock()
        mock_page.mouse = AsyncMock()
        mock_page.keyboard = AsyncMock()
        executor._page = mock_page

        # Type empty string
        await executor.type_text(10, 20, "")

        # Verify click happened
        mock_page.mouse.click.assert_called_once_with(10, 20)

        # Verify type was called with empty string
        mock_page.keyboard.type.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_coordinates_returned_are_integers(self) -> None:
        """Test that find_element_by_text() returns integer coordinates."""
        executor = Executor()
        mock_page = AsyncMock()

        # Mock element with float bounding box values
        mock_element = AsyncMock()
        mock_element.is_visible = AsyncMock(return_value=True)
        mock_element.bounding_box = AsyncMock(
            return_value={
                "x": 100.7,
                "y": 200.3,
                "width": 50.6,
                "height": 30.2,
            }
        )

        mock_locator = AsyncMock()
        mock_locator.all = AsyncMock(return_value=[mock_element])
        mock_page.locator = MagicMock(return_value=mock_locator)

        executor._page = mock_page

        # Find element
        coords = await executor.find_element_by_text("Test")

        # Verify coordinates are integers
        assert coords is not None
        assert isinstance(coords.x, int)
        assert isinstance(coords.y, int)
        assert coords.x == 126  # int(100.7 + 50.6/2)
        assert coords.y == 215  # int(200.3 + 30.2/2)
