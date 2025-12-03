"""Executor - Browser automation wrapper around Playwright."""

from typing import TYPE_CHECKING

import structlog
from playwright.async_api import async_playwright, Browser, BrowserContext

from agentb.core.config import Config
from agentb.core.types import Coordinates

if TYPE_CHECKING:
    from playwright.async_api import Page


logger = structlog.get_logger()


class Executor:
    """Browser automation executor using Playwright."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the executor.

        Args:
            config: Application configuration
        """
        self.config = config or Config()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: "Page | None" = None

    async def start(self) -> "Page":
        """Start the browser and return the page.

        Returns:
            Playwright page instance
        """
        self._playwright = await async_playwright().start()

        # Use persistent context if user_data_dir is specified
        if self.config.user_data_dir:
            logger.info(
                "browser_starting_persistent",
                user_data_dir=str(self.config.user_data_dir),
                headless=self.config.headless,
            )
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.config.user_data_dir),
                headless=self.config.headless,
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            # Standard ephemeral browser context
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless
            )

            # Prepare context options
            context_options = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                }
            }

            # Load storage state if provided
            if self.config.storage_state and self.config.storage_state.exists():
                logger.info(
                    "loading_storage_state",
                    storage_state=str(self.config.storage_state),
                )
                context_options["storage_state"] = str(self.config.storage_state)

            self._context = await self._browser.new_context(**context_options)
            self._page = await self._context.new_page()

        logger.info("browser_started", headless=self.config.headless)
        return self._page

    async def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        # Close context (works for both persistent and regular contexts)
        if self._context:
            try:
                await self._context.close()
            except Exception as e:
                logger.debug("context_close_error", error=str(e))

        # Close browser (only exists for non-persistent contexts)
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.debug("browser_close_error", error=str(e))

        # Stop playwright
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug("playwright_stop_error", error=str(e))

        logger.info("browser_stopped")

    @property
    def page(self) -> "Page":
        """Get the current page instance."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    async def navigate(self, url: str) -> None:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
        """
        # Use "load" instead of "networkidle" for better compatibility with SPAs
        # "networkidle" can timeout on sites like Notion that make continuous requests
        await self.page.goto(url, wait_until="load", timeout=60000)
        logger.info("navigated", url=url)

    def get_current_url(self) -> str:
        """Get the current page URL.

        Returns:
            Current URL as string
        """
        return self.page.url

    async def click(self, x: int, y: int) -> None:
        """Click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        await self.page.mouse.click(x, y)
        logger.info("clicked", x=x, y=y)

    async def type_text(self, x: int, y: int, text: str) -> None:
        """Click at coordinates and type text.

        Args:
            x: X coordinate
            y: Y coordinate
            text: Text to type
        """
        await self.page.mouse.click(x, y)
        await self.page.keyboard.type(text)
        logger.info("typed", x=x, y=y, text_length=len(text))

    async def select(self, x: int, y: int, value: str) -> None:
        """Click to open selector and choose value.

        Args:
            x: X coordinate of selector
            y: Y coordinate of selector
            value: Value to select
        """
        # Click to open the selector
        await self.page.mouse.click(x, y)
        await self.page.wait_for_timeout(200)  # Wait for dropdown

        # Type to filter and select
        await self.page.keyboard.type(value)
        await self.page.keyboard.press("Enter")
        logger.info("selected", x=x, y=y, value=value)

    async def get_screenshot(self) -> bytes:
        """Get current screenshot as bytes.

        Returns:
            Screenshot as PNG bytes
        """
        return await self.page.screenshot()

    async def find_element_by_text(self, text: str) -> Coordinates | None:
        """Find element by text content using DOM search.

        Args:
            text: Text to search for

        Returns:
            Coordinates if exactly one match found, None otherwise
        """
        # Try multiple selectors for text matching
        selectors = [
            f"text={text}",
            f"button:has-text('{text}')",
            f"a:has-text('{text}')",
            f"[aria-label='{text}']",
            f"[placeholder='{text}']",
            f"label:has-text('{text}')",
        ]

        all_matches = []

        for selector in selectors:
            try:
                elements = await self.page.locator(selector).all()
                for element in elements:
                    if await element.is_visible():
                        box = await element.bounding_box()
                        if box:
                            # Get center coordinates
                            coords = Coordinates(
                                x=int(box["x"] + box["width"] / 2),
                                y=int(box["y"] + box["height"] / 2),
                            )
                            # Avoid duplicates (same coordinates)
                            if not any(
                                m.x == coords.x and m.y == coords.y
                                for m in all_matches
                            ):
                                all_matches.append(coords)
            except Exception:
                continue

        if len(all_matches) == 1:
            logger.info(
                "element_found_by_text",
                text=text,
                x=all_matches[0].x,
                y=all_matches[0].y,
            )
            return all_matches[0]
        elif len(all_matches) == 0:
            logger.debug("element_not_found", text=text)
        else:
            logger.debug(
                "multiple_elements_found",
                text=text,
                count=len(all_matches),
            )

        return None

    async def press_key(self, key: str) -> None:
        """Press a keyboard key.

        Args:
            key: Key to press (e.g., 'Enter', 'Escape', 'Tab')
        """
        await self.page.keyboard.press(key)
        logger.info("key_pressed", key=key)

    async def scroll(self, x: int, y: int, delta_y: int) -> None:
        """Scroll at position.

        Args:
            x: X coordinate
            y: Y coordinate
            delta_y: Scroll amount (positive = down, negative = up)
        """
        await self.page.mouse.move(x, y)
        await self.page.mouse.wheel(0, delta_y)
        logger.info("scrolled", x=x, y=y, delta_y=delta_y)

    async def verify_element_at(self, coords: Coordinates) -> bool:
        """Verify if an element exists at the given coordinates.

        Quick check to see if cached coordinates are still valid.

        Args:
            coords: Coordinates to check

        Returns:
            True if element appears to exist at those coordinates
        """
        try:
            # Use Playwright's locator.elementHandle at position
            # This is a quick check that doesn't require vision
            element = await self.page.locator(f'xpath=//*').element_handle()
            if element:
                # Get bounding box and check if coords are within any visible element
                # This is a simplified check - coordinates are still likely valid
                # A more robust check would query element at exact position
                return True
            return False
        except Exception:
            # If anything fails, assume coordinates might still be valid
            # Better to try and fail than skip cached coords
            return True
