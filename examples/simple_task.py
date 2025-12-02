"""Example: Execute a simple task with Agent B."""

import asyncio

from agentb.core.config import Config
from agentb.orchestrator import Orchestrator


async def main() -> None:
    """Run a simple example task."""
    # Create configuration
    config = Config(
        headless=False,  # Show browser for demo
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(config=config)

    try:
        # Start the browser
        await orchestrator.start()

        # Navigate to a starting URL
        await orchestrator.navigate_to("https://example.com")

        # Execute a task
        task = "Click the 'More information...' link"
        success = await orchestrator.execute_task(task)

        if success:
            print(f"Task completed successfully: {task}")
        else:
            print(f"Task failed: {task}")

    finally:
        # Always cleanup
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
