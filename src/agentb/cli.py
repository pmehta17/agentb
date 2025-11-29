"""
Agent B Command-Line Interface

Allows running Agent B tasks from the command line without creating Python files.

Usage:
    agentb "Your task description here"
    agentb "Navigate to notion.so and create a database" --headless
    agentb "Search Google for Python tutorials" --screenshots ./my_screenshots
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse
import argparse
import os
import logging

import structlog

from agentb.core.config import Config
from agentb.orchestrator import Orchestrator


def add_cli_status_messages(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add user-friendly CLI status messages for key events."""
    event = event_dict.get("event", "")
    level = event_dict.get("level", "info")

    # Only show status for important events at INFO level or higher
    if level not in ("info", "warning", "error"):
        return event_dict

    # Map events to user-friendly status messages
    status_messages = {
        "skill_cache_hit": "✨ Found cached workflow - using previous successful approach",
        "skill_cache_miss": "🧠 Generating new plan for this task",
        "checking_authentication_status": "🔍 Checking if already logged in...",
        "authenticated_state_detected": "🔐 Authenticated session detected - skipping login steps",
        "executing_step": lambda d: f"🔄 Step {d.get('step', '?')}: {d.get('action', '?')} - {d.get('target', '?')}",
        "step_completed": lambda d: f"✅ Step {d.get('step', '?')} completed",
        "navigation_skipped": lambda d: f"⏭️  Skipped navigation - already on {urlparse(d.get('current_url', '')).netloc if 'current_url' in d else 'target site'}",
        "login_step_skipped": lambda d: f"⏭️  Skipped login step - using saved session",
        "dom_search_failed_trying_vision": "👁️  Using vision to locate element",
        "vision_search_failed": "⚠️  Could not locate element",
        "action_failed": lambda d: f"❌ Action failed: {d.get('error', 'Unknown error')}",
        "max_replans_reached": "❌ Maximum retry attempts reached",
        "skill_saved": "💾 Workflow saved for future reuse",
        "task_validation_failed": "⚠️  Task validation failed",
    }

    if event in status_messages:
        msg = status_messages[event]
        if callable(msg):
            status = msg(event_dict)
        else:
            status = msg
        print(status, flush=True)

    return event_dict


# Configure structured logging
# Use PrintLoggerFactory for simple CLI output
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        add_cli_status_messages,  # Add our custom CLI status processor
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="agentb",
        description="Agent B - Reflexive AI agent for browser automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentb "Go to google.com and search for Python tutorials"
  agentb "Create a database in Notion" --headless
  agentb "Navigate to github.com and create a new repository" --screenshots ./github_screenshots
  agentb "Click the login button" --url https://example.com

Login Handling (Recommended Approach):
  # Step 1: Save your login session (one-time setup)
  python login.py --url https://notion.so --output notion_state.json

  # Step 2: Use the saved session for automated tasks
  agentb "Create a database in Notion" --storage-state notion_state.json
  agentb "Add a new row to the database" --storage-state notion_state.json

  # Works for any site:
  python login.py --url https://github.com --output github_state.json
  agentb "Create a new repository" --storage-state github_state.json

Alternative - Browser Profile (saves everything between runs):
  agentb "Create a database in Notion" --user-data-dir ./browser_profile

Environment Variables:
  ANTHROPIC_API_KEY    Required: Your Anthropic API key

For more information, visit: https://github.com/pmehta17/agentb
        """
    )

    # Positional argument: the task
    parser.add_argument(
        "task",
        type=str,
        help="Natural language description of the task to execute"
    )

    # Optional arguments
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Starting URL to navigate to before executing task"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)"
    )

    parser.add_argument(
        "--screenshots",
        type=str,
        default="./screenshots",
        help="Directory to save screenshots (default: ./screenshots)"
    )

    parser.add_argument(
        "--chroma-db",
        type=str,
        default="./chroma_db",
        help="Directory for ChromaDB persistence (default: ./chroma_db)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Step timeout in seconds (default: 30.0)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum step retry attempts (default: 3)"
    )

    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for cache hits (default: 0.95)"
    )

    parser.add_argument(
        "--browser-width",
        type=int,
        default=1920,
        help="Browser viewport width (default: 1920)"
    )

    parser.add_argument(
        "--browser-height",
        type=int,
        default=1080,
        help="Browser viewport height (default: 1080)"
    )

    parser.add_argument(
        "--planner-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model to use for planning (default: claude-haiku-4-5-20251001)"
    )

    parser.add_argument(
        "--perceptor-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model to use for vision (default: claude-haiku-4-5-20251001)"
    )

    parser.add_argument(
        "--user-data-dir",
        type=str,
        default=None,
        help="Browser profile directory to persist login sessions (e.g., ./browser_profile)"
    )

    parser.add_argument(
        "--storage-state",
        type=str,
        default=None,
        help="Path to storage state JSON file with saved login session (e.g., ./notion_state.json)"
    )

    return parser


def get_next_run_number(screenshots_dir: Path) -> int:
    """Get the next run number by finding existing numbered subfolders.

    Args:
        screenshots_dir: Base screenshots directory

    Returns:
        Next available run number
    """
    if not screenshots_dir.exists():
        return 1

    # Find all numeric subdirectories
    existing_runs = []
    for item in screenshots_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_runs.append(int(item.name))

    if not existing_runs:
        return 1

    return max(existing_runs) + 1


async def run_task(args: argparse.Namespace) -> bool:
    """Execute the task using Agent B."""

    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("❌ Error: No API key provided.")
        print("Either set ANTHROPIC_API_KEY environment variable or use --api-key flag")
        print()
        print("Get your API key at: https://console.anthropic.com")
        return False

    # Create incremental run subfolder in screenshots directory
    base_screenshots_dir = Path(args.screenshots)
    run_number = get_next_run_number(base_screenshots_dir)
    run_screenshots_dir = base_screenshots_dir / str(run_number)
    run_screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Parse paths if provided
    user_data_dir = Path(args.user_data_dir) if args.user_data_dir else None
    storage_state = Path(args.storage_state) if args.storage_state else None

    # Create configuration
    config = Config(
        anthropic_api_key=api_key,
        headless=args.headless,
        screenshots_dir=run_screenshots_dir,  # Use run-specific subfolder
        chroma_persist_dir=Path(args.chroma_db),
        log_level=args.log_level,
        step_timeout=args.timeout,
        max_retries=args.max_retries,
        similarity_threshold=args.cache_threshold,
        browser_width=args.browser_width,
        browser_height=args.browser_height,
        planner_model=args.planner_model,
        perceptor_model=args.perceptor_model,
        user_data_dir=user_data_dir,
        storage_state=storage_state,
    )

    # Create orchestrator
    orchestrator = Orchestrator(config=config)

    try:
        # Print task info
        print("=" * 70)
        print("🤖 Agent B - Browser Automation Agent")
        print("=" * 70)
        print(f"📋 Task: {args.task}")
        if args.url:
            print(f"🌐 Starting URL: {args.url}")
        print(f"🖥️  Headless: {args.headless}")
        print(f"📸 Screenshots: {run_screenshots_dir} (Run #{run_number})")
        print(f"📊 Log Level: {args.log_level}")
        print(f"🧠 Planner Model: {args.planner_model}")
        print(f"👁️  Perceptor Model: {args.perceptor_model}")
        if user_data_dir:
            print(f"💾 Browser Profile: {user_data_dir}")
        if storage_state:
            if storage_state.exists():
                print(f"🔐 Storage State: {storage_state} (loaded)")
            else:
                print(f"⚠️  Storage State: {storage_state} (not found - will run without login)")
        print("=" * 70)
        print()

        # Start orchestrator
        await orchestrator.start()

        # Navigate to starting URL if provided
        if args.url:
            print(f"Navigating to {args.url}...")
            await orchestrator.navigate_to(args.url)
            print()

        # Execute task
        success = await orchestrator.execute_task(args.task)

        # Print results
        print()
        print("=" * 70)
        if success:
            print("✅ Task completed successfully!")
            print(f"📸 Screenshots saved to: {run_screenshots_dir} (Run #{run_number})")
            if Path(args.chroma_db).exists():
                print(f"💾 Workflow cached for future reuse")
        else:
            print("❌ Task failed")
            print("💡 Try running with --log-level DEBUG for more details")
        print("=" * 70)

        return success

    except KeyboardInterrupt:
        print("\n\n⚠️  Task interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        else:
            print("💡 Run with --log-level DEBUG to see full error details")
        return False

    finally:
        # Cleanup
        print("\nCleaning up...")
        await orchestrator.stop()


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()

    # Handle no arguments case
    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    args = parser.parse_args()

    # Run async task
    success = asyncio.run(run_task(args))

    # Return exit code (0 = success, 1 = failure)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
