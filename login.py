#!/usr/bin/env python
"""
Login Helper for Agent B

This script opens a browser and lets you log in manually at your own pace.
Once you're logged in, it saves the browser state (cookies, tokens, etc.) to a JSON file.
You can then use this state file with Agent B to stay logged in automatically.

Usage:
    python login.py --url https://notion.so --output notion_state.json
    python login.py --url https://github.com --output github_state.json

Then use with Agent B:
    python run_agentb.py "Create a database" --storage-state notion_state.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def login_and_save_state(url: str, output_file: Path) -> None:
    """Open browser for manual login and save the session state.

    Args:
        url: URL to navigate to for login
        output_file: Path where storage state JSON will be saved
    """
    print("=" * 70)
    print("🔐 Agent B - Login Helper")
    print("=" * 70)
    print(f"Opening browser to: {url}")
    print(f"Storage state will be saved to: {output_file}")
    print("=" * 70)
    print()

    async with async_playwright() as p:
        # Launch browser in non-headless mode for manual interaction
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = await context.new_page()

        # Navigate to the login URL
        print(f"Navigating to {url}...")
        await page.goto(url)
        print()

        # Wait for user to complete login
        print("=" * 70)
        print("👤 Please log in to the website in the browser")
        print("=" * 70)
        print()
        print("Instructions:")
        print("  1. Complete the login process in the browser window")
        print("  2. Make sure you're fully logged in and on a logged-in page")
        print("  3. Come back to this terminal")
        print()
        input("Press ENTER when you're logged in and ready to save the session...")
        print()

        # Save the storage state
        print("💾 Saving browser state...")
        await context.storage_state(path=str(output_file))

        print()
        print("=" * 70)
        print("✅ Login session saved successfully!")
        print("=" * 70)
        print()
        print(f"Storage state saved to: {output_file}")
        print()
        print("You can now use this state with Agent B:")
        print(f"  python run_agentb.py \"Your task\" --storage-state {output_file}")
        print()
        print("Note: Keep this file secure - it contains your login credentials!")
        print("=" * 70)

        await browser.close()


def main() -> int:
    """Main entry point for login helper."""
    parser = argparse.ArgumentParser(
        description="Agent B Login Helper - Save browser login state for reuse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python login.py --url https://notion.so --output notion_state.json
  python login.py --url https://github.com --output github_state.json
  python login.py --url https://app.example.com --output example_state.json

After saving the state, use it with Agent B:
  python run_agentb.py "Create a database" --storage-state notion_state.json
  python run_agentb.py "Create a repository" --storage-state github_state.json
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL to navigate to for login (e.g., https://notion.so)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where storage state JSON will be saved (e.g., notion_state.json)",
    )

    args = parser.parse_args()

    # Validate output path
    output_path = Path(args.output)
    if output_path.exists():
        response = input(
            f"⚠️  File {output_path} already exists. Overwrite? (y/N): "
        )
        if response.lower() != "y":
            print("Cancelled.")
            return 1

    # Run the login workflow
    try:
        asyncio.run(login_and_save_state(args.url, output_path))
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Login cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
