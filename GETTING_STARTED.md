# Getting Started with Agent B

## TL;DR - Quick Start in 3 Steps

1. **Set your API key** (create `.env` file):
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

2. **Run a task from command line**:
   ```bash
   python run_agentb.py "Go to google.com and search for 'AI agents'"
   ```

3. **That's it!** No Python files to create, no functions to write.

---

## Two Ways to Use Agent B

### ✅ Method 1: CLI (Recommended for Most Users)

**Just describe what you want in natural language:**

```bash
# Example 1: Google search
python run_agentb.py "Go to google.com and search for 'Python tutorials'"

# Example 2: Create Notion database
python run_agentb.py "Navigate to notion.so and create a database named 'Tasks'"

# Example 3: GitHub repo
python run_agentb.py "Go to github.com and create a new repository named 'test-project'"
```

**Advantages:**
- No code to write
- Works immediately
- Perfect for one-off tasks
- Easy to test and iterate

### Method 2: Python API (For Programmatic Use)

**When you need to:**
- Run tasks from your own Python code
- Integrate Agent B into larger applications
- Customize behavior programmatically

```python
import asyncio
from agentb.config import Config
from agentb.orchestrator import Orchestrator

async def main():
    config = Config(anthropic_api_key="your_key")
    orchestrator = Orchestrator(config)

    try:
        await orchestrator.execute_task("Your task here")
    finally:
        await orchestrator.cleanup()

asyncio.run(main())
```

---

## Understanding the Difference

### ❌ What You DON'T Need to Do

You **DON'T** need to create a new Python file for each task like this:

```python
# ❌ You DON'T need to do this anymore!
async def search_google():
    # ... setup code ...
    await orchestrator.execute_task("Search Google")

async def create_notion_db():
    # ... setup code ...
    await orchestrator.execute_task("Create Notion database")

async def create_github_repo():
    # ... setup code ...
    await orchestrator.execute_task("Create GitHub repo")
```

### ✅ What You DO Instead

Just run tasks directly from the command line:

```bash
# ✅ Much simpler!
python run_agentb.py "Search Google for Python tutorials"
python run_agentb.py "Create a Notion database"
python run_agentb.py "Create a GitHub repository"
```

**The agent figures out the steps automatically!**

---

## Common CLI Options

```bash
# Run without showing browser window
python run_agentb.py "Your task" --headless

# Save screenshots to specific folder
python run_agentb.py "Your task" --screenshots ./my_screenshots

# Start at a specific URL
python run_agentb.py "Click the login button" --url https://example.com

# See detailed logs
python run_agentb.py "Your task" --log-level DEBUG

# Increase timeout for slow sites
python run_agentb.py "Your task" --timeout 60
```

See all options:
```bash
python -m agentb.cli --help
```

---

## Examples for Common Tasks

### Web Search
```bash
python run_agentb.py "Go to google.com and search for 'machine learning tutorials'"
```

### Notion Database
```bash
python run_agentb.py "Navigate to notion.so and create a database table named 'Project Tracker' with columns: Task, Status, Priority, Due Date"
```

### GitHub Repository
```bash
python run_agentb.py "Go to github.com and create a new repository named 'my-awesome-project' with description 'A test repo created by Agent B'"
```

### Form Filling
```bash
python run_agentb.py "Navigate to example.com/contact, fill in the contact form with name 'John Doe', email 'john@example.com', and submit"
```

### Navigation
```bash
python run_agentb.py "Go to reddit.com, navigate to r/programming, and upvote the top post"
```

---

## How It Works

1. **You describe the task in plain English**
2. **Agent B checks if it's done this before** (cache)
   - If yes: Executes immediately (0.5s, $0)
   - If no: Generates a plan (3-5s, ~$0.01)
3. **Executes step-by-step:**
   - Tries DOM selectors first (fast)
   - Falls back to vision if needed (accurate)
4. **Validates success** using vision
5. **Saves the workflow** for future reuse

**Next time you run a similar task:** 10x faster, $0 cost!

---

## Troubleshooting

### "No API key provided"
Create a `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
```

Or use the `--api-key` flag:
```bash
python run_agentb.py "Your task" --api-key sk-ant-api03-xxxxx
```

### "Command not found: agentb"
Use the wrapper script instead:
```bash
python run_agentb.py "Your task"
```

Or the module:
```bash
python -m agentb.cli "Your task"
```

### "Task failed"
1. Run with debug logging:
   ```bash
   python run_agentb.py "Your task" --log-level DEBUG
   ```
2. Check the screenshots in `./screenshots/`
3. Try being more specific in your task description

### Login Required
For sites that need login, add a pause:
```bash
python run_agentb.py "Navigate to example.com, wait 30 seconds for me to log in, then create a new post"
```

---

## Cost and Performance

| Metric | First Run | Cached Run |
|--------|-----------|------------|
| Time | 3-8 seconds | 0.1-0.5 seconds |
| Cost | $0.01-0.05 | ~$0.00 |
| LLM Calls | 2-4 | 0 |

**Cache hit rate:** 60-80% for typical workflows

---

## What's Next?

1. **Try the examples** above
2. **Read NOTION_EXAMPLE.md** for detailed Notion database walkthrough
3. **Check README.md** for advanced features
4. **Experiment!** Agent B learns from each task

---

## Key Concepts

- **No code needed:** Just natural language descriptions
- **Smart caching:** Tasks get faster and cheaper over time
- **Self-healing:** Automatically recovers from errors
- **Visual validation:** Uses vision to confirm success
- **Modular:** Use CLI for simplicity, Python API for integration

**Start simple, iterate fast, let Agent B handle the complexity!**
