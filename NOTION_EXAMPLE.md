# Creating a Notion Database with Agent B

## The Simple Way (CLI)

**You don't need to create any Python files!** Just run this command:

### Basic Database

```bash
python run_agentb.py "Navigate to notion.so, log in, create a new database table named 'Project Tracker' with columns: Task Name, Status (select), Priority (select), and Due Date"
```

### What You'll See

```
======================================================================
🤖 Agent B - Browser Automation Agent
======================================================================
📋 Task: Navigate to notion.so, log in, create a new database...
🖥️  Headless: False
📸 Screenshots: ./screenshots
📊 Log Level: INFO
🧠 Planner Model: claude-3-5-sonnet-20241022
👁️  Perceptor Model: claude-3-5-sonnet-20241022
======================================================================

[Orchestrator] Querying skills library...
[SkillsLibrary] No cached plan found (first time)
[Planner] Generating initial plan...
[Planner] Generated 8-step plan

[Orchestrator] Executing Step 1: NAVIGATE
[Executor] Navigating to: https://notion.so
✓ Step 1 complete

[Orchestrator] Executing Step 2: CLICK
[Executor] Finding element: 'New page' button
✓ Step 2 complete

... (continues for all steps)

======================================================================
✅ Task completed successfully!
📸 Screenshots saved to: ./screenshots
💾 Workflow cached for future reuse
======================================================================
```

---

## CLI Options

### Run Without Browser Window (Headless)

```bash
python run_agentb.py "Create a Notion database" --headless
```

### Save Screenshots to Custom Location

```bash
python run_agentb.py "Create a Notion database" --screenshots ./notion_screenshots
```

### Pre-navigate to Notion

```bash
python run_agentb.py "Create a new database table named 'Tasks'" --url https://notion.so
```

### Enable Debug Logging

```bash
python run_agentb.py "Create a Notion database" --log-level DEBUG
```

### Combine Multiple Options

```bash
python run_agentb.py "Create a Notion database table with columns for tasks and due dates" \
  --headless \
  --screenshots ./notion_db_screenshots \
  --log-level INFO \
  --timeout 45
```

---

## Detailed Example: Product Roadmap Database

Create a sophisticated database with multiple column types:

```bash
python run_agentb.py "Navigate to notion.so and create a new database table named 'Product Roadmap 2024' with the following columns:
- Feature Name (title/text)
- Status (select with options: Backlog, In Progress, Testing, Shipped)
- Priority (select with options: P0 Critical, P1 High, P2 Medium, P3 Low)
- Owner (person)
- Due Date (date)
- Quarter (select with options: Q1 2024, Q2 2024, Q3 2024, Q4 2024)
- Effort (select with options: Small, Medium, Large, XL)
- Tags (multi-select with options: Frontend, Backend, Design, Infrastructure)"
```

---

## All Available Options

Run `python -m agentb.cli --help` to see all options:

```
Options:
  --url URL                     Starting URL to navigate to before executing task
  --api-key API_KEY             Anthropic API key (or set ANTHROPIC_API_KEY env var)
  --headless                    Run browser in headless mode (no visible window)
  --screenshots SCREENSHOTS     Directory to save screenshots (default: ./screenshots)
  --chroma-db CHROMA_DB         Directory for ChromaDB persistence (default: ./chroma_db)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                                Logging level (default: INFO)
  --timeout TIMEOUT             Step timeout in seconds (default: 30.0)
  --max-retries MAX_RETRIES     Maximum step retry attempts (default: 3)
  --cache-threshold CACHE_THRESHOLD
                                Similarity threshold for cache hits (default: 0.95)
  --browser-width BROWSER_WIDTH
                                Browser viewport width (default: 1920)
  --browser-height BROWSER_HEIGHT
                                Browser viewport height (default: 1080)
  --planner-model PLANNER_MODEL
                                Model to use for planning (default: claude-3-5-sonnet-20241022)
  --perceptor-model PERCEPTOR_MODEL
                                Model to use for vision (default: claude-3-5-sonnet-20241022)
```

---

## FAQ

### Q: Do I need to log in manually?

**A:** Yes, on the first run, the browser will open and wait for you to log in to Notion. You can add a pause in your task:

```bash
python run_agentb.py "Navigate to notion.so, wait 30 seconds for me to log in, then create a new database"
```

### Q: Will it remember my login next time?

**A:** Currently, you'll need to log in each time. To save browser state (cookies), this feature would need to be added to the config.

### Q: What if the task fails?

**A:** Agent B automatically retries with a corrected plan. If it still fails, run with `--log-level DEBUG` to see detailed error information.

### Q: Can I run multiple tasks?

**A:** Yes! Just run the command multiple times. The second time will be 10x faster because the workflow is cached:

```bash
# First time: 5-10 seconds
python run_agentb.py "Create a Notion database"

# Second time: 0.5-1 second (uses cached plan)
python run_agentb.py "Create a Notion database"
```

### Q: How much does this cost?

**A:** First run: ~$0.01-0.05 in API costs. Subsequent runs: ~$0.00 (uses cache).

---

## Step-by-Step: Your First Notion Database

1. **Make sure you have your API key in `.env`:**
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
   ```

2. **Run the command:**
   ```bash
   python run_agentb.py "Navigate to notion.so and create a database table named 'My Tasks'"
   ```

3. **Watch the browser:**
   - Browser will open to notion.so
   - Log in if needed (agent waits)
   - Agent will click "New page"
   - Agent will select "Table - Database"
   - Agent will name it "My Tasks"
   - Done!

4. **Check the results:**
   - Your Notion database is created
   - Screenshots saved to `./screenshots/`
   - Workflow cached for next time

5. **Run it again (10x faster):**
   ```bash
   python run_agentb.py "Create a Notion database table named 'Weekly Goals'"
   ```

---

## Next Steps

- Try different database structures
- Experiment with adding rows/data
- Combine with other tasks (e.g., "Create database and add 3 sample tasks")
- Use `--headless` mode once you're confident it works

---

**That's it!** No Python files to create, no functions to write. Just describe what you want in natural language and Agent B figures out the rest.
