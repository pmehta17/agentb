# CLAUDE.md - AI Assistant Development Guide

This document provides essential information for AI assistants working on the Agent B (Reflexive UI Automation Agent) codebase.

## Project Overview

Agent B is a reflexive AI agent that receives natural language tasks (e.g., "Create a project in Linear") and executes them by navigating live web applications. The agent can fail, understand failures, re-plan to recover, and learn from successful executions.

**Core Principles:**
- **Generalizable:** Works on unseen tasks and applications
- **Robust:** Handles dynamic UI without hardcoded selectors
- **Reflexive:** Fails gracefully, understands failures, and re-plans
- **Efficient:** Learns from successes to reduce latency and cost

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ with asyncio |
| **Browser Automation** | Playwright |
| **Vector Database** | ChromaDB |
| **Embeddings** | sentence-transformers |
| **LLM Integration** | Anthropic Claude API |
| **Vision/Multimodal** | Claude 3 Vision |
| **Data Validation** | Pydantic |
| **Logging** | structlog |
| **Image Processing** | Pillow |
| **Retry Logic** | tenacity |

## Directory Structure

```
/agentb
├── /agent                  # Core source code
│   ├── __init__.py
│   ├── orchestrator.py     # Main execution loop
│   ├── skills.py           # Skills library (ChromaDB)
│   ├── planner.py          # LLM planning
│   ├── executor.py         # Playwright browser actions
│   ├── perceptor.py        # Vision model fallback
│   ├── capturer.py         # Screenshot/state change detection
│   └── utils.py            # Helpers and exceptions
├── /dataset                # Output screenshots
├── /db                     # ChromaDB storage
├── .claude                 # Detailed architecture guide
├── .env                    # API keys (not committed)
├── config.py               # Configuration settings
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── CLAUDE.md               # This file
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run async tests
pytest --asyncio-mode=auto
```

### Code Quality
```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

## Architecture Overview

The system uses a **closed-loop architecture** with six modules:

1. **Orchestrator** - Central nervous system, manages execution loop
2. **Skills Library** - Long-term memory, caches successful plans
3. **Planner** - High-level reasoning, generates/refines plans via LLM
4. **Executor** - Browser automation via Playwright (no AI logic)
5. **Perceptor** - Vision model for complex element detection
6. **State Capturer** - Detects UI state changes via screenshots

## Key Development Rules

### Cache-First Strategy
Always query the Skills Library before calling the Planner. This reduces latency and API costs.

### DOM-First Perception
For every step, try `Executor.find_element_by_text()` before using the expensive Vision Perceptor.

### Re-Planning on Failure
Wrap all execution steps in try-except. On failure, bundle context and call `Planner.regenerate_plan()`.

### Executor is "Dumb"
The Executor module contains NO AI logic - only low-level browser actions.

## Rich Action Schema

All plans must follow this JSON structure:

```json
[
  {
    "step": 1,
    "action": "CLICK | TYPE | NAVIGATE | WAIT",
    "target_description": "descriptive text for UI element",
    "value": "text for TYPE actions or null",
    "semantic_role": "primary_action | form_field | confirmation",
    "required_state": "UI state needed before this step"
  }
]
```

## Key Workflows

### Cache Hit (Efficient Path)
1. Task received -> Skills Library finds match
2. Execute cached plan using DOM-first approach
3. Validate success -> Task complete
4. **Result:** Zero Planner API calls

### Cache Miss with Re-planning
1. Task received -> Skills Library returns None
2. Planner generates initial plan
3. Execute steps, handle failures with re-planning
4. Validate success
5. Save new skill to library
6. **Result:** Agent learned new capability

## Configuration

### Environment Variables (.env)
```bash
ANTHROPIC_API_KEY=your_key_here
```

### Key Thresholds
- **Skill match similarity:** 0.95 (high confidence)
- **State change detection:** 2% pixel difference
- **Network idle timeout:** Wait for networkidle event

## Code Conventions

### Async-First Design
All I/O operations use `async/await`:
```python
async def execute_step(self, step: dict) -> bool:
    ...
```

### Type Hints
Use Pydantic models and type hints throughout:
```python
from pydantic import BaseModel

class ActionStep(BaseModel):
    step: int
    action: str
    target_description: str
    value: Optional[str]
```

### Structured Logging
Use structlog for consistent logging:
```python
import structlog
logger = structlog.get_logger()
logger.info("executing_step", step=1, action="CLICK")
```

### Retry Logic
Use tenacity for resilient API calls:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def call_api(self):
    ...
```

## Module Implementation Guidelines

### orchestrator.py
- Implements main async execution loop
- Enforces cache-first and DOM-first rules
- Handles error recovery and re-planning
- Saves successful new skills

### skills.py
- Uses ChromaDB for vector storage
- sentence-transformers for embeddings
- High similarity threshold for retrieval

### planner.py
- Calls Anthropic Claude API (text model)
- Generates Rich Action Schema plans
- Validates task completion

### executor.py
- Playwright browser automation
- Returns coordinates only on single element match
- No AI logic - pure automation

### perceptor.py
- Uses Claude Vision for element detection
- Returns descriptive failures for re-planning
- Expensive fallback for complex UIs

### capturer.py
- Screenshot comparison with Pillow
- Pixel-level change detection
- Waits for UI stability

## Testing Guidelines

- Write async tests using pytest-asyncio
- Mock external APIs (Anthropic, browser)
- Test cache hit and miss paths
- Test re-planning scenarios
- Aim for high coverage on core logic

## Common Tasks

### Adding a New Action Type
1. Update Rich Action Schema in planner.py
2. Add handler in executor.py
3. Update validation in orchestrator.py
4. Add tests

### Adjusting Thresholds
- Skill similarity: skills.py
- State change detection: capturer.py
- Retry parameters: per-module decorators

### Debugging Failures
1. Check structlog output for step details
2. Review screenshots in /dataset
3. Examine re-planning context sent to Planner

## Important Notes

- **Status:** Project is in architecture phase - source code not yet implemented
- **Architecture Guide:** See `.claude` for detailed module specifications
- **Dependencies:** All listed in requirements.txt with version constraints
- Always validate against the Rich Action Schema
- Keep the Executor simple - complexity belongs in Planner/Perceptor
