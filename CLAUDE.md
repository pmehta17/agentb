# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent B is a reflexive AI agent system that receives natural language tasks and executes them through browser automation. It uses a closed-loop architecture with semantic caching, DOM-first perception with vision fallback, and a re-planning loop for error recovery.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Run tests
pytest

# Run single test file
pytest tests/test_orchestrator.py

# Run with coverage
pytest --cov=src

# Code formatting
black src tests

# Linting
ruff check src tests

# Type checking
mypy src
```

## Architecture

The system comprises six core modules managed by a central Orchestrator:

- **Orchestrator** - Main async loop coordinating all components. Handles skill queries, planner fallback, step execution, and re-planning on errors
- **Skills Library** - Vector database (ChromaDB) for semantic caching of successful workflows using sentence-transformers embeddings
- **Planner** - Text-only LLM for high-level reasoning. Generates structured JSON plans with rich action schema
- **Perceptor** - Vision LLM for visual element search when DOM lookup fails
- **Executor** - Playwright wrapper for browser actions (click, type, navigate). Uses DOM-first element finding
- **State Capturer** - Screenshot capture and comparison using pixelmatch for detecting UI state changes

### Plan Action Schema

Plans use this JSON structure:
```json
{
  "step": 1,
  "action": "CLICK|TYPE|SELECT|NAVIGATE",
  "target_description": "description of UI element",
  "value": null,
  "semantic_role": "primary_action|navigation|confirmation|form_field",
  "required_state": "expected_ui_state"
}
```

### Execution Flow

1. Task received → Query Skills Library (similarity > 0.95 = cache hit)
2. Cache miss → Planner generates initial plan
3. Execute steps: DOM lookup first → Vision fallback on failure
4. On error: Bundle context → Planner re-generates plan
5. Success validation → Save new skill to library

## Key Dependencies

- **Playwright** - Browser automation
- **ChromaDB** - Vector database for skills
- **sentence-transformers** - Embedding generation
- **Anthropic API** - LLM for planning and vision
- **Pillow + pixelmatch** - Screenshot diffing
