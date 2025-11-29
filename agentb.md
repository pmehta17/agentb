# 1.0 Introduction

## 1.1 Project Goal
To build a **reflexive** AI agent (Agent B) that can receive open-ended, natural language tasks from another agent (Agent A) (e.g., "Create a project in Linear," "Filter a database in Notion").  
This system must be capable of:

- Generalizing to unseen tasks and applications.  
- Navigating live web applications programmatically.  
- Handling dynamic UI states that do not have unique URLs (e.g., modals, form fields, menus).  
- Capturing a visual dataset of screenshots for each distinct UI state in the workflow.  
- Learning from successful task executions to improve future performance.  

## 1.2 Core Solution
The solution is a **closed-loop, modular** agent architecture. It separates high-level planning from low-level execution and incorporates:

- A semantic cache (Skills Library) for efficiency  
- A DOM-first / Vision-fallback perception model for robustness  
- A re-planning loop to recover from errors  

---

# 2.0 System Architecture

The system is comprised of six core modules managed by a central **Orchestrator**.

### Modules Overview
- **Orchestrator** – Central nervous system for planning, execution, and error recovery  
- **Skills Library** – Long-term semantic cache of successful workflows  
- **Planner (LLM)** – High-level reasoning  
- **Executor (Browser)** – Low-level actions (click, type, navigate)  
- **Perceptor (VLM)** – Vision system for detecting UI elements  
- **State Capturer (Observer)** – Captures UI state transitions visually  

---

# 3.0 Component Requirements

## 3.1 Orchestrator
Central Python script running the main async loop.

### Responsibilities
- Receive the initial natural language task  
- **Skill Query**: generate task embedding → query the Skills Library  
- **Planner Fallback**: call Planner when no cached plan exists  
- Execute plan step-by-step  
- **DOM-first perception** via `Executor.find_element_by_text()`  
- **Vision fallback** via `Perceptor.find_element()`  
- **Re-planning loop** on error:  
  - bundle context (goal, plan history, error, screenshot)  
  - call `Planner.regenerate_plan()`  
- Validate task success with `Planner.validate_success()`  
- Save new skill with `SkillsLibrary.add_skill()`  

---

## 3.2 Skills Library (Workflow Memory)
Long-term workflow memory implemented as a vector database.

### Methods

**`add_skill(task: str, plan: dict)`**
- Generate sentence-transformer embedding  
- Save plan JSON to document store  
- Save embedding in vector index  

**`find_skill(task_embedding: list) -> Optional[dict]`**
- Perform semantic similarity search  
- If similarity > 0.95 → return stored plan  
- Else → return `None`  

---

## 3.3 Planner (LLM Agent)
High-level reasoning module (text-only LLM).

### Responsibilities
- **`generate_initial_plan(task)`** → baseline structured JSON plan  
- **`regenerate_plan(context_bundle)`** → corrected plan using failure context  
- **`validate_success(goal, final_screenshot)`** → return True/False  

### Rich Action Schema (Required)

```json
[
  {
    "step": 1,
    "action": "CLICK", // CLICK, TYPE, SELECT, NAVIGATE
    "target_description": "the 'New Project' button in the left sidebar",
    "value": null, // or "My New AI Project" for TYPE actions
    "semantic_role": "primary_action", // primary_action, navigation, confirmation, form_field
    "required_state": "projects_list_visible"
  }
]
```

---

## 3.4 Perceptor (Multimodal LLM Agent)
Vision-based search and error explanation.

### Responsibilities
**`find_element(screenshot: bytes, step: dict)`**
- Receive screenshot + full step context  
- Return `{ "x": ..., "y": ... }` coordinates  
- On failure: return structured Failure object  

---

## 3.5 Executor (Browser Agent)
Wrapper around Playwright.

### Responsibilities
- `click(x, y)`  
- `type(x, y, text)`  
- `navigate(url)`  
- `get_screenshot()`  
- **`find_element_by_text(text) -> Optional[dict]`**  
  - Return coordinates if 1 unique match  
  - Return `None` if 0 or >1 matches  

---

## 3.6 State Capturer (Observer)
Handles capturing **non-URL UI states**.

### Responsibilities
**`capture_state(step_name)`**
- Save current screenshot with descriptive name  

**`wait_for_change()`**
- Capture "before" screenshot  
- Wait for `networkidle` load state  
- Poll screenshots + compare using pixelmatch  
- Exit when diff > threshold (e.g., 2%)  

---

# 4.0 Core Workflows

## 4.1 Workflow 1: Cache Hit (Efficient Path)
**Task:** "Make a new Notion page."

1. Orchestrator queries Skills Library → match found  
2. Cached JSON plan returned  
3. Orchestrator executes plan step-by-step  
4. `Planner.validate_success()` returns True  

**Result:** Low latency, minimal LLM usage.

---

## 4.2 Workflow 2: Cache Miss & Re-planning (Robust Path)
**Task:** "Archive my 'Q4-bugs' project in Asana."

1. Skills Library → no match  
2. Planner generates initial plan  
3. Execute Step 1  
4. Step 2 fails (DOM search returns None)  
5. Perceptor returns Failure object  
6. Orchestrator bundles failure context → calls re-plan  
7. Planner returns corrected plan  
8. Corrected plan succeeds  
9. Validation = True  
10. Orchestrator saves new skill  

**Result:** The agent recovers and learns a new skill.

---

# 5.0 Technology Stack

| Module | Technology | Purpose |
|--------|------------|---------|
| Orchestration | Python 3.10+ (`asyncio`) | Main asynchronous control loop |
| Skills Library | ChromaDB / FAISS | Vector DB for semantic caching |
|  | sentence-transformers | Local embedding generation |
| Planner | OpenAI API / Anthropic API (text) | Planning + validation |
| Perceptor | OpenAI API / Anthropic API (vision) | Visual element search |
| Executor | Playwright | Browser automation |
| State Capturer | Pillow (PIL) + pixelmatch | Screenshot diffing |


