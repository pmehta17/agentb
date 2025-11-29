"""Configuration management for Agent B."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # API Keys
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Model settings
    planner_model: str = Field(default="claude-haiku-4-5-20251001")
    perceptor_model: str = Field(default="claude-haiku-4-5-20251001")

    # Skills Library settings
    chroma_persist_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        )
    )
    skill_similarity_threshold: float = Field(default=0.95)
    similarity_threshold: float = Field(default=0.95)  # Alias for compatibility

    # Screenshot settings
    screenshots_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("SCREENSHOTS_DIR", "./data/screenshots")
        )
    )

    # State change detection
    pixelmatch_threshold: float = Field(
        default=0.02, description="Minimum diff ratio to detect state change"
    )
    state_change_timeout: float = Field(
        default=10.0, description="Max seconds to wait for state change"
    )
    state_change_poll_interval: float = Field(
        default=0.5, description="Seconds between state change polls"
    )

    # Browser settings
    headless: bool = Field(default=False)
    viewport_width: int = Field(default=1280)
    viewport_height: int = Field(default=720)
    browser_width: int = Field(default=1920)  # Alias for viewport_width
    browser_height: int = Field(default=1080)  # Alias for viewport_height
    user_data_dir: Path | None = Field(default=None, description="Browser profile directory for persistent sessions")
    storage_state: Path | None = Field(default=None, description="Path to storage state JSON for session persistence")

    # Retry settings
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    step_timeout: float = Field(default=30.0)  # Timeout for individual steps

    # Logging settings
    log_level: str = Field(default="INFO")

    class Config:
        arbitrary_types_allowed = True
