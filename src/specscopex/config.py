from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env if present (local development)
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_url: str | None
    openai_api_key: str | None
    openai_model: str


def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "local"),
        database_url=os.getenv("DATABASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )
