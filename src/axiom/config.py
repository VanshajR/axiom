import os
from dataclasses import dataclass, field


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    cerebras_api_key: str = os.getenv("CEREBRAS_API_KEY", "")
    # where the site publishes its data (data/*.yaml -> public/profile.json)
    profile_url: str = os.getenv("PROFILE_URL", "https://vanshajraghuvanshi.dev/profile.json")
    allowed_origins: list[str] = field(
        default_factory=lambda: _list(os.getenv("ALLOWED_ORIGINS", "https://vanshajraghuvanshi.dev"))
    )
    # visitor-facing limits that keep the free tiers alive
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "12"))
    max_message_chars: int = int(os.getenv("MAX_MESSAGE_CHARS", "1200"))
    max_turns: int = int(os.getenv("MAX_AGENT_TURNS", "5"))


settings = Settings()
