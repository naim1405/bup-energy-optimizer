"""Application settings loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration.

    Values can be overridden via environment variables or a local `.env` file.
    """

    app_name: str = "BUP Energy Optimizer API"
    version: str = "0.1.0"

    # Reserved for the upcoming queue layer (no code uses this yet).
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
