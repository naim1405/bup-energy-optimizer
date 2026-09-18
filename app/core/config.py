"""Application settings loaded from environment variables / .env file."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration.

    Values can be overridden via environment variables or a local `.env` file.
    The API key is a ``SecretStr`` so it can never be printed, logged, or
    serialized by accident.
    """

    app_name: str = "BUP Energy Optimizer API"
    version: str = "0.1.0"

    # --- LLM (operator-note interpretation) -------------------------------
    #: Model name is configurable so it can be swapped without a code change.
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    #: 0 keeps extraction deterministic; there is nothing to be creative about.
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    #: Per-call timeout. Kept low so one slow call plus one repair retry
    #: still fits inside the judge's 30s per-request limit.
    llm_timeout_seconds: float = Field(default=8.0, alias="LLM_TIMEOUT_SECONDS")
    #: Provider-level retries for transient network errors.
    llm_max_retries: int = Field(default=1, alias="LLM_MAX_RETRIES")
    #: Hard ceiling for the whole interpretation step. Once exceeded we stop
    #: retrying and fall back to no_op rather than risk a request timeout.
    llm_budget_seconds: float = Field(default=24.0, alias="LLM_BUDGET_SECONDS")
    #: Standard OpenAI variable name. Passed explicitly rather than relying
    #: on the SDK to read the process environment, so a `.env` file alone is
    #: enough to configure the service.
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")

    # Reserved for the upcoming queue layer (no code uses this yet).
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
