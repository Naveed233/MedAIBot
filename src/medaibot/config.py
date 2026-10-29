from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set with a MEDAIBOT_ env var."""

    model_config = SettingsConfigDict(env_prefix="MEDAIBOT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    model_cache_dir: Path = Path(".cache/models")

    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    llm_provider: Literal["offline", "openai"] = "offline"
    llm_model: str | None = None
    judge_model: str | None = None
    llm_timeout_s: float = Field(default=60.0, gt=0)
    # The SDK's standard variable name, read without the MEDAIBOT_ prefix.
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    # Point the OpenAI adapter at any OpenAI-compatible server (Ollama, vLLM).
    openai_base_url: str | None = None

    agent_max_steps: int = Field(default=6, ge=1, le=20)
    tool_timeout_s: float = Field(default=15.0, gt=0)
    retrieval_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    min_support: float = Field(default=0.5, ge=0, le=1)

    openfda_base_url: str = "https://api.fda.gov"
    openfda_api_key: SecretStr | None = None
    enable_live_openfda: bool = True

    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("llm_model", "judge_model", "openai_base_url", "openai_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # An empty line in .env such as "MEDAIBOT_LLM_MODEL=" means "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def sources_path(self) -> Path:
        return self.data_dir / "sources.json"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "openfda"


@lru_cache
def get_settings() -> Settings:
    return Settings()
