from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_env: str = "development"
    app_name: str = "Industrial Predictive Maintenance Intelligence"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    api_key: str = ""
    api_key_header: str = "X-API-Key"
    allowed_origins: list[str] | str = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8080"]
    )
    database_url: str = "sqlite:///./industrial_pm.db"
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    model_registry_root: Path = Path("model_registry")
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "cmapss-fd001"
    model_stage: str = "champion"
    random_seed: int = 42

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production")
        return normalized

    def validate_secure_production(self) -> None:
        if self.app_env == "production" and not self.api_key:
            raise ValueError("API_KEY is required in production")
        if self.app_env == "production" and "change-me" in self.database_url:
            raise ValueError("Default database credentials are forbidden in production")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_secure_production()
    return settings
