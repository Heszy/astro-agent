from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    astro_data_path: Path = Path("data/sample_catalog.csv")
    astro_db_path: Path = Path("data/astro_catalog.duckdb")
    astro_dataset_id: str = "default"
    astro_catalog_version: str = "latest"
    astro_results_dir: Path = Path("results")
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.astro_results_dir.mkdir(parents=True, exist_ok=True)
    return settings
