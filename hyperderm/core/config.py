from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    bytez_api_key: str
    bytez_model: str = "ASAIs-TDDI-2025/MedTurk-MedGemma-4b"

    ncbi_api_key: str | None = None
    crossref_mailto: str = "you@example.com"

    dataset_csv_path: str | None = None
    backup_dir: str = "backup"
    bytez_timeout_seconds: int = 45
    build_resume: bool = True
    clear_graph_on_start: bool = True
    strict_neo4j_only: bool = False


settings = Settings()
