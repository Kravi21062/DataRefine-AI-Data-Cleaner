from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_title: str = "DataRefine"
    app_version: str = "2.0.0"
    environment: str = "development"
    frontend_origin: str = "http://localhost:5173"
    allowed_origins: str = "http://localhost:5173"
    max_upload_mb: int = 200
    session_ttl_minutes: int = 60
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "phi3"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
