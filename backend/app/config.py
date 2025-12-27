"""
Application configuration using Pydantic settings.

Environment variables are loaded from .env file and validated.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI
from qdrant_client import QdrantClient


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database Configuration
    DATABASE_URL: str

    # Gemini API Configuration (via OpenAI SDK)
    GEMINI_API_KEY: str
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"

    # Qdrant Configuration
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "physical_ai_docs"

    # JWT Authentication
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # CORS Configuration
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True

    # Application Settings
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS string into list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# Global settings instance
settings = Settings()


# Lazy initialization functions for clients
def get_gemini_client():
    """Get or create Gemini client using Google Generative AI library."""
    import google.generativeai as genai

    # Configure the API key
    genai.configure(api_key=settings.GEMINI_API_KEY)

    # Create and return the model
    model = genai.GenerativeModel(settings.GEMINI_GENERATION_MODEL)
    return model


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client (lazy initialization)."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
