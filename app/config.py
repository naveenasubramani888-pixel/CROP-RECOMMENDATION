"""
Centralized configuration management for AgriSense AI FastAPI application.
"""

import os
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AgriSense AI"
    PROJECT_TAGLINE: str = "Smart Crop Recommendations Powered by AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./agrisense.db"

settings = Settings()
