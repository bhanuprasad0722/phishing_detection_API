from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "PhiKitA - Advanced Phishing Detector"
    VERSION: str = "2.0.0"
    MODEL_PATH: str = "app/models/model2.h5"
    SCALER_PATH: str = "app/models/scaler.pkl"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()