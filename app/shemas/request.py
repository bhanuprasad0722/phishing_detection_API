# app/schemas/request.py
from pydantic import BaseModel, HttpUrl, validator
from typing import Literal

class URLRequest(BaseModel):
    url: str

    @validator("url")
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError("URL cannot be empty")
        if len(v) > 2000:
            raise ValueError("URL too long (max 2000 chars)")
        # Basic scheme check
        if not v.startswith(("http://", "https://")):
            v = "http://" + v
        return v

class PredictionResponse(BaseModel):
    url: str
    phishing_probability: float
    is_phishing: bool
    confidence: float
    label: Literal["PHISHING", "LEGITIMATE"]
    model: str
    features_extracted: int
    timestamp: str = None

    class Config:
        json_encoders = {
            # Auto format timestamp if you add it later
        }