# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.core.security import SecurityHeadersMiddleware, rate_limit_exceeded_handler
from app.api.v1.endpoints.predict import router as predict_router
from app.dependencies import limiter
from app.utils.logger import logger
import tensorflow as tf

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = FastAPI(
    title="PhiKitA Pro - Production Phishing Detection API",
    description="""
    Major Project by Bhanu Prasad | 2025  
    ABS-CNN with Self-Attention + 17 Handcrafted Features  
    99%+ Accuracy | Real-time Inference | Production Ready
    """,
    version="2.0.0",
    contact={
        "name": "Bhanu Prasad",
        "url": "https://github.com/bhanuprasad0722"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# === Middlewares ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "*.onrender.com", "127.0.0.1"])
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(429, rate_limit_exceeded_handler)

# === Routes ===
@app.get("/", tags=["Health"])
@limiter.limit("60/minute")
async def root(request: Request):
    return {
        "message": "PhiKitA Pro is LIVE & Secured!",
        "version": "2.0.0",
        "docs": "/docs",
        "author": "Bhanu Prasad",
        "accuracy": "99%+ on unseen phishing kits"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "model": "ABS-CNN loaded"}

app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])

logger.success("PhiKitA Production API Started Successfully!")