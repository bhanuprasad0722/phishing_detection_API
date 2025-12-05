# app/dependencies.py
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limiter (100 requests per minute per IP — perfect for free Render tier)
limiter = Limiter(key_func=get_remote_address)

def get_limiter():
    return limiter