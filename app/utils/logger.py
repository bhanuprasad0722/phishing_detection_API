# app/utils/logger.py
from loguru import logger
import sys

logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="ERROR")

# Create logs folder if not exists
import os
if not os.path.exists("logs"):
    os.makedirs("logs")