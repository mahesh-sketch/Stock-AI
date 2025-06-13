# Handles Loading .env variables

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

settings = Settings()