"""
DHUN AI - Configuration Module
Centralises all app configuration with environment variable loading.
"""

import os
import base64
import hashlib
from dotenv import load_dotenv

# Load .env in local dev; Vercel injects env vars natively in production
load_dotenv()


class Config:
    # ── Core ──────────────────────────────────────────────────────────
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dhun-ai-secret-dev-key-2024")
    JWT_SECRET: str = os.environ.get("JWT_SECRET", "dhun-jwt-dev-secret-2024")
    ADMIN_PIN: str = os.environ.get("ADMIN_PIN", "2015")
    FLASK_ENV: str = os.environ.get("FLASK_ENV", "production")

    # ── Database ──────────────────────────────────────────────────────
    MONGO_URI: str = os.environ.get("MONGO_URI", "")
    DB_NAME: str = "dhun_ai"

    # ── AI / OpenRouter ───────────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_REFERER: str = "https://dhun-ai.vercel.app"
    OPENROUTER_TITLE: str = "DHUN AI Music Generator"

    # AI models (overridable from admin panel)
    LYRICS_MODEL: str = os.environ.get("LYRICS_MODEL", "openrouter/free")
    COVER_MODEL: str = os.environ.get("COVER_MODEL", "pollinations")
    AUDIO_MODEL: str = os.environ.get("AUDIO_MODEL", "procedural")
    SUNO_COOKIE: str = os.environ.get("SUNO_COOKIE", "")

    # ── Encryption ────────────────────────────────────────────────────
    @staticmethod
    def get_fernet_key() -> bytes:
        """Derive a stable Fernet key from SECRET_KEY (deterministic, no extra env var)."""
        secret = os.environ.get("SECRET_KEY", "dhun-ai-secret-dev-key-2024")
        raw = hashlib.sha256(secret.encode("utf-8")).digest()  # 32 bytes
        return base64.urlsafe_b64encode(raw)

    # ── Face Recognition ──────────────────────────────────────────────
    FACE_MATCH_THRESHOLD: float = float(os.environ.get("FACE_THRESHOLD", "0.55"))

    # ── Token TTL ─────────────────────────────────────────────────────
    TOKEN_EXPIRE_DAYS: int = 7


config = Config()
