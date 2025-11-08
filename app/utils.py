import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

CHINA_TZ = timezone(timedelta(hours=8))


def now_utc() -> datetime:
    """Return current time in Beijing timezone."""
    return datetime.now(CHINA_TZ)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(CHINA_TZ).isoformat()


def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return "pbkdf2_sha256${}${}".format(
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(derived).decode("utf-8"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt_b64, hash_b64 = stored.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except Exception:
        return False
    calculated = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(calculated, expected)


__all__ = ["now_utc", "to_iso", "parse_iso8601", "hash_password", "verify_password"]
