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


__all__ = ["now_utc", "to_iso", "parse_iso8601"]
