"""Optional, explicit generation deadlines; historical dates are not recurring curfews."""
from datetime import datetime, timezone


def deadline_reached(value, now=None):
    if value is None:
        return False
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        raise ValueError("deadline must include its UTC offset, e.g. 2026-11-09T22:50:00+08:00")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current >= deadline
