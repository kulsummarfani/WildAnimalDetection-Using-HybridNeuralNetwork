"""Server-side Supabase Auth and database access for VanRakshak AI."""

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

ROLES = ("admin", "watchman", "resident")
CACHE_TTL_SECONDS = 30
_principal_cache = {}


class SupabaseConfigError(RuntimeError):
    """A required Supabase environment variable is missing."""


class InvalidToken(Exception):
    """The bearer token is missing, expired, or not accepted by Supabase Auth."""


def _env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise SupabaseConfigError(f"{name} is not set. Add it to .env (see supa.py).")
    return value


def public_config():
    return {"supabase_url": _env("SUPABASE_URL"), "supabase_anon_key": _env("SUPABASE_ANON_KEY")}


@lru_cache(maxsize=1)
def client():
    from supabase import create_client
    return create_client(_env("SUPABASE_URL"), _env("SUPABASE_SERVICE_KEY"))


def get_principal(token):
    if not token:
        raise InvalidToken("missing token")
    cache_key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    cached = _principal_cache.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    from supabase import AuthError
    db = client()
    try:
        response = db.auth.get_user(token)
    except AuthError as exc:
        if (getattr(exc, "status", None) or 0) >= 500:
            raise
        raise InvalidToken(str(exc)) from exc
    user = getattr(response, "user", None)
    if user is None:
        raise InvalidToken("no user for token")

    rows = db.table("profiles").select("username,role").eq("id", user.id).limit(1).execute().data
    profile = rows[0] if rows else {}
    role = profile.get("role") if profile.get("role") in ROLES else "resident"
    principal = {
        "id": user.id,
        "email": user.email,
        "username": profile.get("username") or (user.email or "").split("@")[0],
        "role": role,
    }
    if len(_principal_cache) > 1000:
        _principal_cache.clear()
    _principal_cache[cache_key] = (now + CACHE_TTL_SECONDS, principal)
    return principal


def insert_alert(row):
    return client().table("alerts").insert(row).execute().data


def list_alerts(limit=200):
    rows = client().table("alerts").select("*").order("created_at", desc=True).limit(limit).execute().data
    return list(reversed(rows or []))


def last_sms_time(species, since_iso):
    """Return the newest sent-SMS timestamp for a species in the time window."""
    rows = (client().table("alerts").select("created_at")
            .eq("species", species).eq("sms_sent", "true").gte("created_at", since_iso)
            .order("created_at", desc=True).limit(1).execute().data)
    return rows[0]["created_at"] if rows else None
