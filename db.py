"""
db.py — Supabase Postgres persistence for generation + feedback logging.

Migrated from an earlier SQLite implementation. The schema (generations,
comments) already exists in Supabase, created directly via SQL — this module
only owns the connection layer and queries, never table creation or
migrations. Every function signature and return shape below is unchanged
from the SQLite version, so app.py needed no changes for this migration.

SQLite is no longer supported: DATABASE_URL is required to log anything, and
there is no local-file fallback. As before, each function opens and closes
its own connection rather than holding one open or pooling — this task is
scoped to swapping the connection layer, not adding resilience features.

Logging must never break the actual user-facing request it's attached to —
every write function below catches its own exceptions, prints a warning, and
returns None/False instead of raising, so a DB hiccup (including Supabase
being briefly unreachable) degrades to "this one row didn't get logged"
rather than a 500 on /api/generate.
"""

import os
import psycopg2
import psycopg2.extras

DATABASE_URL = os.getenv("DATABASE_URL")


def _get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — cannot connect to the database.")
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    """Tables already exist in Supabase — this no longer creates anything.
    It only validates connectivity at startup and logs clearly on failure,
    without ever raising: a missing or briefly unreachable database must not
    prevent the Flask app itself from starting and serving requests, just
    degrade logging until connectivity returns."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            conn.close()
    except Exception as e:
        print(f"WARNING: database connectivity check failed at startup: {e}")


def insert_generation(user_name: str, raw_prompt: str, business_logic: str,
                       languages_requested: str, language_prompts: dict, model_used: str):
    """Logs one /api/generate call. Returns the new row id, or None if the
    write failed for any reason — callers must treat None as 'not logged,
    proceed anyway,' never as a request-ending error."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO generations "
                    "(user_name, raw_prompt, business_logic, languages_requested, "
                    "language_prompts, model_used) VALUES (%s, %s, %s, %s, %s, %s) "
                    "RETURNING id",
                    (
                        user_name or "",
                        raw_prompt or "",
                        business_logic or "",
                        languages_requested or "",
                        psycopg2.extras.Json(language_prompts or {}),
                        model_used or "",
                    ),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
            return new_id
        finally:
            conn.close()
    except Exception as e:
        print(f"WARNING: failed to log generation to database: {e}")
        return None


def insert_comment(generation_id: int, commenter_name: str, feedback_text: str) -> bool:
    """Logs one piece of feedback against an existing generation. Returns False
    (rather than raising) if the write failed, e.g. generation_id doesn't
    exist and the foreign key constraint rejects it."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO comments (generation_id, commenter_name, feedback_text) "
                    "VALUES (%s, %s, %s)",
                    (generation_id, commenter_name or "", feedback_text or ""),
                )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:
        print(f"WARNING: failed to log comment to database: {e}")
        return False


def get_generations_since(cutoff_iso: str) -> list:
    """Returns every generation row (as dicts) with timestamp >= cutoff_iso,
    each with its own comments nested in under a 'comments' key — this is the
    shape the daily report builder consumes directly, one row per generation
    with feedback already joined in rather than requiring a second query per
    row from the caller."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM generations WHERE "timestamp" >= %s ORDER BY "timestamp" ASC',
                (cutoff_iso,),
            )
            gen_rows = cur.fetchall()

            results = []
            for row in gen_rows:
                gen = dict(row)
                cur.execute(
                    'SELECT commenter_name, feedback_text, "timestamp" FROM comments '
                    'WHERE generation_id = %s ORDER BY "timestamp" ASC',
                    (gen["id"],),
                )
                gen["comments"] = [dict(c) for c in cur.fetchall()]
                results.append(gen)
            return results
    finally:
        conn.close()
