import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class BotDatabase:
    """Lightweight SQLite storage for users, groups, and activity logs."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    join_date REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS groups (
                    group_id INTEGER PRIMARY KEY,
                    title TEXT,
                    type TEXT,
                    username TEXT,
                    added_date REAL NOT NULL,
                    last_active REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS group_members (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    joined REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (group_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    user_id INTEGER,
                    group_id INTEGER,
                    action TEXT NOT NULL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS warns (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    warn_count INTEGER NOT NULL DEFAULT 0,
                    last_reason TEXT,
                    last_warned_at REAL,
                    last_warned_by INTEGER,
                    PRIMARY KEY (group_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS warn_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    warned_by INTEGER,
                    reason TEXT,
                    ts REAL NOT NULL
                );
                """
            )

    def upsert_user(self, user_id: int, username: Optional[str], first_name: Optional[str]) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, username, first_name, join_date, last_seen, message_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    last_seen=excluded.last_seen,
                    message_count=users.message_count + 1
                """,
                (user_id, username, first_name, now, now),
            )

    def remove_user(self, user_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def upsert_group(self, group_id: int, title: Optional[str], group_type: Optional[str], username: Optional[str]) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO groups (group_id, title, type, username, added_date, last_active)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    title=excluded.title,
                    type=excluded.type,
                    username=excluded.username,
                    last_active=excluded.last_active
                """,
                (group_id, title, group_type, username, now, now),
            )

    def remove_group(self, group_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
            conn.execute("DELETE FROM group_members WHERE group_id = ?", (group_id,))

    def upsert_group_member(
        self,
        group_id: int,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO group_members (group_id, user_id, username, first_name, joined, last_seen, message_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    last_seen=excluded.last_seen,
                    message_count=group_members.message_count + 1
                """,
                (group_id, user_id, username, first_name, now, now),
            )

    def log_activity(
        self,
        action: str,
        user_id: Optional[int] = None,
        group_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO activities (ts, user_id, group_id, action, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), user_id, group_id, action, json.dumps(metadata or {})),
            )

    def get_recipient_ids(self) -> tuple[list[int], list[int]]:
        with self._lock, self._connect() as conn:
            users = [row["user_id"] for row in conn.execute("SELECT user_id FROM users")]
            groups = [row["group_id"] for row in conn.execute("SELECT group_id FROM groups")]
        return users, groups

    def get_overview(self) -> dict[str, int]:
        day_ago = time.time() - 86400
        with self._lock, self._connect() as conn:
            return {
                "users": conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"],
                "groups": conn.execute("SELECT COUNT(*) AS c FROM groups").fetchone()["c"],
                "active_users_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE last_seen >= ?", (day_ago,)
                ).fetchone()["c"],
                "active_groups_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM groups WHERE last_active >= ?", (day_ago,)
                ).fetchone()["c"],
                "activities_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM activities WHERE ts >= ?", (day_ago,)
                ).fetchone()["c"],
            }

    def get_recent_activities(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, user_id, group_id, action, metadata
                FROM activities
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_warn(self, group_id: int, user_id: int, warned_by: int, reason: str) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO warns (group_id, user_id, warn_count, last_reason, last_warned_at, last_warned_by)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    warn_count=warns.warn_count + 1,
                    last_reason=excluded.last_reason,
                    last_warned_at=excluded.last_warned_at,
                    last_warned_by=excluded.last_warned_by
                """,
                (group_id, user_id, reason, now, warned_by),
            )
            conn.execute(
                """
                INSERT INTO warn_logs (group_id, user_id, warned_by, reason, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_id, user_id, warned_by, reason, now),
            )
            row = conn.execute(
                "SELECT warn_count FROM warns WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            ).fetchone()
        return int(row["warn_count"]) if row else 0

    def get_warn(self, group_id: int, user_id: int) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT warn_count, last_reason, last_warned_at, last_warned_by
                FROM warns
                WHERE group_id = ? AND user_id = ?
                """,
                (group_id, user_id),
            ).fetchone()
        if not row:
            return {
                "warn_count": 0,
                "last_reason": None,
                "last_warned_at": None,
                "last_warned_by": None,
            }
        return dict(row)

    def reset_warn(self, group_id: int, user_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM warns WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )

    def get_top_warned(self, group_id: int, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, warn_count, last_reason, last_warned_at
                FROM warns
                WHERE group_id = ?
                ORDER BY warn_count DESC, last_warned_at DESC
                LIMIT ?
                """,
                (group_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
