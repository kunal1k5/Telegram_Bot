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

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    language_pref TEXT NOT NULL DEFAULT 'auto',
                    persona_notes TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_connections (
                    user_id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    connected_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notes (
                    group_id INTEGER NOT NULL,
                    note_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_by INTEGER,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (group_id, note_name)
                );

                CREATE TABLE IF NOT EXISTS filters (
                    group_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_by INTEGER,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (group_id, keyword)
                );

                CREATE TABLE IF NOT EXISTS federations (
                    fed_id TEXT PRIMARY KEY,
                    fed_name TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS federation_chats (
                    fed_id TEXT NOT NULL,
                    group_id INTEGER NOT NULL,
                    added_by INTEGER,
                    joined_at REAL NOT NULL,
                    PRIMARY KEY (fed_id, group_id)
                );

                CREATE TABLE IF NOT EXISTS federation_bans (
                    fed_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    reason TEXT,
                    banned_by INTEGER,
                    banned_at REAL NOT NULL,
                    PRIMARY KEY (fed_id, user_id)
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

    def get_all_users(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, username, first_name, join_date, last_seen, message_count
                FROM users
                ORDER BY join_date DESC
                """
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

    def get_usage_snapshot(self) -> dict[str, int]:
        now = time.time()
        one_hour_ago = now - 3600
        day_ago = now - 86400
        with self._lock, self._connect() as conn:
            return {
                "total_users_all_time": conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"],
                "total_groups_all_time": conn.execute("SELECT COUNT(*) AS c FROM groups").fetchone()["c"],
                "users_active_1h": conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE last_seen >= ?", (one_hour_ago,)
                ).fetchone()["c"],
                "users_active_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE last_seen >= ?", (day_ago,)
                ).fetchone()["c"],
                "groups_active_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM groups WHERE last_active >= ?", (day_ago,)
                ).fetchone()["c"],
                "new_users_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE join_date >= ?", (day_ago,)
                ).fetchone()["c"],
                "new_groups_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM groups WHERE added_date >= ?", (day_ago,)
                ).fetchone()["c"],
                "events_24h": conn.execute(
                    "SELECT COUNT(*) AS c FROM activities WHERE ts >= ?", (day_ago,)
                ).fetchone()["c"],
            }

    def set_user_language(self, user_id: int, language_pref: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, language_pref, persona_notes, updated_at)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    language_pref=excluded.language_pref,
                    updated_at=excluded.updated_at
                """,
                (user_id, language_pref, now),
            )

    def get_user_language(self, user_id: int) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT language_pref FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return "auto"
        return row["language_pref"] or "auto"

    def add_chat_memory(self, user_id: int, chat_id: int, role: str, content: str) -> None:
        now = time.time()
        text = (content or "").strip()
        if not text:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_memory (user_id, chat_id, role, content, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, chat_id, role, text[:2000], now),
            )
            # Keep memory bounded per user+chat.
            conn.execute(
                """
                DELETE FROM chat_memory
                WHERE id IN (
                    SELECT id FROM chat_memory
                    WHERE user_id = ? AND chat_id = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET 40
                )
                """,
                (user_id, chat_id),
            )

    def get_chat_memory(self, user_id: int, chat_id: int, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, ts
                FROM chat_memory
                WHERE user_id = ? AND chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, chat_id, limit),
            ).fetchall()
        # return oldest -> newest for model context
        return [dict(row) for row in reversed(rows)]

    # -------------------- Connect System -------------------- #
    def set_connection(self, user_id: int, group_id: int) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_connections (user_id, group_id, connected_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    group_id=excluded.group_id,
                    connected_at=excluded.connected_at
                """,
                (user_id, group_id, now),
            )

    def get_connection(self, user_id: int) -> Optional[int]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT group_id FROM user_connections WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return int(row["group_id"])

    def remove_connection(self, user_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM user_connections WHERE user_id = ?", (user_id,))

    # -------------------- Notes -------------------- #
    def save_note(self, group_id: int, note_name: str, content: str, created_by: int) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notes (group_id, note_name, content, created_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(group_id, note_name) DO UPDATE SET
                    content=excluded.content,
                    created_by=excluded.created_by,
                    updated_at=excluded.updated_at
                """,
                (group_id, note_name.lower().strip(), content.strip()[:4000], created_by, now),
            )

    def get_note(self, group_id: int, note_name: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT note_name, content, created_by, updated_at
                FROM notes
                WHERE group_id = ? AND note_name = ?
                """,
                (group_id, note_name.lower().strip()),
            ).fetchone()
        return dict(row) if row else None

    def list_notes(self, group_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT note_name, updated_at
                FROM notes
                WHERE group_id = ?
                ORDER BY note_name ASC
                LIMIT ?
                """,
                (group_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_note(self, group_id: int, note_name: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM notes WHERE group_id = ? AND note_name = ?",
                (group_id, note_name.lower().strip()),
            )
        return cur.rowcount > 0

    # -------------------- Filters -------------------- #
    def save_filter(self, group_id: int, keyword: str, response: str, created_by: int) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO filters (group_id, keyword, response, created_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(group_id, keyword) DO UPDATE SET
                    response=excluded.response,
                    created_by=excluded.created_by,
                    updated_at=excluded.updated_at
                """,
                (group_id, keyword.lower().strip(), response.strip()[:4000], created_by, now),
            )

    def get_matching_filter(self, group_id: int, message_text: str) -> Optional[dict[str, Any]]:
        query_text = (message_text or "").lower().strip()
        if not query_text:
            return None
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT keyword, response
                FROM filters
                WHERE group_id = ?
                ORDER BY LENGTH(keyword) DESC
                """,
                (group_id,),
            ).fetchall()
        for row in rows:
            kw = (row["keyword"] or "").strip()
            if kw and kw in query_text:
                return dict(row)
        return None

    def list_filters(self, group_id: int, limit: int = 300) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT keyword, updated_at
                FROM filters
                WHERE group_id = ?
                ORDER BY keyword ASC
                LIMIT ?
                """,
                (group_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_filter(self, group_id: int, keyword: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM filters WHERE group_id = ? AND keyword = ?",
                (group_id, keyword.lower().strip()),
            )
        return cur.rowcount > 0

    # -------------------- Federation Scaffold -------------------- #
    def create_federation(self, fed_id: str, fed_name: str, owner_id: int) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO federations (fed_id, fed_name, owner_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (fed_id.strip(), fed_name.strip()[:120], owner_id, now),
            )

    def get_federation(self, fed_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT fed_id, fed_name, owner_id, created_at FROM federations WHERE fed_id = ?",
                (fed_id.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def list_owner_federations(self, owner_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT fed_id, fed_name, created_at
                FROM federations
                WHERE owner_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (owner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def join_federation_chat(self, fed_id: str, group_id: int, added_by: int) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO federation_chats (fed_id, group_id, added_by, joined_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fed_id, group_id) DO UPDATE SET
                    added_by=excluded.added_by,
                    joined_at=excluded.joined_at
                """,
                (fed_id.strip(), group_id, added_by, now),
            )

    def leave_federation_chat(self, group_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM federation_chats WHERE group_id = ?",
                (group_id,),
            )

    def get_group_federation(self, group_id: int) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT fc.fed_id, f.fed_name, f.owner_id
                FROM federation_chats fc
                JOIN federations f ON f.fed_id = fc.fed_id
                WHERE fc.group_id = ?
                """,
                (group_id,),
            ).fetchone()
        return dict(row) if row else None

    def count_federation_chats(self, fed_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM federation_chats WHERE fed_id = ?",
                (fed_id.strip(),),
            ).fetchone()
        return int(row["c"]) if row else 0

    def fed_ban(self, fed_id: str, user_id: int, banned_by: int, reason: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO federation_bans (fed_id, user_id, reason, banned_by, banned_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fed_id, user_id) DO UPDATE SET
                    reason=excluded.reason,
                    banned_by=excluded.banned_by,
                    banned_at=excluded.banned_at
                """,
                (fed_id.strip(), user_id, reason.strip()[:500], banned_by, now),
            )

    def fed_unban(self, fed_id: str, user_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM federation_bans WHERE fed_id = ? AND user_id = ?",
                (fed_id.strip(), user_id),
            )
        return cur.rowcount > 0
