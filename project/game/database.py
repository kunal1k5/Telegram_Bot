import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "mafia_database.db"

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_cursor = _conn.cursor()
_lock = threading.RLock()


def init_db() -> None:
    with _lock:
        _cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                coins INTEGER DEFAULT 100,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                games_played INTEGER DEFAULT 0,
                season_points INTEGER DEFAULT 0
            )
            """
        )

        _cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                user_id INTEGER,
                item TEXT,
                quantity INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, item)
            )
            """
        )
        _conn.commit()


def register_user(user_id: int, username: str | None = None) -> None:
    if not user_id:
        return
    with _lock:
        _cursor.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        row = _cursor.fetchone()
        if not row:
            _cursor.execute(
                "INSERT INTO users (user_id, username) VALUES (?, ?)",
                (user_id, username or ""),
            )
        elif username:
            _cursor.execute("UPDATE users SET username=? WHERE user_id=?", (username, user_id))
        _conn.commit()


def get_user_profile(user_id: int):
    with _lock:
        _cursor.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        return _cursor.fetchone()


def add_coins(user_id: int, amount: int) -> None:
    register_user(user_id)
    with _lock:
        _cursor.execute(
            "UPDATE users SET coins = coins + ? WHERE user_id=?",
            (amount, user_id),
        )
        _conn.commit()


def get_coins(user_id: int) -> int:
    register_user(user_id)
    with _lock:
        _cursor.execute("SELECT coins FROM users WHERE user_id=?", (user_id,))
        row = _cursor.fetchone()
        return int(row["coins"]) if row else 0


def add_item(user_id: int, item: str, qty: int = 1) -> None:
    register_user(user_id)
    with _lock:
        _cursor.execute(
            """
            INSERT INTO inventory (user_id, item, quantity)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, item)
            DO UPDATE SET quantity = quantity + ?
            """,
            (user_id, item, qty, qty),
        )
        _conn.commit()


def get_inventory(user_id: int) -> dict:
    register_user(user_id)
    with _lock:
        _cursor.execute("SELECT item, quantity FROM inventory WHERE user_id=?", (user_id,))
        rows = _cursor.fetchall()
        return {row["item"]: int(row["quantity"]) for row in rows}


def use_item(user_id: int, item: str) -> bool:
    register_user(user_id)
    with _lock:
        _cursor.execute(
            "SELECT quantity FROM inventory WHERE user_id=? AND item=?",
            (user_id, item),
        )
        row = _cursor.fetchone()
        if not row or int(row["quantity"]) <= 0:
            return False

        new_qty = int(row["quantity"]) - 1
        if new_qty <= 0:
            _cursor.execute("DELETE FROM inventory WHERE user_id=? AND item=?", (user_id, item))
        else:
            _cursor.execute(
                "UPDATE inventory SET quantity=? WHERE user_id=? AND item=?",
                (new_qty, user_id, item),
            )
        _conn.commit()
        return True


def add_win(user_id: int, season_points: int = 10) -> None:
    register_user(user_id)
    with _lock:
        _cursor.execute(
            """
            UPDATE users
            SET wins = wins + 1,
                games_played = games_played + 1,
                season_points = season_points + ?
            WHERE user_id=?
            """,
            (season_points, user_id),
        )
        _conn.commit()


def add_loss(user_id: int) -> None:
    register_user(user_id)
    with _lock:
        _cursor.execute(
            """
            UPDATE users
            SET losses = losses + 1,
                games_played = games_played + 1
            WHERE user_id=?
            """,
            (user_id,),
        )
        _conn.commit()


def get_all_wins() -> dict:
    with _lock:
        _cursor.execute("SELECT user_id, wins FROM users")
        rows = _cursor.fetchall()
        return {str(row["user_id"]): int(row["wins"]) for row in rows}


def top_wins(limit: int = 10):
    with _lock:
        _cursor.execute(
            "SELECT user_id, wins FROM users ORDER BY wins DESC, user_id ASC LIMIT ?",
            (limit,),
        )
        rows = _cursor.fetchall()
        return [(str(row["user_id"]), int(row["wins"])) for row in rows]


init_db()
