import json
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FILE = DATA_DIR / "leaderboard.json"
SEASON_FILE = DATA_DIR / "season.json"


def _default_season() -> dict:
    return {
        "season": 1,
        "reset_date": "2026-12-31",
    }


def _load_json(path: Path, default):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load() -> dict:
    _maybe_reset_season()
    return _load_json(FILE, {})


def save(data: dict) -> None:
    _save_json(FILE, data)


def load_season() -> dict:
    return _load_json(SEASON_FILE, _default_season())


def save_season(data: dict) -> None:
    _save_json(SEASON_FILE, data)


def reset_season() -> None:
    save({})


def _maybe_reset_season() -> None:
    season = load_season()
    try:
        reset_dt = date.fromisoformat(season["reset_date"])
    except Exception:
        reset_dt = date(2026, 12, 31)
        season["reset_date"] = "2026-12-31"
        save_season(season)

    if date.today() > reset_dt:
        reset_season()
        season["season"] = int(season.get("season", 1)) + 1
        season["reset_date"] = f"{date.today().year}-12-31"
        save_season(season)


def add_win(user_id: int) -> None:
    data = load()
    uid = str(user_id)
    data[uid] = data.get(uid, 0) + 1
    save(data)


def top_players():
    data = load()
    return sorted(data.items(), key=lambda x: x[1], reverse=True)[:10]


def get_rank(wins: int) -> str:
    if wins <= 5:
        return "Bronze"
    if wins <= 15:
        return "Silver"
    if wins <= 30:
        return "Gold"
    return "Legend"
