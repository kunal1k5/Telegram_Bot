import json
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "data" / "user_coins.json"


def load() -> dict:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    if not FILE.exists():
        return {}
    with FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(data: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    with FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add(user_id: int, amount: int) -> None:
    data = load()
    uid = str(user_id)
    data[uid] = data.get(uid, 0) + amount
    save(data)


def balance(user_id: int) -> int:
    return load().get(str(user_id), 0)
