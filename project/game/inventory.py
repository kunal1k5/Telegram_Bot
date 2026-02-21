import json
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "data" / "user_inventory.json"


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


def add_item(user_id: int, item: str) -> None:
    data = load()
    uid = str(user_id)
    if uid not in data:
        data[uid] = {}
    data[uid][item] = data[uid].get(item, 0) + 1
    save(data)


def get_inventory(user_id: int) -> dict:
    return load().get(str(user_id), {})


def use_item(user_id: int, item: str) -> bool:
    data = load()
    uid = str(user_id)
    if uid not in data:
        return False
    if data[uid].get(item, 0) <= 0:
        return False
    data[uid][item] -= 1
    if data[uid][item] <= 0:
        del data[uid][item]
    if not data[uid]:
        del data[uid]
    save(data)
    return True
