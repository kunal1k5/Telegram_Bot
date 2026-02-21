from __future__ import annotations

import json
import os
from pathlib import Path


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "user_memory.json"


def load_memory() -> dict[str, str]:
    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_memory(data: dict[str, str]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_user_nickname(user_id: int, default_name: str) -> str:
    memory = load_memory()
    return memory.get(str(user_id), default_name)


def set_user_nickname(user_id: int, nickname: str) -> None:
    memory = load_memory()
    memory[str(user_id)] = nickname.strip()
    save_memory(memory)

