from game.database import add_item as db_add_item
from game.database import get_inventory as db_get_inventory
from game.database import register_user, use_item as db_use_item


def add_item(user_id: int, item: str) -> None:
    register_user(user_id)
    db_add_item(user_id, item, 1)


def get_inventory(user_id: int) -> dict:
    register_user(user_id)
    return db_get_inventory(user_id)


def use_item(user_id: int, item: str) -> bool:
    register_user(user_id)
    return db_use_item(user_id, item)
