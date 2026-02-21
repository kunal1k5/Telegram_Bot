from game.economy import add, balance
from game.inventory import add_item

SHOP_ITEMS = {
    "shield": 30,
    "doublevote": 40,
    "extralife": 50,
    "reveal": 120,
}

def buy(user_id: int, item: str) -> str:
    if item not in SHOP_ITEMS:
        return "Invalid item."

    cost = SHOP_ITEMS[item]
    if balance(user_id) < cost:
        return "Not enough coins."

    add(user_id, -cost)
    add_item(user_id, item)
    return f"{item} purchased successfully!"
