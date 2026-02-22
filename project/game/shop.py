from game.economy import add, balance
from game.inventory import add_item

SHOP_ITEMS = {
    "shield": 500,
    "voteboost": 600,
    "reveal": 200,
    "silencetoken": 400,
    "nightimmunity": 600,
    # Backward compatibility aliases
    "doublevote": 600,
    "extralife": 500,
}

def buy(user_id: int, item: str) -> str:
    key = (item or "").strip().lower().replace(" ", "")
    aliases = {
        "voteboost": "voteboost",
        "boostvote": "voteboost",
        "doublevote": "voteboost",
        "reveal": "reveal",
        "revealscan": "reveal",
        "silencetoken": "silencetoken",
        "silence": "silencetoken",
        "nightimmunity": "nightimmunity",
        "immunity": "nightimmunity",
        "shield": "shield",
        "extralife": "shield",
    }
    key = aliases.get(key, key)

    if key not in SHOP_ITEMS:
        return "Invalid item."

    cost = SHOP_ITEMS[key]
    if balance(user_id) < cost:
        return "Not enough coins."

    add(user_id, -cost)
    add_item(user_id, key)
    return f"{key} purchased successfully!"
