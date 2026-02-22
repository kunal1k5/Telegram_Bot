from game.database import add_coins, get_coins, register_user


def add(user_id: int, amount: int) -> None:
    register_user(user_id)
    add_coins(user_id, amount)


def balance(user_id: int) -> int:
    register_user(user_id)
    return get_coins(user_id)
