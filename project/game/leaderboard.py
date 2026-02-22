from game.database import add_win as db_add_win
from game.database import get_all_wins, register_user, top_wins


def load() -> dict:
    return get_all_wins()


def save(data: dict) -> None:
    _ = data


def load_season() -> dict:
    return {"season": 1, "reset_date": "never"}


def save_season(data: dict) -> None:
    _ = data


def reset_season() -> None:
    return


def add_win(user_id: int) -> None:
    register_user(user_id)
    db_add_win(user_id, season_points=10)


def top_players():
    return top_wins(10)


def get_rank(wins: int) -> str:
    if wins <= 5:
        return "Bronze"
    if wins <= 15:
        return "Silver"
    if wins <= 30:
        return "Gold"
    return "Legend"
