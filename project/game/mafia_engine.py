import asyncio
import random
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from game.economy import add
from game.inventory import get_inventory, use_item
from game.leaderboard import add_win

active_games = {}
player_game_map = {}

MIN_PLAYERS = 5
MAX_PLAYERS = 25
DEFAULT_JOIN_TIME = 60


async def dramatic_send(chat_id: int, text: str, context) -> None:
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    await asyncio.sleep(2)
    await context.bot.send_message(chat_id, text)


def _cleanup_game(chat_id: int) -> None:
    game = active_games.get(chat_id)
    if not game:
        return

    join_task = game.get("join_task")
    if join_task:
        join_task.cancel()

    for player_id in game.get("players", []):
        player_game_map.pop(player_id, None)
    active_games.pop(chat_id, None)


def cancel_game(chat_id: int) -> bool:
    if chat_id not in active_games:
        return False
    _cleanup_game(chat_id)
    return True


def create_game(chat_id: int, join_time: int = DEFAULT_JOIN_TIME) -> None:
    old_game = active_games.get(chat_id)
    if old_game and old_game.get("join_task"):
        old_game["join_task"].cancel()

    join_time = max(10, min(600, int(join_time)))
    active_games[chat_id] = {
        "players": [],
        "roles": {},
        "alive": [],
        "started": False,
        "phase": "joining",
        "join_time": join_time,
        "join_deadline": time.monotonic() + join_time,
        "join_task": None,
        "votes": {},
        "night_actions": {},
        "day_voters": set(),
        "silenced": None,
        "witch_potions": {"heal": 1, "poison": 1},
    }


def extend_join_time(chat_id: int, extra_seconds: int = 30):
    game = active_games.get(chat_id)
    if not game or game.get("phase") != "joining":
        return None

    game["join_deadline"] += extra_seconds
    remaining = max(0, int(game["join_deadline"] - time.monotonic()))
    game["join_time"] = remaining
    return remaining


async def start_join_timer(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return

    await context.bot.send_message(
        chat_id,
        f"🎮 Mafia Game Created!\n⏳ You have {game['join_time']} seconds to join!\nUse /join",
    )

    try:
        while True:
            game = active_games.get(chat_id)
            if not game:
                return
            if game.get("phase") != "joining" or game.get("started"):
                return

            remaining = game["join_deadline"] - time.monotonic()
            if remaining <= 0:
                break

            game["join_time"] = int(remaining)
            await asyncio.sleep(min(1.0, remaining))
    except asyncio.CancelledError:
        return
    finally:
        game = active_games.get(chat_id)
        if game:
            game["join_task"] = None

    game = active_games.get(chat_id)
    if not game or game.get("phase") != "joining":
        return

    if len(game["players"]) >= MIN_PLAYERS:
        await context.bot.send_message(chat_id, "🚀 Auto Starting Game...")
        await start_game(chat_id, context)
    else:
        await context.bot.send_message(chat_id, "❌ Not enough players. Game cancelled.")
        _cleanup_game(chat_id)


def join_game(chat_id: int, user_id: int) -> tuple[bool, str]:
    game = active_games.get(chat_id)
    if not game:
        return False, "No active game. Use /mafia first."
    if game.get("phase") != "joining":
        return False, "Join phase is over."
    if len(game["players"]) >= MAX_PLAYERS:
        return False, "Game full. Max 25 players."
    if user_id in game["players"]:
        return False, "You already joined."

    game["players"].append(user_id)
    player_game_map[user_id] = chat_id
    return True, f"Joined! Players: {len(game['players'])}/{MAX_PLAYERS}"


async def start_game(chat_id: int, context) -> bool:
    game = active_games.get(chat_id)
    if not game or len(game["players"]) < MIN_PLAYERS or game.get("started"):
        return False

    join_task = game.get("join_task")
    if join_task:
        join_task.cancel()
        game["join_task"] = None

    players = game["players"].copy()
    random.shuffle(players)
    mafia_count = max(2, len(players) // 5)

    roles = {}
    for i, player in enumerate(players):
        roles[player] = "mafia" if i < mafia_count else "villager"

    non_mafia = [p for p in players if roles[p] == "villager"]
    random.shuffle(non_mafia)
    for role_name in ["doctor", "detective", "witch", "silencer", "mayor"]:
        if non_mafia:
            roles[non_mafia.pop()] = role_name

    game["roles"] = roles
    game["alive"] = players.copy()
    game["started"] = True
    game["phase"] = "night"
    game["night_actions"] = {}
    game["silenced"] = None
    game["witch_potions"] = {"heal": 1, "poison": 1}

    bot_username = context.bot.username
    if not bot_username:
        try:
            me = await context.bot.get_me()
            bot_username = me.username
        except Exception:
            bot_username = None

    if bot_username:
        open_bot_markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "📩 Open Bot",
                        url=f"https://t.me/{bot_username}?start=mafia",
                    )
                ]
            ]
        )
        await context.bot.send_message(
            chat_id,
            "🔐 Role DMs are sent now. If DM is blocked, open bot and use /myrole.",
            reply_markup=open_bot_markup,
        )

    for player in players:
        try:
            await context.bot.send_message(
                player,
                f"🎭 YOUR ROLE: {roles[player].upper()}\nKeep it secret!",
            )
        except Exception:
            await context.bot.send_message(
                chat_id,
                f"⚠ Player [{player}](tg://user?id={player}) must start bot to see their role.",
                parse_mode="Markdown",
            )

    await dramatic_send(chat_id, "🌙 The night falls... shadows move...", context)
    await night_phase(chat_id, context)
    return True


async def night_phase(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    game["phase"] = "night"
    game["night_actions"] = {}

    await context.bot.send_message(
        chat_id,
        "🌙 *THE NIGHT HAS FALLEN...*\nSpecial roles, use your powers!",
        parse_mode="Markdown",
    )

    for player in game["alive"]:
        role = game["roles"][player]
        buttons = []

        if role == "mafia":
            buttons = [
                InlineKeyboardButton(f"Kill {p}", callback_data=f"night_kill_{chat_id}_{p}")
                for p in game["alive"]
                if p != player
            ]
        elif role == "doctor":
            buttons = [
                InlineKeyboardButton(f"Save {p}", callback_data=f"night_save_{chat_id}_{p}")
                for p in game["alive"]
            ]
        elif role == "detective":
            buttons = [
                InlineKeyboardButton(f"Check {p}", callback_data=f"night_check_{chat_id}_{p}")
                for p in game["alive"]
            ]
        elif role == "witch":
            heal_buttons = [
                InlineKeyboardButton(f"Heal {p}", callback_data=f"night_heal_{chat_id}_{p}")
                for p in game["alive"]
            ]
            poison_buttons = [
                InlineKeyboardButton(
                    f"Poison {p}", callback_data=f"night_poison_{chat_id}_{p}"
                )
                for p in game["alive"]
            ]
            buttons = heal_buttons + poison_buttons
        elif role == "silencer":
            buttons = [
                InlineKeyboardButton(
                    f"Silence {p}", callback_data=f"night_silence_{chat_id}_{p}"
                )
                for p in game["alive"]
                if p != player
            ]

        if buttons:
            keyboard = InlineKeyboardMarkup(
                [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
            )
            try:
                await context.bot.send_message(
                    player,
                    "🌙 Use your night ability:",
                    reply_markup=keyboard,
                )
            except Exception:
                pass

    await asyncio.sleep(25)
    await resolve_night(chat_id, context)


async def resolve_night(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    actions = game.get("night_actions", {})

    kill_target = actions.get("kill")
    save_target = actions.get("save")
    heal_target = actions.get("heal")
    poison_target = actions.get("poison")

    if poison_target and poison_target in game["alive"]:
        game["alive"].remove(poison_target)
        await context.bot.send_message(chat_id, f"🧙 Witch poisoned {poison_target}!")

    if kill_target and kill_target in game["alive"]:
        inv = get_inventory(kill_target)
        if inv.get("shield", 0) > 0 and use_item(kill_target, "shield"):
            await context.bot.send_message(chat_id, "🛡 Shield activated! Attack blocked.")
        elif save_target == kill_target or heal_target == kill_target:
            await context.bot.send_message(chat_id, "💉 Target was saved during the night!")
        else:
            game["alive"].remove(kill_target)
            await context.bot.send_message(
                chat_id,
                f"💀 Player {kill_target} was killed during the night.",
            )

    if not game["alive"]:
        _cleanup_game(chat_id)
        return

    winner = check_win(chat_id)
    if winner:
        await context.bot.send_message(chat_id, f"🏆 {winner}")
        return

    await day_phase(chat_id, context)


async def day_phase(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    game["phase"] = "day"
    game["votes"] = {}
    game["day_voters"] = set()

    await context.bot.send_message(chat_id, "☀️ DAY PHASE STARTED!\nDiscuss and vote.")
    if game.get("silenced"):
        await context.bot.send_message(chat_id, f"🤫 Player {game['silenced']} is silenced today!")

    buttons = [
        InlineKeyboardButton(str(p), callback_data=f"vote_{chat_id}_{p}")
        for p in game["alive"]
    ]
    keyboard = InlineKeyboardMarkup(
        [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    )
    await context.bot.send_message(chat_id, "🗳 Vote to eliminate:", reply_markup=keyboard)

    await asyncio.sleep(30)
    await resolve_votes(chat_id, context)


async def resolve_votes(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return

    votes = game["votes"]
    if not votes:
        await context.bot.send_message(chat_id, "No one was voted out.")
        game["silenced"] = None
        await next_round(chat_id, context)
        return

    eliminated = max(votes, key=votes.get)
    inv = get_inventory(eliminated)
    if inv.get("extralife", 0) > 0 and use_item(eliminated, "extralife"):
        await context.bot.send_message(chat_id, "❤️ Extra Life used! Player revived!")
    else:
        if eliminated in game["alive"]:
            game["alive"].remove(eliminated)
        await context.bot.send_message(chat_id, f"💀 Player {eliminated} was eliminated!")

    game["silenced"] = None

    if not game["alive"]:
        _cleanup_game(chat_id)
        return

    winner = check_win(chat_id)
    if winner:
        await context.bot.send_message(chat_id, f"🏆 {winner}")
        return

    await next_round(chat_id, context)


async def next_round(chat_id: int, context) -> None:
    await dramatic_send(chat_id, "🌙 The next night begins...", context)
    await night_phase(chat_id, context)


def check_win(chat_id: int):
    game = active_games.get(chat_id)
    if not game:
        return None

    roles = game["roles"]
    alive = game["alive"]
    mafia_alive = [p for p in alive if roles[p] == "mafia"]
    villagers_alive = [p for p in alive if roles[p] != "mafia"]

    if not mafia_alive:
        for p in villagers_alive:
            add(p, 20)
            add_win(p)
        _cleanup_game(chat_id)
        return "Villagers Win!"

    if len(mafia_alive) >= len(villagers_alive):
        for p in mafia_alive:
            add(p, 25)
            add_win(p)
        _cleanup_game(chat_id)
        return "Mafia Wins!"

    return None
