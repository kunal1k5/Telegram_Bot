import asyncio
import random
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from game.database import add_loss, register_user
from game.economy import add
from game.inventory import get_inventory, use_item
from game.leaderboard import add_win
from game.roles import role_label

active_games = {}
player_game_map = {}

MIN_PLAYERS = 5
MAX_PLAYERS = 25
DEFAULT_JOIN_TIME = 60


def generate_roles(player_count: int) -> list[str]:
    roles: list[str] = []

    # Professional balanced model
    if 5 <= player_count <= 6:
        mafia_count = 1
        special_roles = ["doctor"]
    elif 7 <= player_count <= 8:
        mafia_count = 2
        special_roles = ["doctor", "detective"]
    elif 9 <= player_count <= 10:
        mafia_count = 2
        special_roles = ["doctor", "detective", "mayor"]
    elif 11 <= player_count <= 13:
        mafia_count = 3
        special_roles = ["doctor", "detective", "mayor", "witch"]
    elif 14 <= player_count <= 16:
        mafia_count = 3
        special_roles = ["doctor", "detective", "mayor", "witch", "silencer"]
    elif 17 <= player_count <= 20:
        mafia_count = 4
        special_roles = ["doctor", "detective", "mayor", "witch", "silencer", "guardian", "sniper"]
    elif 21 <= player_count <= 25:
        mafia_count = 5
        special_roles = [
            "doctor",
            "detective",
            "mayor",
            "witch",
            "silencer",
            "guardian",
            "sniper",
            "oracle",
            "bomber",
            "judge",
        ]
    else:
        # Fallback for unexpected values
        mafia_count = max(1, player_count // 5)
        special_roles = ["doctor"]

    roles += ["mafia"] * mafia_count
    roles += special_roles

    while len(roles) < player_count:
        roles.append("villager")

    random.shuffle(roles)
    return roles[:player_count]


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
        "sniper_used": set(),
        "judge_used": False,
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
        f"ðŸŽ® Mafia Game Created!\nâ³ You have {game['join_time']} seconds to join!\nUse /join",
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
        await context.bot.send_message(chat_id, "ðŸš€ Auto Starting Game...")
        await start_game(chat_id, context)
    else:
        await context.bot.send_message(chat_id, "âŒ Not enough players. Game cancelled.")
        _cleanup_game(chat_id)


def join_game(chat_id: int, user_id: int) -> tuple[bool, str]:
    register_user(user_id, str(user_id))
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
    if not game:
        return False
    if game.get("started"):
        return False
    if len(game["players"]) < MIN_PLAYERS:
        await context.bot.send_message(chat_id, "âŒ Minimum 5 players required.")
        return False

    join_task = game.get("join_task")
    if join_task:
        join_task.cancel()
        game["join_task"] = None

    players = game["players"].copy()
    random.shuffle(players)
    role_list = generate_roles(len(players))
    roles = {}
    for player, role in zip(players, role_list):
        roles[player] = role

    game["roles"] = roles
    game["alive"] = players.copy()
    game["started"] = True
    game["phase"] = "night"
    game["night_actions"] = {}
    game["silenced"] = None
    game["witch_potions"] = {"heal": 1, "poison": 1}
    game["sniper_used"] = set()
    game["judge_used"] = False

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
                        "ðŸ“© Open Bot",
                        url=f"https://t.me/{bot_username}?start=mafia",
                    )
                ]
            ]
        )
        await context.bot.send_message(
            chat_id,
            "ðŸ” Role DMs are sent now. If DM is blocked, open bot and use /myrole.",
            reply_markup=open_bot_markup,
        )

    for player in players:
        try:
            await context.bot.send_message(
                player,
                f"ðŸŽ­ YOUR ROLE: {role_label(roles[player])}\nKeep it secret!",
            )
        except Exception:
            await context.bot.send_message(
                chat_id,
                f"âš  Player [{player}](tg://user?id={player}) must start bot to see their role.",
                parse_mode="Markdown",
            )

    await dramatic_send(chat_id, "ðŸŒ™ The night falls... shadows move...", context)
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
        "ðŸŒ™ *THE NIGHT HAS FALLEN...*\nSpecial roles, use your powers!",
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
        elif role == "guardian":
            buttons = [
                InlineKeyboardButton(f"Guard {p}", callback_data=f"night_guard_{chat_id}_{p}")
                for p in game["alive"]
            ]
        elif role == "oracle":
            buttons = [
                InlineKeyboardButton(f"Foresee {p}", callback_data=f"night_foresee_{chat_id}_{p}")
                for p in game["alive"]
            ]
        elif role == "sniper" and player not in game.get("sniper_used", set()):
            buttons = [
                InlineKeyboardButton(f"Snipe {p}", callback_data=f"night_snipe_{chat_id}_{p}")
                for p in game["alive"]
                if p != player
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
                    "ðŸŒ™ Use your night ability:",
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
    guard_target = actions.get("guard")
    heal_target = actions.get("heal")
    snipe_target = actions.get("snipe")
    poison_target = actions.get("poison")

    if snipe_target and snipe_target in game["alive"]:
        game["alive"].remove(snipe_target)
        await context.bot.send_message(chat_id, f"🎯 Sniper eliminated {snipe_target}!")

    if poison_target and poison_target in game["alive"]:
        game["alive"].remove(poison_target)
        await context.bot.send_message(chat_id, f"ðŸ§™ Witch poisoned {poison_target}!")

    if kill_target and kill_target in game["alive"]:
        inv = get_inventory(kill_target)
        if inv.get("nightimmunity", 0) > 0 and use_item(kill_target, "nightimmunity"):
            await context.bot.send_message(chat_id, "🛡 Night Immunity activated! Attack blocked.")
        elif inv.get("shield", 0) > 0 and use_item(kill_target, "shield"):
            await context.bot.send_message(chat_id, "🛡 Shield activated! Attack blocked.")
        elif save_target == kill_target or heal_target == kill_target or guard_target == kill_target:
            await context.bot.send_message(chat_id, "💉 Target was saved during the night!")
        else:
            game["alive"].remove(kill_target)
            await context.bot.send_message(
                chat_id,
                f"💀 Player {kill_target} was killed during the night.",
            )
            if game["roles"].get(kill_target) == "bomber":
                mafia_targets = [p for p in game["alive"] if game["roles"].get(p) == "mafia"]
                if mafia_targets:
                    boom_target = random.choice(mafia_targets)
                    game["alive"].remove(boom_target)
                    await context.bot.send_message(chat_id, f"💣 Bomber blast eliminated mafia {boom_target}!")

    if not game["alive"]:
        _cleanup_game(chat_id)
        return

    winner = check_win(chat_id)
    if winner:
        await context.bot.send_message(chat_id, f"ðŸ† {winner}")
        return

    await day_phase(chat_id, context)


async def day_phase(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    game["phase"] = "day"
    game["votes"] = {}
    game["day_voters"] = set()
    game["vote_log"] = {}

    await context.bot.send_message(chat_id, "â˜€ï¸ DAY PHASE STARTED!\nDiscuss and vote.")
    if game.get("silenced"):
        await context.bot.send_message(chat_id, f"ðŸ¤« Player {game['silenced']} is silenced today!")

    buttons = [
        InlineKeyboardButton(str(p), callback_data=f"vote_{chat_id}_{p}")
        for p in game["alive"]
    ]
    keyboard = InlineKeyboardMarkup(
        [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    )
    await context.bot.send_message(chat_id, "ðŸ—³ Vote to eliminate:", reply_markup=keyboard)

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
    vote_log = game.get("vote_log", {})
    mvp_candidates = [uid for uid, target in vote_log.items() if target == eliminated]
    if mvp_candidates:
        mvp_user = random.choice(mvp_candidates)
        add(mvp_user, 30)
        await context.bot.send_message(chat_id, f"🏅 MVP Vote bonus +30 coins to {mvp_user}!")

    judge_alive = next((p for p in game["alive"] if game["roles"].get(p) == "judge"), None)
    if judge_alive and not game.get("judge_used", False):
        game["judge_used"] = True
        await context.bot.send_message(chat_id, f"⚖ Judge {judge_alive} cancelled this vote once!")
        game["silenced"] = None
        await next_round(chat_id, context)
        return

    inv = get_inventory(eliminated)
    if inv.get("shield", 0) > 0 and use_item(eliminated, "shield"):
        await context.bot.send_message(chat_id, "â¤ï¸ Shield saved the player from elimination!")
    else:
        if eliminated in game["alive"]:
            game["alive"].remove(eliminated)
        await context.bot.send_message(chat_id, f"ðŸ’€ Player {eliminated} was eliminated!")

    game["silenced"] = None

    if not game["alive"]:
        _cleanup_game(chat_id)
        return

    winner = check_win(chat_id)
    if winner:
        await context.bot.send_message(chat_id, f"ðŸ† {winner}")
        return

    await next_round(chat_id, context)


async def next_round(chat_id: int, context) -> None:
    await dramatic_send(chat_id, "ðŸŒ™ The next night begins...", context)
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
        for player, role in roles.items():
            if role != "mafia":
                add(player, 50)
                add_win(player)
                if player in alive:
                    add(player, 20)  # Survival bonus
            else:
                add(player, 10)  # Participation
                add_loss(player)
        _cleanup_game(chat_id)
        return "Villagers Win!"

    if len(mafia_alive) >= len(villagers_alive):
        for player, role in roles.items():
            if role == "mafia":
                add(player, 50)
                add_win(player)
                if player in alive:
                    add(player, 20)  # Survival bonus
            else:
                add(player, 10)  # Participation
                add_loss(player)
        _cleanup_game(chat_id)
        return "Mafia Wins!"

    return None

