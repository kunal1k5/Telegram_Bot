import asyncio
import random
import time

from telegram import (
    ChatMemberAdministrator,
    ChatMemberOwner,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

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
    await _upsert_game_message(chat_id, text, context)


async def check_bot_permissions(chat_id: int, context) -> tuple[bool, str | None]:
    bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)

    if not isinstance(bot_member, (ChatMemberAdministrator, ChatMemberOwner)):
        return False, "❌ Bot must be admin to start Mafia game."

    if isinstance(bot_member, ChatMemberOwner):
        return True, None

    if not bot_member.can_delete_messages:
        return False, "❌ Enable 'Delete Messages' permission for bot."

    if not bot_member.can_manage_chat:
        return False, "❌ Enable 'Manage Chat' permission for bot."

    if not bot_member.can_restrict_members:
        return False, "❌ Enable 'Restrict Members' permission for bot."

    return True, None


def _track_game_message(game: dict, message_id: int | None) -> None:
    if not game or not message_id:
        return
    messages = game.get("messages")
    if isinstance(messages, set):
        messages.add(message_id)


async def _upsert_game_message(
    chat_id: int,
    text: str,
    context,
    reply_markup: InlineKeyboardMarkup | None = None,
    parse_mode: str | None = None,
) -> int | None:
    game = active_games.get(chat_id)
    if not game:
        sent = await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
        )
        return sent.message_id

    message_id = game.get("main_message_id")
    if message_id:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
            _track_game_message(game, message_id)
            return message_id
        except Exception as e:
            if "message is not modified" in str(e).lower():
                return message_id

    sent = await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
    )
    game["main_message_id"] = sent.message_id
    _track_game_message(game, sent.message_id)
    return sent.message_id


async def clean_game_messages(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    ids = set()
    if game.get("join_message_id"):
        ids.add(game["join_message_id"])
    if game.get("main_message_id"):
        ids.add(game["main_message_id"])
    if isinstance(game.get("messages"), set):
        ids.update(game["messages"])

    for msg_id in ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass


def _unmute_permissions() -> "ChatPermissions":
    from telegram import ChatPermissions

    return ChatPermissions(
        can_send_messages=True,
        can_send_audios=True,
        can_send_documents=True,
        can_send_photos=True,
        can_send_videos=True,
        can_send_video_notes=True,
        can_send_voice_notes=True,
        can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True,
    )


async def _set_member_mute(chat_id: int, user_id: int, context, muted: bool) -> bool:
    from telegram import ChatPermissions

    try:
        perms = ChatPermissions(can_send_messages=False) if muted else _unmute_permissions()
        await context.bot.restrict_chat_member(chat_id=chat_id, user_id=user_id, permissions=perms)
        return True
    except Exception:
        return False


async def _mute_game_players_for_night(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    game["night_mode"] = True
    game["muted_players"] = set()

    # Previous day silenced users are reset when new night starts.
    for user_id in list(game.get("silenced_players", set())):
        await _set_member_mute(chat_id, user_id, context, muted=False)
    game["silenced_players"] = set()
    game["silenced"] = None

    for user_id in game.get("players", []):
        if await _set_member_mute(chat_id, user_id, context, muted=True):
            game["muted_players"].add(user_id)


async def _unmute_night_players_for_day(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    for user_id in list(game.get("muted_players", set())):
        await _set_member_mute(chat_id, user_id, context, muted=False)
    game["muted_players"] = set()
    game["night_mode"] = False


async def _apply_day_silencer(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    pending = set(game.get("pending_silenced", set()))
    game["pending_silenced"] = set()
    game["silenced_players"] = set()

    for user_id in pending:
        if user_id not in game.get("alive", []):
            continue
        if await _set_member_mute(chat_id, user_id, context, muted=True):
            game["silenced_players"].add(user_id)

    if game["silenced_players"]:
        game["silenced"] = next(iter(game["silenced_players"]))
    else:
        game["silenced"] = None


async def cleanup_game(chat_id: int, context, delete_messages: bool = False) -> None:
    game = active_games.get(chat_id)
    if not game:
        return

    for user_id in set(game.get("players", [])) | set(game.get("muted_players", set())) | set(game.get("silenced_players", set())):
        await _set_member_mute(chat_id, user_id, context, muted=False)

    if delete_messages:
        await clean_game_messages(chat_id, context)
    _cleanup_game(chat_id)


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
        "night_mode": False,
        "join_time": join_time,
        "join_deadline": time.monotonic() + join_time,
        "join_task": None,
        "join_message_id": None,
        "main_message_id": None,
        "messages": set(),
        "phase_token": 0,
        "votes": {},
        "night_actions": {},
        "message_count": {},
        "muted_players": set(),
        "silenced_players": set(),
        "pending_silenced": set(),
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
        started = await start_game(chat_id, context)
        if not started:
            await cleanup_game(chat_id, context, delete_messages=True)
    else:
        await context.bot.send_message(chat_id, "âŒ Not enough players. Game cancelled.")
        await cleanup_game(chat_id, context, delete_messages=True)


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

    try:
        ok, error_msg = await check_bot_permissions(chat_id, context)
    except Exception:
        await context.bot.send_message(chat_id, "❌ Could not verify bot admin permissions.")
        return False
    if not ok:
        await context.bot.send_message(chat_id, error_msg or "❌ Missing bot permissions.")
        return False

    join_task = game.get("join_task")
    if join_task:
        join_task.cancel()
        game["join_task"] = None

    join_message_id = game.get("join_message_id")
    if join_message_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=join_message_id)
        except Exception:
            pass
        game["join_message_id"] = None

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
    game["night_mode"] = True
    game["night_actions"] = {}
    game["message_count"] = {}
    game["muted_players"] = set()
    game["silenced_players"] = set()
    game["pending_silenced"] = set()
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
        await _upsert_game_message(
            chat_id,
            "ðŸ” Role DMs are sent now. If DM is blocked, open bot and use /myrole.",
            context,
            reply_markup=open_bot_markup,
        )

    failed_dm: list[int] = []
    for player in players:
        try:
            await context.bot.send_message(
                player,
                f"ðŸŽ­ YOUR ROLE: {role_label(roles[player])}\nKeep it secret!",
            )
        except Exception:
            failed_dm.append(player)

    if failed_dm:
        mention_list = ", ".join(f"[{uid}](tg://user?id={uid})" for uid in failed_dm[:10])
        extra = "" if len(failed_dm) <= 10 else f" ... +{len(failed_dm) - 10} more"
        await _upsert_game_message(
            chat_id,
            f"âš  Players must start bot to see role DM:\n{mention_list}{extra}",
            context,
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
    game["night_mode"] = True
    game["night_actions"] = {}
    game["phase_token"] = int(game.get("phase_token", 0)) + 1
    phase_token = game["phase_token"]

    await _mute_game_players_for_night(chat_id, context)

    await _upsert_game_message(
        chat_id,
        "🌙 *NIGHT PHASE*\nGame players are muted.\nSpecial roles, use your powers in DM.",
        context,
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
    game = active_games.get(chat_id)
    if not game or game.get("phase") != "night" or game.get("phase_token") != phase_token:
        return
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
    updates: list[str] = []

    if snipe_target and snipe_target in game["alive"]:
        game["alive"].remove(snipe_target)
        updates.append(f"🎯 Sniper eliminated {snipe_target}!")

    if poison_target and poison_target in game["alive"]:
        game["alive"].remove(poison_target)
        updates.append(f"🧙 Witch poisoned {poison_target}!")

    if kill_target and kill_target in game["alive"]:
        inv = get_inventory(kill_target)
        if inv.get("nightimmunity", 0) > 0 and use_item(kill_target, "nightimmunity"):
            updates.append("🛡 Night Immunity activated! Attack blocked.")
        elif inv.get("shield", 0) > 0 and use_item(kill_target, "shield"):
            updates.append("🛡 Shield activated! Attack blocked.")
        elif save_target == kill_target or heal_target == kill_target or guard_target == kill_target:
            updates.append("💉 Target was saved during the night!")
        else:
            game["alive"].remove(kill_target)
            updates.append(f"💀 Player {kill_target} was killed during the night.")
            if game["roles"].get(kill_target) == "bomber":
                mafia_targets = [p for p in game["alive"] if game["roles"].get(p) == "mafia"]
                if mafia_targets:
                    boom_target = random.choice(mafia_targets)
                    game["alive"].remove(boom_target)
                    updates.append(f"💣 Bomber blast eliminated mafia {boom_target}!")

    if not updates:
        updates.append("🌙 Quiet night. No one died.")

    if not game["alive"]:
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await cleanup_game(chat_id, context)
        return

    winner = check_win(chat_id)
    if winner:
        updates.append(f"🏆 {winner}")
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await cleanup_game(chat_id, context)
        return

    await _upsert_game_message(chat_id, "\n".join(updates), context)
    await asyncio.sleep(2)
    await day_phase(chat_id, context)


async def day_phase(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return
    game["phase"] = "day"
    game["night_mode"] = False
    game["phase_token"] = int(game.get("phase_token", 0)) + 1
    phase_token = game["phase_token"]
    game["votes"] = {}
    game["day_voters"] = set()
    game["vote_log"] = {}
    game["message_count"] = {}

    await _unmute_night_players_for_day(chat_id, context)
    await _apply_day_silencer(chat_id, context)

    day_text = "☀️ DAY PHASE STARTED!\nDiscuss and vote."
    silenced = sorted(game.get("silenced_players", set()))
    if silenced:
        day_text += f"\n\n🤫 Silenced today: {', '.join(str(uid) for uid in silenced)}"

    buttons = [
        InlineKeyboardButton(str(p), callback_data=f"vote_{chat_id}_{p}")
        for p in game["alive"]
    ]
    keyboard = InlineKeyboardMarkup(
        [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    )
    await _upsert_game_message(
        chat_id,
        f"{day_text}\n\n🗳 Vote to eliminate:",
        context,
        reply_markup=keyboard,
    )

    await asyncio.sleep(30)
    game = active_games.get(chat_id)
    if not game or game.get("phase") != "day" or game.get("phase_token") != phase_token:
        return
    await resolve_votes(chat_id, context)


async def resolve_votes(chat_id: int, context) -> None:
    game = active_games.get(chat_id)
    if not game:
        return

    votes = game["votes"]
    updates: list[str] = []
    if not votes:
        updates.append("No one was voted out.")
        game["silenced"] = None
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await next_round(chat_id, context)
        return

    eliminated = max(votes, key=votes.get)
    vote_log = game.get("vote_log", {})
    mvp_candidates = [uid for uid, target in vote_log.items() if target == eliminated]
    if mvp_candidates:
        mvp_user = random.choice(mvp_candidates)
        add(mvp_user, 30)
        updates.append(f"🏅 MVP Vote bonus +30 coins to {mvp_user}!")

    judge_alive = next((p for p in game["alive"] if game["roles"].get(p) == "judge"), None)
    if judge_alive and not game.get("judge_used", False):
        game["judge_used"] = True
        updates.append(f"⚖ Judge {judge_alive} cancelled this vote once!")
        game["silenced"] = None
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await next_round(chat_id, context)
        return

    inv = get_inventory(eliminated)
    if inv.get("shield", 0) > 0 and use_item(eliminated, "shield"):
        updates.append("❤️ Shield saved the player from elimination!")
    else:
        if eliminated in game["alive"]:
            game["alive"].remove(eliminated)
        updates.append(f"💀 Player {eliminated} was eliminated!")

    game["silenced"] = None

    if not game["alive"]:
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await cleanup_game(chat_id, context)
        return

    winner = check_win(chat_id)
    if winner:
        updates.append(f"🏆 {winner}")
        await _upsert_game_message(chat_id, "\n".join(updates), context)
        await cleanup_game(chat_id, context)
        return

    await _upsert_game_message(chat_id, "\n".join(updates), context)
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
        return "Mafia Wins!"

    return None

