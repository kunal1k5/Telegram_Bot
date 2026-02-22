import asyncio
from datetime import datetime
import os
import requests
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ChatMemberStatus, ChatType
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from game.economy import balance
from game.inventory import get_inventory, use_item
from game.leaderboard import get_rank, load as load_leaderboard, top_players
from game.mafia_engine import (
    MIN_PLAYERS,
    active_games,
    cancel_game,
    create_game,
    extend_join_time,
    join_game,
    start_game,
    start_join_timer,
)
from game.shop import buy
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from start_card import create_start_card
except Exception:
    create_start_card = None

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BOT_USERNAME = os.getenv("BOT_USERNAME", "YOUR_BOT_USERNAME").strip("@")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "@AnimxClan_Channel").strip()
CONTACT_USERNAME = (os.getenv("CONTACT_USERNAME") or os.getenv("CONTACT_ID") or "").strip()
PROMOTION_USERNAME = (os.getenv("PROMOTION_USERNAME") or os.getenv("PROMOTION_ID") or "").strip()
CONTACT_PROMOTION_IDS = (os.getenv("CONTACT_PROMOTION_IDS") or os.getenv("CONTACT_AND_PROMOTION") or "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GREETING_TIMEZONE = os.getenv("GREETING_TIMEZONE", "").strip()
START_PANEL_PHOTO_FILE_ID = os.getenv("START_PANEL_PHOTO_FILE_ID", "").strip()
START_PANEL_PHOTO_URL = os.getenv("START_PANEL_PHOTO_URL", "").strip()
START_BANNER_PATH = os.getenv("START_BANNER_PATH", "banner.jpg").strip()


def _telegram_url(raw: str) -> Optional[str]:
    val = (raw or "").strip()
    if not val:
        return None
    upper = val.upper()
    if upper in {"YOUR_CHANNEL", "YOUR_CONTACT", "YOUR_PROMOTION"}:
        return None
    if val.startswith("https://t.me/") or val.startswith("http://t.me/"):
        return val
    if val.startswith("@"):
        return f"https://t.me/{val[1:]}"
    return f"https://t.me/{val}"


def _resolve_contact_promo() -> tuple[str, str]:
    contact = CONTACT_USERNAME
    promo = PROMOTION_USERNAME
    if CONTACT_PROMOTION_IDS:
        parts = [p.strip() for p in CONTACT_PROMOTION_IDS.split(",") if p.strip()]
        if parts:
            contact = parts[0]
        if len(parts) > 1:
            promo = parts[1]
        return contact, promo

    # Backward compatibility: allow two handles inside CONTACT_USERNAME.
    raw_contact = (CONTACT_USERNAME or "").replace(",", " ").strip()
    packed = [p for p in raw_contact.split() if p]
    if len(packed) > 1:
        contact = packed[0]
        if not promo:
            promo = packed[1]
    return contact, promo


def _current_hour() -> int:
    if GREETING_TIMEZONE and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(GREETING_TIMEZONE)).hour
        except Exception:
            pass
    return datetime.now().hour


def _dynamic_greeting() -> str:
    hour = _current_hour()
    if 5 <= hour < 12:
        return "Good Morning"
    if 12 <= hour < 17:
        return "Good Afternoon"
    if 17 <= hour < 22:
        return "Good Evening"
    return "Good Night"


def start_panel_text(user_name: str) -> str:
    greeting = _dynamic_greeting()
    return (
        f"HEY BABY {user_name} NICE TO MEET YOU\n\n"
        "THIS IS ANIMX GAME\n\n"
        "A premium designed game + chat bot for Telegram groups & channels.\n\n"
        "------------------------\n"
        "Multiplayer Mafia Battles\n"
        "Fast - Smart - Always Active\n"
        "Chat Naturally Like a Friend\n"
        "------------------------\n\n"
        f"{greeting} 💖"
    )


def build_start_keyboard(bot_username: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Chat With Bot", url=f"https://t.me/{bot_username}")],
        [InlineKeyboardButton("Add Bot To Group", url=f"https://t.me/{bot_username}?startgroup=true")],
        [
            InlineKeyboardButton("Help", callback_data="help"),
            InlineKeyboardButton("Chat Guide", callback_data="chatguide"),
        ],
    ]

    channel_url = _telegram_url(CHANNEL_USERNAME)
    if channel_url:
        rows.append([InlineKeyboardButton("Channel", url=channel_url)])

    rows.append([InlineKeyboardButton("Mafia Game Hub", callback_data="mafia_hub")])

    contact_handle, promo_handle = _resolve_contact_promo()
    contact_url = _telegram_url(contact_handle)
    promo_url = _telegram_url(promo_handle)
    contact_row = []
    if contact_url:
        contact_row.append(InlineKeyboardButton("Contact", url=contact_url))
    if promo_url and promo_url != contact_url:
        contact_row.append(InlineKeyboardButton("Promotion", url=promo_url))
    if contact_row:
        rows.append(contact_row)

    return InlineKeyboardMarkup(rows)


SYSTEM_PROMPT = (
    "You are Baby, a warm and playful Hinglish chat companion. "
    "Reply naturally in short helpful style, like a close friend."
)


def _fallback_chat_reply(user_text: str) -> str:
    text = (user_text or "").lower().strip()
    if any(word in text for word in {"hi", "hello", "hey", "gm", "good morning"}):
        return "Hii baby, main yahin hoon. Aaj kya baat karein? 💖"
    if any(word in text for word in {"sad", "upset", "depressed", "dukhi"}):
        return "Main tumhare saath hoon. Bolo kya hua, step by step solve karte hain. 🤍"
    if any(word in text for word in {"help", "commands", "command"}):
        return "Main chat, mafia game aur utility commands me help kar sakti hoon. /help try karo."
    return "Samjha. Isko aur clear karke bolo, main best possible help dungi. ✨"


def _call_openrouter(user_text: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.8,
                "max_tokens": 220,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip() or None
    except Exception:
        return None


def _call_gemini(user_text: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nUser: {user_text}"}]}],
                "generationConfig": {"temperature": 0.8, "maxOutputTokens": 220},
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        return (parts[0].get("text") or "").strip() or None
    except Exception:
        return None


async def generate_chat_reply(user_text: str) -> str:
    reply = await asyncio.to_thread(_call_openrouter, user_text)
    if reply:
        return reply
    reply = await asyncio.to_thread(_call_gemini, user_text)
    if reply:
        return reply
    return _fallback_chat_reply(user_text)


def build_mafia_lobby_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Join Game", callback_data="mafia_join")],
            [InlineKeyboardButton("Start Now", callback_data="mafia_force_start")],
            [InlineKeyboardButton("Extend Join Time", callback_data="mafia_extend")],
            [InlineKeyboardButton("Cancel Game", callback_data="mafia_cancel")],
            [
                InlineKeyboardButton("Shop", callback_data="mafia_shop"),
                InlineKeyboardButton("My Profile", callback_data="mafia_profile"),
            ],
            [InlineKeyboardButton("Back", callback_data="mafia_hub")],
        ]
    )


def build_mafia_shop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Buy Shield", callback_data="buy_shield")],
            [InlineKeyboardButton("Buy Double Vote", callback_data="buy_doublevote")],
            [InlineKeyboardButton("Buy Extra Life", callback_data="buy_extralife")],
            [InlineKeyboardButton("Back", callback_data="mafia_hub")],
        ]
    )


def build_profile_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Open Shop", callback_data="mafia_shop")],
            [InlineKeyboardButton("Leaderboard", callback_data="mafia_leaderboard")],
            [InlineKeyboardButton("Back", callback_data="mafia_hub")],
        ]
    )


def _lobby_text(chat_id: int) -> str:
    game = active_games.get(chat_id)
    joined = len(game["players"]) if game else 0
    return f"MAFIA GAME LOBBY\n\nPlayers Joined: {joined} / 25\n\nWaiting for players..."


def _launch_join_lobby(chat_id: int, join_time: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    create_game(chat_id, join_time=join_time)
    task = asyncio.create_task(start_join_timer(chat_id, context))
    active_games[chat_id]["join_task"] = task


async def _is_admin_chat(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    member = await context.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}


def _wins_rank_text(user_id: int) -> tuple[int, str, str]:
    all_wins = load_leaderboard()
    wins = int(all_wins.get(str(user_id), 0))
    sorted_rows = sorted(all_wins.items(), key=lambda x: x[1], reverse=True)
    pos = next((i for i, (uid, _) in enumerate(sorted_rows, 1) if uid == str(user_id)), None)
    pos_text = f"#{pos}" if pos else "Unranked"
    return wins, get_rank(wins), pos_text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_username = context.bot.username or BOT_USERNAME
    user = update.effective_user.first_name or "Friend"
    text = start_panel_text(user)
    keyboard = build_start_keyboard(bot_username)

    # Send premium start image first (if available), then send panel text + buttons.
    try:
        sent_photo = False

        if START_PANEL_PHOTO_FILE_ID:
            await update.message.reply_photo(photo=START_PANEL_PHOTO_FILE_ID)
            sent_photo = True
        elif START_PANEL_PHOTO_URL:
            await update.message.reply_photo(photo=START_PANEL_PHOTO_URL)
            sent_photo = True

        if not sent_photo and create_start_card:
            profile_url = ""
            photos = await context.bot.get_user_profile_photos(update.effective_user.id, limit=1)
            if photos.photos:
                file_id = photos.photos[0][-1].file_id
                pf = await context.bot.get_file(file_id)
                profile_url = pf.file_path or ""

            banner_path = Path(START_BANNER_PATH)
            if not banner_path.is_absolute():
                banner_path = ROOT_DIR / banner_path

            # start_card has built-in gradient fallback even if banner file is missing.
            card = create_start_card(str(banner_path), user, profile_url)
            await update.message.reply_photo(photo=card)
            sent_photo = True

        if not sent_photo:
            banner_path = Path(START_BANNER_PATH)
            if not banner_path.is_absolute():
                banner_path = ROOT_DIR / banner_path
            if banner_path.exists():
                with banner_path.open("rb") as f:
                    await update.message.reply_photo(photo=f)
    except Exception:
        pass

    await update.message.reply_text(text, reply_markup=keyboard)


async def mafia_hub(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "MAFIA GAME HUB\n\n"
        "Social strategy battle for your group.\n\n"
        "Deception. Investigation. Survival.\n\n"
        "Choose what you want to explore:"
    )
    keyboard = [
        [InlineKeyboardButton("Roles & Powers", callback_data="mafia_roles")],
        [InlineKeyboardButton("Shop", callback_data="mafia_shop")],
        [InlineKeyboardButton("My Profile", callback_data="mafia_profile")],
        [InlineKeyboardButton("How To Play", callback_data="mafia_guide")],
        [InlineKeyboardButton("Start Game (Group)", callback_data="mafia_start_group")],
        [InlineKeyboardButton("Back", callback_data="back_start")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def mafia_roles(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Mafia", callback_data="role_mafia")],
        [InlineKeyboardButton("Doctor", callback_data="role_doctor")],
        [InlineKeyboardButton("Detective", callback_data="role_detective")],
        [InlineKeyboardButton("Witch", callback_data="role_witch")],
        [InlineKeyboardButton("Silencer", callback_data="role_silencer")],
        [InlineKeyboardButton("Mayor", callback_data="role_mayor")],
        [InlineKeyboardButton("Back", callback_data="mafia_hub")],
    ]
    await update.callback_query.edit_message_text(
        "Select a role to see its power:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def role_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    role = update.callback_query.data.split("_")[1]
    role_texts = {
        "mafia": "Mafia: Kill one player each night. Goal: Outnumber town.",
        "doctor": "Doctor: Save one player each night.",
        "detective": "Detective: Reveal role of one player.",
        "witch": "Witch: 1 Heal potion + 1 Poison potion.",
        "silencer": "Silencer: Mute one player next day.",
        "mayor": "Mayor: Your vote counts as 2 permanently.",
    }
    keyboard = [
        [InlineKeyboardButton("Open Shop", callback_data="mafia_shop")],
        [InlineKeyboardButton("My Profile", callback_data="mafia_profile")],
        [InlineKeyboardButton("Back", callback_data="mafia_roles")],
    ]
    await update.callback_query.edit_message_text(
        role_texts.get(role, "Role not found."),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def mafia_shop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    coins = balance(user_id)
    text = (
        "MAFIA SHOP\n\n"
        f"Coins: {coins}\n\n"
        "Shield - 30 coins\n"
        "Double Vote - 40 coins\n"
        "Extra Life - 50 coins"
    )
    await update.callback_query.edit_message_text(text, reply_markup=build_mafia_shop_keyboard())


async def mafia_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    coins = balance(user_id)
    inv = get_inventory(user_id)
    wins, tier, pos_text = _wins_rank_text(user_id)
    text = (
        "PLAYER PROFILE\n\n"
        f"Coins: {coins}\n"
        f"Rank: {pos_text} ({tier})\n"
        f"Wins: {wins}\n\n"
        "Inventory:\n"
        f"Shield: {inv.get('shield', 0)}\n"
        f"Double Vote: {inv.get('doublevote', 0)}\n"
        f"Extra Life: {inv.get('extralife', 0)}"
    )
    await update.callback_query.edit_message_text(text, reply_markup=build_profile_keyboard())


async def mafia_guide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "HOW TO PLAY MAFIA\n\n"
        "1. Start /mafia in group\n"
        "2. Players join\n"
        "3. Roles are assigned\n"
        "4. Night phase actions\n"
        "5. Day phase voting\n"
        "6. Town vs Mafia until win\n\n"
        "Commands:\n"
        "/mafia, /join, /myrole, /leaderboard"
    )
    keyboard = [
        [InlineKeyboardButton("Start Game", callback_data="mafia_start_group")],
        [InlineKeyboardButton("Back", callback_data="mafia_hub")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


def _help_text() -> str:
    return (
        "HELP PANEL\n\n"
        "Chat Commands:\n"
        "/chat - Start chat mode\n"
        "/ask <question> - Ask anything\n"
        "/help - Open this panel\n\n"
        "Mafia Commands:\n"
        "/mafia [seconds] - Create mafia lobby in group\n"
        "/join - Join active mafia lobby\n"
        "/myrole - Show your role\n"
        "/buy <item> - Buy item\n"
        "/leaderboard - Show top players\n\n"
        "Group Admin:\n"
        "/extend - Extend join timer\n"
        "/forcestart - Force start game\n\n"
        "Utility:\n"
        "/start - Open main panel\n"
        "/buildinfo - Show deployed commit\n\n"
        "Chat works in private chat directly.\n"
        "In groups, mention bot, reply to bot, or use 'baby'."
    )


async def help_panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = _help_text()
    keyboard = [
        [InlineKeyboardButton("Open Game Hub", callback_data="mafia_hub")],
        [InlineKeyboardButton("Chat Guide", callback_data="chatguide")],
        [InlineKeyboardButton("Back", callback_data="back_start")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Open Game Hub", callback_data="mafia_hub")],
        [InlineKeyboardButton("Chat Guide", callback_data="chatguide")],
    ]
    await update.message.reply_text(_help_text(), reply_markup=InlineKeyboardMarkup(keyboard))


async def chatguide_panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "CHAT GUIDE\n\n"
        "Private Chat:\n"
        "- Just send a normal message.\n"
        "- Use /ask <question> for direct Q/A.\n\n"
        "Group Chat:\n"
        "- Mention bot username\n"
        "- Reply to bot message\n"
        "- Or include keyword: baby\n\n"
        "Quick commands: /chat, /ask, /help"
    )
    keyboard = [
        [InlineKeyboardButton("Help", callback_data="help")],
        [InlineKeyboardButton("Back", callback_data="back_start")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def settings_panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "SETTINGS\n\n"
        "Current module settings are environment based.\n\n"
        "Required env:\n"
        "- BOT_TOKEN\n"
        "- BOT_USERNAME\n"
        "- CHANNEL_USERNAME\n"
        "- CONTACT_USERNAME\n\n"
        "Use /buildinfo to verify deployment commit."
    )
    keyboard = [
        [InlineKeyboardButton("Help", callback_data="help")],
        [InlineKeyboardButton("Back", callback_data="back_start")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def chat_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Chat mode active. Message bhejo, main reply karungi.")


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Use: /ask <question>")
        return
    prompt = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    reply = await generate_chat_reply(prompt)
    await update.message.reply_text(reply)


def _should_reply_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    lowered = text.lower()
    bot_username = (context.bot.username or BOT_USERNAME).lower()
    mention_hit = bot_username and f"@{bot_username}" in lowered
    keyword_hit = "baby" in lowered
    reply_hit = (
        update.effective_message.reply_to_message is not None
        and update.effective_message.reply_to_message.from_user is not None
        and update.effective_message.reply_to_message.from_user.id == context.bot.id
    )
    return mention_hit or keyword_hit or reply_hit


async def text_chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    text = msg.text.strip()
    if not text:
        return

    if update.effective_chat.type != ChatType.PRIVATE and not _should_reply_in_group(update, context, text):
        return

    bot_username = (context.bot.username or BOT_USERNAME).lower()
    if bot_username:
        text = text.replace(f"@{bot_username}", "").strip()
    if not text:
        text = "Hi"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    reply = await generate_chat_reply(text)
    await msg.reply_text(reply)


async def mafia_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    join_time = 60
    if context.args:
        try:
            join_time = int(context.args[0])
        except ValueError:
            pass
    _launch_join_lobby(chat_id, join_time, context)
    await update.message.reply_text(_lobby_text(chat_id), reply_markup=build_mafia_lobby_keyboard())


async def join_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ok, message = join_game(update.effective_chat.id, update.effective_user.id)
    await update.message.reply_text(message if ok else f"❌ {message}")


async def extend_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remaining = extend_join_time(update.effective_chat.id, 30)
    if remaining is None:
        await update.message.reply_text("No active joining phase to extend.")
        return
    await update.message.reply_text(f"Join time extended by 30 seconds. Now: {remaining} sec")


async def forcestart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    game = active_games.get(chat_id)
    if not game:
        await update.message.reply_text("No active game.")
        return
    if not await _is_admin_chat(chat_id, update.effective_user.id, context):
        await update.message.reply_text("Only admins can use /forcestart.")
        return
    if len(game["players"]) < MIN_PLAYERS:
        await update.message.reply_text("Not enough players.")
        return
    await update.message.reply_text("Admin started the game manually.")
    await start_game(chat_id, context)


async def myrole_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    for _, game in active_games.items():
        if user_id in game.get("players", []):
            role = game.get("roles", {}).get(user_id)
            if role:
                await update.message.reply_text(f"Your Role: {role.upper()}")
                return
    await update.message.reply_text("You are not in any active game.")


async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = top_players()
    if not rows:
        await update.message.reply_text("Leaderboard is empty.")
        return
    lines = ["Top Players"]
    for idx, (uid, wins) in enumerate(rows, start=1):
        lines.append(f"{idx}. {uid} - {wins} wins ({get_rank(int(wins))})")
    await update.message.reply_text("\n".join(lines))


async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Use: /buy <item>")
        return
    await update.message.reply_text(buy(update.effective_user.id, context.args[0]))


async def buildinfo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    commit = (
        os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("SOURCE_VERSION")
        or os.getenv("GIT_COMMIT")
        or ""
    )
    if not commit:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"
    await update.message.reply_text(f"Build commit: {commit}")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    data = q.data
    user_id = q.from_user.id
    chat_id = q.message.chat.id

    if data == "back_start":
        first_name = q.from_user.first_name or "Friend"
        await q.edit_message_text(
            start_panel_text(first_name),
            reply_markup=build_start_keyboard(context.bot.username or BOT_USERNAME),
        )
        return
    if data == "mafia_hub":
        await mafia_hub(update, context)
        return
    if data == "mafia_roles":
        await mafia_roles(update, context)
        return
    if data.startswith("role_"):
        await role_info(update, context)
        return
    if data == "mafia_shop":
        await mafia_shop(update, context)
        return
    if data == "mafia_profile":
        await mafia_profile(update, context)
        return
    if data == "mafia_guide":
        await mafia_guide(update, context)
        return
    if data == "help":
        await help_panel(update, context)
        return
    if data == "chatguide":
        await chatguide_panel(update, context)
        return
    if data == "settings":
        await q.answer("Settings is temporarily hidden.", show_alert=True)
        return

    if data == "mafia_leaderboard":
        rows = top_players()
        text = "Top Players\n\n" + (
            "\n".join([f"{i}. {u} - {w} Wins" for i, (u, w) in enumerate(rows, 1)])
            if rows
            else "No entries yet."
        )
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="mafia_hub")]]))
        return
    if data == "mafia_start_group":
        if q.message.chat.type == ChatType.PRIVATE:
            await q.edit_message_text(
                "Start mafia in a group using /mafia there.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Add To Group", url=f"https://t.me/{BOT_USERNAME}?startgroup=true")]]
                ),
            )
            return
        _launch_join_lobby(chat_id, 60, context)
        await q.edit_message_text(_lobby_text(chat_id), reply_markup=build_mafia_lobby_keyboard())
        return
    if data == "mafia_join":
        ok, msg = join_game(chat_id, user_id)
        prefix = "OK" if ok else "ERR"
        await q.edit_message_text(f"{_lobby_text(chat_id)}\n\n{prefix}: {msg}", reply_markup=build_mafia_lobby_keyboard())
        return
    if data == "mafia_force_start":
        if not await _is_admin_chat(chat_id, user_id, context):
            await q.answer("Admins only.", show_alert=True)
            return
        game = active_games.get(chat_id)
        if not game or len(game["players"]) < MIN_PLAYERS:
            await q.answer("Not enough players.", show_alert=True)
            return
        await q.edit_message_text("Admin started the game manually.")
        await start_game(chat_id, context)
        return
    if data == "mafia_extend":
        remaining = extend_join_time(chat_id, 30)
        if remaining is None:
            await q.answer("No joining phase active.", show_alert=True)
            return
        await q.edit_message_text(f"{_lobby_text(chat_id)}\n\nExtended. Remaining: {remaining}s", reply_markup=build_mafia_lobby_keyboard())
        return
    if data == "mafia_cancel":
        if not await _is_admin_chat(chat_id, user_id, context):
            await q.answer("Admins only.", show_alert=True)
            return
        if not cancel_game(chat_id):
            await q.answer("No active game.", show_alert=True)
            return
        await q.edit_message_text("Game cancelled by admin.")
        return
    if data in {"buy_shield", "buy_doublevote", "buy_extralife"}:
        item = data.split("buy_", 1)[1]
        msg = buy(user_id, item)
        coins = balance(user_id)
        await q.edit_message_text(
            f"MAFIA SHOP\n\nCoins: {coins}\n\n{msg}",
            reply_markup=build_mafia_shop_keyboard(),
        )
        return

    try:
        parts = data.split("_")
        action = parts[0]

        if action == "vote":
            _, chat_id_str, target_str = parts
            game_chat_id = int(chat_id_str)
            target = int(target_str)

            game = active_games.get(game_chat_id)
            if not game or game.get("phase") != "day":
                return
            if user_id not in game.get("alive", []):
                return
            if user_id == game.get("silenced"):
                return
            if user_id in game.get("day_voters", set()):
                return

            role = game["roles"].get(user_id)
            vote_power = 2 if role == "mayor" else 1
            inv = get_inventory(user_id)
            if role != "mayor" and inv.get("doublevote", 0) > 0 and use_item(user_id, "doublevote"):
                vote_power = 2
            game["votes"][target] = game["votes"].get(target, 0) + vote_power
            game["day_voters"].add(user_id)
            return

        if action == "night":
            _, ability, chat_id_str, target_str = parts
            game_chat_id = int(chat_id_str)
            target = int(target_str)

            game = active_games.get(game_chat_id)
            if not game or game.get("phase") != "night":
                return
            if user_id not in game.get("alive", []):
                return

            if ability == "kill" and game["roles"].get(user_id) == "mafia":
                game["night_actions"]["kill"] = target
            elif ability == "save" and game["roles"].get(user_id) == "doctor":
                game["night_actions"]["save"] = target
            elif ability == "check" and game["roles"].get(user_id) == "detective":
                role = game["roles"].get(target, "unknown")
                await context.bot.send_message(user_id, f"Role: {role}")
            elif ability == "heal" and game["roles"].get(user_id) == "witch" and game["witch_potions"]["heal"] > 0:
                game["night_actions"]["heal"] = target
                game["witch_potions"]["heal"] -= 1
            elif ability == "poison" and game["roles"].get(user_id) == "witch" and game["witch_potions"]["poison"] > 0:
                game["night_actions"]["poison"] = target
                game["witch_potions"]["poison"] -= 1
            elif ability == "silence" and game["roles"].get(user_id) == "silencer":
                game["silenced"] = target
            return
    except Exception:
        pass


def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("Set BOT_TOKEN environment variable.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("mafia", mafia_cmd))
    app.add_handler(CommandHandler("join", join_cmd))
    app.add_handler(CommandHandler("extend", extend_cmd))
    app.add_handler(CommandHandler("forcestart", forcestart_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("myrole", myrole_cmd))
    app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
    app.add_handler(CommandHandler("buildinfo", buildinfo_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_chat_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.run_polling()


if __name__ == "__main__":
    main()
