import asyncio
import os
import subprocess

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatMemberStatus
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from game.economy import balance
from game.inventory import get_inventory, use_item
from game.leaderboard import get_rank, top_players
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
from game.roles import ROLE_INFO
from game.shop import SHOP_ITEMS, buy

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BOT_USERNAME = os.getenv("BOT_USERNAME", "YOUR_BOT_USERNAME").strip("@")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "YOUR_CHANNEL").strip("@")
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "YOUR_CONTACT").strip("@")


def build_start_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                "💬 Chat With Me",
                url=f"https://t.me/{BOT_USERNAME}",
            )
        ],
        [
            InlineKeyboardButton(
                "➕ Add To Group",
                url=f"https://t.me/{BOT_USERNAME}?startgroup=true",
            )
        ],
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("🎤 VC Guide", callback_data="vcguide"),
        ],
        [
            InlineKeyboardButton("📢 Channel", url=f"https://t.me/{CHANNEL_USERNAME}"),
            InlineKeyboardButton("⚙ Settings", callback_data="settings"),
        ],
        [
            InlineKeyboardButton(
                "📩 Contact / Promotion",
                url=f"https://t.me/{CONTACT_USERNAME}",
            )
        ],
        [
            InlineKeyboardButton("🎮 Mafia Game", callback_data="mafia"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_mafia_lobby_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("➕ Join Game", callback_data="mafia_join")],
            [InlineKeyboardButton("🚀 Start Now", callback_data="mafia_force_start")],
            [InlineKeyboardButton("⏱ Extend Join Time", callback_data="mafia_extend")],
            [InlineKeyboardButton("❌ Cancel Game", callback_data="mafia_cancel")],
            [
                InlineKeyboardButton("🛒 Shop", callback_data="mafia_shop"),
                InlineKeyboardButton("👤 My Profile", callback_data="mafia_profile"),
            ],
        ]
    )


def build_mafia_shop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🛡 Shield (30)", callback_data="mafia_buy_shield")],
            [InlineKeyboardButton("🗳 Double Vote (40)", callback_data="mafia_buy_doublevote")],
            [InlineKeyboardButton("❤️ Extra Life (50)", callback_data="mafia_buy_extralife")],
            [InlineKeyboardButton("🔙 Back", callback_data="mafia_back")],
        ]
    )


def build_profile_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🛒 Open Shop", callback_data="mafia_shop")],
            [InlineKeyboardButton("🏆 Leaderboard", callback_data="mafia_leaderboard")],
            [InlineKeyboardButton("🔙 Back", callback_data="mafia_back")],
        ]
    )


def build_back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="mafia_back")]])


def _lobby_text(chat_id: int) -> str:
    game = active_games.get(chat_id)
    joined = len(game["players"]) if game else 0
    return (
        "🎭 MAFIA GAME LOBBY\n\n"
        f"Players Joined: {joined} / 25\n\n"
        "⏳ Waiting for players..."
    )


def _launch_join_lobby(chat_id: int, join_time: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    create_game(chat_id, join_time=join_time)
    task = asyncio.create_task(start_join_timer(chat_id, context))
    active_games[chat_id]["join_task"] = task


async def _is_admin_chat(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    member = await context.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}


def _wins_and_rank(user_id: int) -> tuple[int, str, int | None]:
    rows = top_players()
    wins = 0
    rank_pos = None
    for idx, (uid, w) in enumerate(rows, start=1):
        if str(user_id) == str(uid):
            wins = int(w)
            rank_pos = idx
            break
    return wins, get_rank(wins), rank_pos


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user.first_name
    text = (
        f"✨ HEY BABY {user} NICE TO MEET YOU 🌹\n\n"
        "◎ THIS IS 『ANIMX MUSIC』\n\n"
        "➤ A premium designed music player bot for Telegram groups & channels.\n\n"
        "🎧 HD Voice Chat Streaming\n"
        "🚀 Fast • Smart • Always Active\n"
        "💬 Chat Naturally Like a Friend\n\n"
        "🌙 Good Evening 💖"
    )
    await update.message.reply_text(text, reply_markup=build_start_keyboard())


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
    chat_id = update.effective_chat.id
    remaining = extend_join_time(chat_id, 30)
    if remaining is None:
        await update.message.reply_text("No active joining phase to extend.")
        return
    await update.message.reply_text(f"⏳ Join time extended by 30 seconds.\nNow: {remaining} sec")


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
        await update.message.reply_text("⚠ Not enough players.")
        return
    await update.message.reply_text("🚀 Admin started the game manually!")
    await start_game(chat_id, context)


async def myrole_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    for _, game in active_games.items():
        if user_id in game.get("players", []):
            role = game.get("roles", {}).get(user_id)
            if role:
                await update.message.reply_text(f"🎭 Your Role: {role.upper()}")
                return
    await update.message.reply_text("❌ You are not in any active game.")


async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = top_players()
    if not rows:
        await update.message.reply_text("🏆 Leaderboard is empty.")
        return
    lines = ["🏆 Top Players"]
    for idx, (uid, wins) in enumerate(rows, start=1):
        lines.append(f"{idx}. {uid} - {wins} wins ({get_rank(int(wins))})")
    await update.message.reply_text("\n".join(lines))


async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Use: /buy <item>")
        return
    msg = buy(update.effective_user.id, context.args[0])
    await update.message.reply_text(msg)


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

    if data in {"help", "vcguide", "settings"}:
        await q.edit_message_text("This panel is under setup.", reply_markup=build_start_keyboard())
        return

    if data == "mafia":
        _launch_join_lobby(chat_id, 60, context)
        await q.edit_message_text(_lobby_text(chat_id), reply_markup=build_mafia_lobby_keyboard())
        return

    if data == "mafia_join":
        ok, msg = join_game(chat_id, user_id)
        prefix = "✅" if ok else "❌"
        await q.edit_message_text(
            f"{_lobby_text(chat_id)}\n\n{prefix} {msg}",
            reply_markup=build_mafia_lobby_keyboard(),
        )
        return

    if data == "mafia_force_start":
        if not await _is_admin_chat(chat_id, user_id, context):
            await q.answer("Admins only.", show_alert=True)
            return
        game = active_games.get(chat_id)
        if not game or len(game["players"]) < MIN_PLAYERS:
            await q.answer("Not enough players.", show_alert=True)
            return
        await q.edit_message_text("🚀 Admin started the game manually!")
        await start_game(chat_id, context)
        return

    if data == "mafia_extend":
        remaining = extend_join_time(chat_id, 30)
        if remaining is None:
            await q.answer("No joining phase active.", show_alert=True)
            return
        await q.edit_message_text(
            f"{_lobby_text(chat_id)}\n\n⏳ Extended. Remaining: {remaining}s",
            reply_markup=build_mafia_lobby_keyboard(),
        )
        return

    if data == "mafia_cancel":
        if not await _is_admin_chat(chat_id, user_id, context):
            await q.answer("Admins only.", show_alert=True)
            return
        if not cancel_game(chat_id):
            await q.answer("No active game.", show_alert=True)
            return
        await q.edit_message_text("❌ Game cancelled by admin.")
        return

    if data == "mafia_shop":
        coins = balance(user_id)
        await q.edit_message_text(
            f"🛒 MAFIA SHOP\n\n💰 Your Coins: {coins}",
            reply_markup=build_mafia_shop_keyboard(),
        )
        return

    if data.startswith("mafia_buy_"):
        item = data.split("mafia_buy_", 1)[1]
        if item not in SHOP_ITEMS:
            await q.answer("Invalid item", show_alert=True)
            return
        msg = buy(user_id, item)
        coins = balance(user_id)
        await q.edit_message_text(
            f"🛒 MAFIA SHOP\n\n💰 Your Coins: {coins}\n\n{msg}",
            reply_markup=build_mafia_shop_keyboard(),
        )
        return

    if data == "mafia_profile":
        inv = get_inventory(user_id)
        wins, tier, rank_pos = _wins_and_rank(user_id)
        coins = balance(user_id)
        rank_text = f"#{rank_pos}" if rank_pos is not None else "Unranked"
        text = (
            "👤 PLAYER PROFILE\n\n"
            f"Coins: {coins}\n"
            f"Wins: {wins}\n"
            "Losses: 0\n"
            f"Season Rank: {rank_text} ({tier})\n\n"
            "Inventory:\n"
            f"🛡 Shield x{inv.get('shield', 0)}\n"
            f"🗳 Double Vote x{inv.get('doublevote', 0)}\n"
            f"❤️ Extra Life x{inv.get('extralife', 0)}"
        )
        await q.edit_message_text(text, reply_markup=build_profile_keyboard())
        return

    if data == "mafia_leaderboard":
        rows = top_players()
        if not rows:
            text = "🏆 Top Players\n\nNo entries yet."
        else:
            lines = ["🏆 Top Players\n"]
            for idx, (uid, wins) in enumerate(rows, start=1):
                lines.append(f"{idx}️⃣ {uid} - {wins} Wins")
            text = "\n".join(lines)
        await q.edit_message_text(text, reply_markup=build_back_keyboard())
        return

    if data == "mafia_back":
        await q.edit_message_text(_lobby_text(chat_id), reply_markup=build_mafia_lobby_keyboard())
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

            if ability == "kill":
                if game["roles"].get(user_id) == "mafia":
                    game["night_actions"]["kill"] = target
            elif ability == "save":
                if game["roles"].get(user_id) == "doctor":
                    game["night_actions"]["save"] = target
            elif ability == "check":
                if game["roles"].get(user_id) == "detective":
                    role = game["roles"].get(target, "unknown")
                    await context.bot.send_message(user_id, f"🕵 Role: {role}")
            elif ability == "heal":
                if game["roles"].get(user_id) == "witch" and game["witch_potions"]["heal"] > 0:
                    game["night_actions"]["heal"] = target
                    game["witch_potions"]["heal"] -= 1
            elif ability == "poison":
                if game["roles"].get(user_id) == "witch" and game["witch_potions"]["poison"] > 0:
                    game["night_actions"]["poison"] = target
                    game["witch_potions"]["poison"] -= 1
            elif ability == "silence":
                if game["roles"].get(user_id) == "silencer":
                    game["silenced"] = target
            return
    except Exception:
        pass


def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("Set BOT_TOKEN environment variable.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mafia", mafia_cmd))
    app.add_handler(CommandHandler("join", join_cmd))
    app.add_handler(CommandHandler("extend", extend_cmd))
    app.add_handler(CommandHandler("forcestart", forcestart_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("myrole", myrole_cmd))
    app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
    app.add_handler(CommandHandler("buildinfo", buildinfo_cmd))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.run_polling()


if __name__ == "__main__":
    main()
