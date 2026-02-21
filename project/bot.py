import asyncio
import os

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
    create_game,
    extend_join_time,
    join_game,
    start_game,
    start_join_timer,
)
from game.roles import ROLE_INFO
from game.shop import buy

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎮 Mafia Game", callback_data="mafia")],
            [InlineKeyboardButton("👤 Profile", callback_data="profile")],
            [InlineKeyboardButton("🛒 Shop", callback_data="shop")],
        ]
    )


async def _is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not update.effective_chat or not update.effective_user:
        return False
    member = await context.bot.get_chat_member(
        chat_id=update.effective_chat.id,
        user_id=update.effective_user.id,
    )
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}


def _launch_join_lobby(chat_id: int, join_time: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    create_game(chat_id, join_time=join_time)
    task = asyncio.create_task(start_join_timer(chat_id, context))
    active_games[chat_id]["join_task"] = task


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🔥 ANIMX GAME SYSTEM", reply_markup=main_menu())


async def mafia_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    join_time = 60
    if context.args:
        try:
            join_time = int(context.args[0])
        except ValueError:
            pass

    _launch_join_lobby(chat_id, join_time, context)
    await update.message.reply_text(f"🎮 Lobby created with {join_time}s join time. Use /join")


async def extend_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    remaining = extend_join_time(chat_id, 30)
    if remaining is None:
        await update.message.reply_text("No active joining phase to extend.")
        return
    await update.message.reply_text(
        f"⏳ Join time extended by 30 seconds.\nNow: {remaining} sec"
    )


async def forcestart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    game = active_games.get(chat_id)
    if not game:
        await update.message.reply_text("No active game.")
        return

    if not await _is_admin(update, context):
        await update.message.reply_text("Only admins can use /forcestart.")
        return

    if len(game["players"]) < MIN_PLAYERS:
        await update.message.reply_text("⚠ Not enough players.")
        return

    await update.message.reply_text("🚀 Admin started the game manually!")
    await start_game(chat_id, context)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()

    data = q.data
    user_id = q.from_user.id
    chat_id = q.message.chat.id

    if data == "mafia":
        _launch_join_lobby(chat_id, 60, context)
        await q.edit_message_text("🎮 Lobby created! You have 60s to join with /join")
        return

    if data == "profile":
        coins = balance(user_id)
        inv = get_inventory(user_id)
        await q.edit_message_text(f"Coins: {coins}\nInventory: {inv}")
        return

    if data == "shop":
        await q.edit_message_text("Use /buy shield | extralife | doublevote | reveal")
        return

    if data.startswith("role_"):
        key = data.split("_", 1)[1]
        await q.edit_message_text(ROLE_INFO.get(key, "No role info found."))
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


async def join_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ok, message = join_game(update.effective_chat.id, update.effective_user.id)
    await update.message.reply_text(message if ok else f"❌ {message}")


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await start_game(update.effective_chat.id, context):
        await update.message.reply_text("Game Started!")
    else:
        await update.message.reply_text("Need at least 5 players.")


async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Use: /buy <item>")
        return
    item = context.args[0]
    msg = buy(update.effective_user.id, item)
    await update.message.reply_text(msg)


async def leaderboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = top_players()
    if not rows:
        await update.message.reply_text("🏆 Leaderboard is empty.")
        return

    lines = ["🏆 Top Players"]
    for idx, (uid, wins) in enumerate(rows, start=1):
        lines.append(f"{idx}. {uid} - {wins} wins ({get_rank(int(wins))})")
    await update.message.reply_text("\n".join(lines))


async def myrole_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    for _, game in active_games.items():
        if user_id in game.get("players", []):
            role = game.get("roles", {}).get(user_id)
            if role:
                await update.message.reply_text(f"🎭 Your Role: {role.upper()}")
                return

    await update.message.reply_text("❌ You are not in any active game.")


def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("Set BOT_TOKEN environment variable.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mafia", mafia_cmd))
    app.add_handler(CommandHandler("join", join_cmd))
    app.add_handler(CommandHandler("extend", extend_cmd))
    app.add_handler(CommandHandler("forcestart", forcestart_cmd))
    app.add_handler(CommandHandler("startgame", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("myrole", myrole_cmd))
    app.add_handler(CommandHandler("leaderboard", leaderboard_cmd))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.run_polling()


if __name__ == "__main__":
    main()
