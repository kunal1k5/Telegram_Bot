import asyncio
import os
import subprocess

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatMemberStatus, ChatType
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
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
from game.shop import SHOP_ITEMS, buy

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BOT_USERNAME = os.getenv("BOT_USERNAME", "YOUR_BOT_USERNAME").strip("@")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "YOUR_CHANNEL").strip("@")
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "YOUR_CONTACT").strip("@")


def build_start_keyboard(bot_username: str) -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("💬 Chat With Me", url=f"https://t.me/{bot_username}")],
        [
            InlineKeyboardButton(
                "➕ Add To Group",
                url=f"https://t.me/{bot_username}?startgroup=true",
            )
        ],
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("🎤 VC Guide", callback_data="vcguide"),
        ],
        [
            InlineKeyboardButton("📢 Channel", url="https://t.me/YOUR_CHANNEL"),
            InlineKeyboardButton("⚙ Settings", callback_data="settings"),
        ],
        [InlineKeyboardButton("🎮 Mafia Game", callback_data="mafia_hub")],
        [
            InlineKeyboardButton(
                "📩 Contact / Promotion",
                url="https://t.me/YOUR_CONTACT",
            )
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
            [InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")],
        ]
    )


def build_mafia_shop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🛡 Buy Shield", callback_data="buy_shield")],
            [InlineKeyboardButton("🗳 Buy Double Vote", callback_data="buy_doublevote")],
            [InlineKeyboardButton("❤️ Buy Extra Life", callback_data="buy_extralife")],
            [InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")],
        ]
    )


def build_profile_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🛒 Open Shop", callback_data="mafia_shop")],
            [InlineKeyboardButton("🏆 Leaderboard", callback_data="mafia_leaderboard")],
            [InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")],
        ]
    )


def _lobby_text(chat_id: int) -> str:
    game = active_games.get(chat_id)
    joined = len(game["players"]) if game else 0
    return f"🎭 MAFIA GAME LOBBY\n\nPlayers Joined: {joined} / 25\n\n⏳ Waiting for players..."


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
    bot_username = context.bot.username
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
    await update.message.reply_text(text, reply_markup=build_start_keyboard(bot_username))


async def mafia_hub(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🎭 MAFIA GAME HUB\n\n"
        "Welcome to the ultimate social strategy battle.\n\n"
        "💀 Deception.\n"
        "🕵 Investigation.\n"
        "🛡 Protection.\n"
        "🔥 Survival.\n\n"
        "Choose what you want to explore:"
    )
    keyboard = [
        [InlineKeyboardButton("📜 Roles & Powers", callback_data="mafia_roles")],
        [InlineKeyboardButton("🛒 Shop", callback_data="mafia_shop")],
        [InlineKeyboardButton("👤 My Profile", callback_data="mafia_profile")],
        [InlineKeyboardButton("📖 How To Play", callback_data="mafia_guide")],
        [InlineKeyboardButton("🚀 Start Game (Group)", callback_data="mafia_start_group")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_start")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def mafia_roles(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("🔪 Mafia", callback_data="role_mafia")],
        [InlineKeyboardButton("🛡 Doctor", callback_data="role_doctor")],
        [InlineKeyboardButton("🕵 Detective", callback_data="role_detective")],
        [InlineKeyboardButton("🧙 Witch", callback_data="role_witch")],
        [InlineKeyboardButton("🤫 Silencer", callback_data="role_silencer")],
        [InlineKeyboardButton("👑 Mayor", callback_data="role_mayor")],
        [InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")],
    ]
    await update.callback_query.edit_message_text(
        "📜 Select a role to see its power:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def role_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    role = update.callback_query.data.split("_")[1]
    role_texts = {
        "mafia": "🔪 Mafia\nKill one player each night.\nGoal: Outnumber town.",
        "doctor": "🛡 Doctor\nSave one player each night.",
        "detective": "🕵 Detective\nReveal role of one player.",
        "witch": "🧙 Witch\n1 Heal potion + 1 Poison potion.",
        "silencer": "🤫 Silencer\nMute one player next day.",
        "mayor": "👑 Mayor\nYour vote counts as 2 permanently.",
    }
    keyboard = [
        [InlineKeyboardButton("🛒 Open Shop", callback_data="mafia_shop")],
        [InlineKeyboardButton("👤 My Profile", callback_data="mafia_profile")],
        [InlineKeyboardButton("🔙 Back", callback_data="mafia_roles")],
    ]
    await update.callback_query.edit_message_text(
        role_texts.get(role, "Role not found."),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def mafia_shop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    coins = balance(user_id)
    text = (
        "🛒 MAFIA SHOP\n\n"
        f"💰 Coins: {coins}\n\n"
        "🛡 Shield - 30 coins\n"
        "🗳 Double Vote - 40 coins\n"
        "❤️ Extra Life - 50 coins"
    )
    await update.callback_query.edit_message_text(text, reply_markup=build_mafia_shop_keyboard())


async def mafia_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    coins = balance(user_id)
    inv = get_inventory(user_id)
    wins, tier, pos_text = _wins_rank_text(user_id)
    text = (
        "👤 PLAYER PROFILE\n\n"
        f"💰 Coins: {coins}\n"
        f"🏆 Rank: {pos_text} ({tier})\n"
        f"🎯 Wins: {wins}\n\n"
        "🎒 Inventory:\n"
        f"🛡 Shield: {inv.get('shield', 0)}\n"
        f"🗳 Double Vote: {inv.get('doublevote', 0)}\n"
        f"❤️ Extra Life: {inv.get('extralife', 0)}"
    )
    await update.callback_query.edit_message_text(text, reply_markup=build_profile_keyboard())


async def mafia_guide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📖 HOW TO PLAY MAFIA\n\n"
        "1️⃣ Group me /mafia start karo\n"
        "2️⃣ Players join karte hain\n"
        "3️⃣ Roles secretly DM me milte hain\n"
        "4️⃣ Night phase - special powers use\n"
        "5️⃣ Day phase - voting\n"
        "6️⃣ Mafia vs Town battle until one wins\n\n"
        "Commands:\n"
        "🎮 /mafia - Start game\n"
        "📜 /myrole - Check your role\n"
        "🏆 /leaderboard - View ranking\n\n"
        "Goal:\n"
        "Town wins -> Eliminate all mafia\n"
        "Mafia wins -> Outnumber town"
    )
    keyboard = [
        [InlineKeyboardButton("🚀 Start Game", callback_data="mafia_start_group")],
        [InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")],
    ]
    await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


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
        await q.edit_message_text(
            "Main panel",
            reply_markup=build_start_keyboard(context.bot.username),
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
    if data == "mafia_leaderboard":
        rows = top_players()
        text = "🏆 Top Players\n\n" + (
            "\n".join([f"{i}. {u} - {w} Wins" for i, (u, w) in enumerate(rows, 1)])
            if rows
            else "No entries yet."
        )
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="mafia_hub")]]))
        return
    if data == "mafia_start_group":
        if q.message.chat.type == ChatType.PRIVATE:
            await q.edit_message_text(
                "🚀 Start mafia in a group using /mafia there.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("➕ Add To Group", url=f"https://t.me/{BOT_USERNAME}?startgroup=true")]]
                ),
            )
            return
        _launch_join_lobby(chat_id, 60, context)
        await q.edit_message_text(_lobby_text(chat_id), reply_markup=build_mafia_lobby_keyboard())
        return
    if data == "mafia_join":
        ok, msg = join_game(chat_id, user_id)
        await q.edit_message_text(f"{_lobby_text(chat_id)}\n\n{'✅' if ok else '❌'} {msg}", reply_markup=build_mafia_lobby_keyboard())
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
        await q.edit_message_text(f"{_lobby_text(chat_id)}\n\n⏳ Extended. Remaining: {remaining}s", reply_markup=build_mafia_lobby_keyboard())
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
    if data in {"buy_shield", "buy_doublevote", "buy_extralife"}:
        item = data.split("buy_", 1)[1]
        msg = buy(user_id, item)
        coins = balance(user_id)
        await q.edit_message_text(
            f"🛒 MAFIA SHOP\n\n💰 Coins: {coins}\n\n{msg}",
            reply_markup=build_mafia_shop_keyboard(),
        )
        return
    if data in {"help", "vcguide", "settings"}:
        await q.edit_message_text(
            "This panel is under setup.",
            reply_markup=build_start_keyboard(context.bot.username),
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
                await context.bot.send_message(user_id, f"🕵 Role: {role}")
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
