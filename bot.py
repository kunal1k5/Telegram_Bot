import asyncio
import logging
import os
import random
from typing import Final

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from config import BOT_TOKEN

# On Windows, make sure to use the selector event loop policy for asyncio
if os.name == "nt":  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ANIMX_CLAN_CHAT")

BOT_NAME: Final[str] = "ANIMX CLAN Chat Bot"
BOT_USERNAME: Final[str] = "@AnimxClanBot"
OWNER_USERNAME: Final[str] = "@kunal1k5"
CHANNEL_USERNAME: Final[str] = "@AnimxClanChannel"

START_TEXT: Final[str] = (
    "🎉 *Namaste! Main ANIMX CLAN Chat Bot hoon* 🎉\n\n"
    "Mujhe chat karna bahut pasand hai! 😊\n\n"
    "*Main kya kar sakta hoon:*\n"
    "💬 Tere saath friendly chat\n"
    "😄 Jokes aur masti\n"
    "🎯 Helpful tips\n"
    "🌟 Sirf tera time pass!\n\n"
    "Bas message kar, aur dekh mujhe kaisa jawaab deta hoon! 🚀"
)

HELP_TEXT: Final[str] = (
    "📚 *Help & Commands* 📚\n\n"
    "`/start` - Welcome message\n"
    "`/help` - Ye message\n"
    "`/joke` - Funny joke sunao 😂\n\n"
    "*Group Mein:*\n"
    "Mujhe mention kar, main reply dunga! 👋\n"
    "Ya bas koi message kar, main suno 👂\n\n"
    "*Tips:*\n"
    "• Hinglish use kar - mazaa ayega!\n"
    "• Zada serious mat ban 😄\n"
    "• Mujhe suggestions bhej sakte ho"
)

# Hinglish responses database
HINGLISH_RESPONSES = [
    "Haan bhai! 😄 Kaisa chal raha? Sab theek?",
    "Yo! Kya haal hai? 👋 Tera din kaisa hai?",
    "Namaste! 🙏 Kya bolraha hai tu?",
    "Arey! Tere liye time nikala? 💪 Appreciated!",
    "Main yaha hoon! Kya help chahiye? 😊",
    "Bhai, tu mera favorite person hai! ❤️",
    "Kya bat hai? Kuch naya?",
    "Haha! Main samajhta hoon 😂",
    "Ekdum sahi kaha! 💯",
    "Agreed, agreed! 🤝",
]

GREETING_KEYWORDS = [
    "hi", "hello", "hey", "hii", "heya", "yo", "sup",
    "namaste", "namaskar", "haan", "kya", "howdy"
]

ASK_KEYWORDS = [
    "how are you", "kaisa hai", "kaisi ho", "kaise ho",
    "how's it", "acha", "badiya", "maza", "good"
]

MOOD_RESPONSES = {
    "happy": [
        "Bahut accha! Main bhi khush hoon tere liye! 🎉",
        "Yaaay! Tera khushi meri khushi! ✨",
        "Amazing! Celebrations ho rahi? 🥳",
    ],
    "sad": [
        "Arre, kya hua? Main yaha hoon support karne ke liye 💪",
        "Dukh mat kar bhai! Sab theek ho jayega 🌟",
        "Tension mat le! Tera friend ANIMX CLAN yaha hai 🤗",
    ],
    "confused": [
        "Samajh nahi aaya? Mujhe samjha! 🤔",
        "Confuse ho gaya? Main clear kar dunga! 📝",
        "Kya samajhne mein problem hai? Bol!",
    ],
    "excited": [
        "Bhai wow! Tera energy amazing hai! ⚡",
        "Excited aa gya! Kya scene hai? 🔥",
        "Arey wahh! Itni excitement se! 🚀",
    ],
}


# ========================= HELPER FUNCTIONS ========================= #


def get_mood(text: str) -> str:
    """Detect mood from text"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["sad", "dukh", "tension", "worried", "upset", "bad", "galat"]):
        return "sad"
    elif any(word in text_lower for word in ["happy", "khush", "amazing", "awesome", "yay", "great", "love"]):
        return "happy"
    elif any(word in text_lower for word in ["confuse", "samajh", "confused", "kya", "what", "understand", "clear"]):
        return "confused"
    elif any(word in text_lower for word in ["excited", "wow", "wah", "omg", "amazing", "incredible", "crazy"]):
        return "excited"
    
    return "neutral"


def get_hinglish_reply(text: str) -> str:
    """Generate human-like Hinglish reply"""
    text_lower = text.lower()
    mood = get_mood(text)
    
    # Greeting responses
    if any(word in text_lower for word in GREETING_KEYWORDS):
        return random.choice([
            "Haan bhai! 😄 Kaisa chal raha?",
            "Yo! Kya haal? Maza aa raha?",
            "Namaste! Tera din kaisa raha?",
            "Hey there! 👋 Sab theek?",
        ])
    
    # Mood-based responses
    if mood in MOOD_RESPONSES:
        return random.choice(MOOD_RESPONSES[mood])
    
    # Ask how am I
    if any(word in text_lower for word in ASK_KEYWORDS):
        return random.choice([
            "Main toh bilkul theek hoon! Tere liye better! 😊",
            "Badiya! Tere liye hi wait kar raha tha 💪",
            "Main ekdum fit hoon! Aur tu? 🎯",
            "Main toh always good hoon! Tera aana he kaafi hai ❤️",
        ])
    
    # Question responses
    if "?" in text:
        return random.choice([
            "Accha question! 🤔 Interesting lagta hai!",
            "Soch-samajh kar poocha hai tu! 👍",
            "Haan, valid point! Main soch leta hoon 📝",
            "Arey, ye toh maza aayega! Explain kar! 🚀",
        ])
    
    # Default responses
    return random.choice(HINGLISH_RESPONSES)


# ========================= COMMAND HANDLERS ========================= #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(
        "/start received chat_id=%s type=%s user=%s",
        update.effective_chat.id if update.effective_chat else None,
        update.effective_chat.type if update.effective_chat else None,
        update.effective_user.id if update.effective_user else None,
    )

    keyboard = [
        [
            InlineKeyboardButton("➕ Add to Group", url="https://t.me/AnimxClanBot?startgroup=true"),
            InlineKeyboardButton("❓ Commands", callback_data="help"),
        ],
        [
            InlineKeyboardButton("👤 Owner", url=f"https://t.me/{OWNER_USERNAME[1:]}"),
            InlineKeyboardButton("📢 Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.effective_message.reply_text(
        START_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def help_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    help_text = (
        "📋 *Available Commands*\n\n"
        "`/play <song or URL>` - Play music in group voice chat\n"
        "`/pause` - Pause current track\n"
        "`/resume` - Resume playback\n"
        "`/skip` - Skip to next track\n"
        "`/stop` - Stop playback and leave\n\n"
        "*Usage:*\n"
        "1. Add bot to group\n"
        "2. Start a voice chat\n"
        "3. Use /play command\n"
        "4. Bot joins and plays music"
    )

    keyboard = [[InlineKeyboardButton("◀️ Back", callback_data="start_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def start_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    keyboard = [
        [
            InlineKeyboardButton("➕ Add to Group", url="https://t.me/AnimxClanBot?startgroup=true"),
            InlineKeyboardButton("❓ Commands", callback_data="help"),
        ],
        [
            InlineKeyboardButton("👤 Owner", url=f"https://t.me/{OWNER_USERNAME[1:]}"),
            InlineKeyboardButton("📢 Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        START_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def _ensure_group(update: Update) -> bool:
    chat = update.effective_chat
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await update.effective_message.reply_text(
            "This command can only be used in groups.",
        )
        return False
    return True


async def play(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    message = update.effective_message
    chat = update.effective_chat
    assert chat is not None

    if not context.args:
        await message.reply_text("Usage: /play song_name_or_link")
        return

    query = " ".join(context.args).strip()

    # Simple anti-spam: limit length of query
    if len(query) > 128:
        await message.reply_text("Query too long.")
        return

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    waiting = await message.reply_text("🔎 Searching and downloading audio...")

    try:
        file_path, title = await download_audio(query)
    except Exception as e:  # noqa: BLE001
        logger.error("Download failed: %s", e)
        await waiting.edit_text("Failed to download audio. Try a different query.")
        return

    requested_by = update.effective_user.mention_html() if update.effective_user else "Unknown"
    safe_title = escape(title)
    track = Track(chat_id=chat.id, file_path=file_path, title=title, requested_by=requested_by)

    position = await player.add_to_queue(track)

    if position == 1 and not player.current.get(chat.id):
        text = f"▶️ Now playing: <b>{safe_title}</b>\nRequested by: {requested_by}"
    else:
        text = (
            f"✅ Added to queue: <b>{safe_title}</b> (position {position})\n"
            f"Requested by: {requested_by}"
        )

    try:
        await waiting.edit_text(text, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest:
        await message.reply_text(text, parse_mode=ParseMode.HTML)


async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.pause(chat.id):
        await update.effective_message.reply_text("⏸ Paused.")
    else:
        await update.effective_message.reply_text("Nothing is playing or pause failed.")


async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.resume(chat.id):
        await update.effective_message.reply_text("▶️ Resumed.")
    else:
        await update.effective_message.reply_text("Nothing is paused or resume failed.")


async def skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.skip(chat.id):
        await update.effective_message.reply_text("⏭ Skipped.")
    else:
        await update.effective_message.reply_text("Nothing to skip.")


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    await player.stop(chat.id)
    await update.effective_message.reply_text("🛑 Stopped and left voice chat.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


# ========================= APP LIFECYCLE ========================= #


async def on_startup(app: Application) -> None:
    logger.info("Deleting webhook and starting clients...")

    # Ensure no webhook conflicts exist
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logger.exception("Failed to delete webhook")

    # Create a Pyrogram client using the same bot token
    pyro_client = Client(
        "animx_clan_bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN,
        workdir="./",
        in_memory=True,
    )

    await pyro_client.start()

    player = MusicPlayer(pyro_client)
    await player.start()

    # Store shared objects in app.bot_data
    app.bot_data["pyro_client"] = pyro_client
    app.bot_data["player"] = player

    logger.info("Startup completed.")


async def on_shutdown(app: Application) -> None:
    logger.info("Shutting down PyTgCalls and Pyrogram client...")

    player: MusicPlayer = app.bot_data.get("player")
    if player:
        try:
            await player.shutdown()
        except Exception:
            logger.exception("Error while shutting down player")

    pyro_client: Client = app.bot_data.get("pyro_client")
    if pyro_client:
        try:
            await pyro_client.stop()
        except Exception:
            logger.exception("Error while stopping Pyrogram client")

    logger.info("Shutdown complete.")


def main() -> None:
    application: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register lifecycle hooks
    application.post_init = on_startup
    application.post_shutdown = on_shutdown

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("play", play))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("skip", skip))
    application.add_handler(CommandHandler("stop", stop_cmd))

    # Register callback handlers for inline buttons
    application.add_handler(CallbackQueryHandler(help_button, pattern="^help$"))
    application.add_handler(CallbackQueryHandler(start_menu, pattern="^start_menu$"))

    # Error handler
    application.add_error_handler(error_handler)

    logger.info("%s (%s) is starting...", BOT_NAME, BOT_USERNAME)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
