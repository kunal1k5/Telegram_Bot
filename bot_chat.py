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
    "Yo! Kya haal hai? Tera din kaisa hai?",
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
            InlineKeyboardButton("💬 Chat With Me", callback_data="chat"),
            InlineKeyboardButton("❓ Help", callback_data="help_btn"),
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


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help command"""
    keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.effective_message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def joke_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Random joke command"""
    jokes = [
        "Ek aadmi tha, doosra bhi tha! 😂",
        "Kya hota hai jab programmer ka computer crash ho jaye? Fir woh developer ban jaye! 💻",
        "Tera wifi password kya hai? 'Tera-aashirwad'! 📡",
        "Mujhe coding pasand hai, liking pasand hai... Lakhan pasand hai! 🤣",
        "Question: Tum code likhte ho ya code likha hua chalate ho? Answer: Haan! 😄",
    ]
    await update.effective_message.reply_text(
        f"{random.choice(jokes)} 🎉"
    )


async def help_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help button callback"""
    query = update.callback_query
    await query.answer()

    keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def chat_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chat button callback"""
    query = update.callback_query
    await query.answer("Chal! Chat shuru karte hain! 💬")

    await query.edit_message_text(
        "🎉 *Haan! Chal shuru karte hain!* 🎉\n\n"
        "Kuch bhi puoch, kisi ko bhi roast kar, ya fir apna din sunao! 😄\n\n"
        "Main yaha hoon tere liye! Bol na! 👂"
    )


async def start_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start menu callback"""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [
            InlineKeyboardButton("💬 Chat With Me", callback_data="chat"),
            InlineKeyboardButton("❓ Help", callback_data="help_btn"),
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages"""
    if not update.message or not update.message.text:
        return

    user_message = update.message.text
    user_name = update.effective_user.first_name or "Bhai"

    logger.info("Message from %s: %s", user_name, user_message)

    # Generate Hinglish reply
    reply = get_hinglish_reply(user_message)

    # Add casual mention
    if random.random() > 0.6:
        reply = f"{reply}\n\nKya scene hai, {user_name}? 😄"

    await update.message.reply_text(reply)


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages and mentions"""
    if not update.message or not update.message.text:
        return

    message_text = update.message.text.lower()
    bot_mentioned = False

    # Check if bot is mentioned
    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.is_bot:
            bot_mentioned = True

    if "@animxclanbot" in message_text or bot_mentioned:
        user_message = update.message.text
        user_name = update.effective_user.first_name or "Bhai"

        reply = get_hinglish_reply(user_message)
        reply = f"{user_name}, {reply} 😊"

        await update.message.reply_text(reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


# ========================= APP LIFECYCLE ========================= #


async def on_startup(app: Application) -> None:
    logger.info("Bot starting up... 🚀")

    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logger.exception("Failed to delete webhook")

    logger.info("Bot ready! Let's chat! 💬")


async def on_shutdown(app: Application) -> None:
    logger.info("Bot shutting down... 👋")


def main() -> None:
    application: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register lifecycle hooks
    application.post_init = on_startup
    application.post_shutdown = on_shutdown

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("joke", joke_cmd))

    # Register callback handlers for inline buttons
    application.add_handler(CallbackQueryHandler(help_button, pattern="^help_btn$"))
    application.add_handler(CallbackQueryHandler(chat_button, pattern="^chat$"))
    application.add_handler(CallbackQueryHandler(start_menu, pattern="^start_menu$"))

    # Register message handlers
    # Private chat messages
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            handle_message,
        )
    )

    # Group/Supergroup messages (mentions only)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            handle_group_message,
        )
    )

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
