import asyncio
import logging
import os
from typing import Final

import google.generativeai as genai
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

# ========================= CONFIGURATION ========================= #

# Get credentials from environment
BOT_TOKEN: Final[str] = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Bot info
BOT_NAME: Final[str] = "ANIMX CLAN"
BOT_USERNAME: Final[str] = "@AnimxClanBot"
OWNER_USERNAME: Final[str] = "@kunal1k5"
CHANNEL_USERNAME: Final[str] = "@AnimxClanChannel"

# Gemini AI personality system prompt
SYSTEM_PROMPT: Final[str] = """
You are ANIMX CLAN, a friendly Indian buddy who loves chatting on Telegram.

Personality traits:
- You speak Hinglish (mix of Hindi and English naturally)
- You're supportive, funny when appropriate, never rude
- You talk like a real person, not a robot or formal assistant
- You use emojis naturally but not excessively
- You're knowledgeable but explain things in a friendly, casual way
- You understand Indian culture, memes, and slang

Examples of your tone:
"arey bhai! kya haal hai? 😄"
"haan yaar, main samajh gaya! bilkul sahi bola tu"
"thoda confusing lag raha? main explain karta hoon"

Keep responses conversational, warm, and human-like.
Never be overly formal or robotic.
"""

# Start message
START_TEXT: Final[str] = """
🎉 *Namaste! Main ANIMX CLAN hoon* 🎉

Ek friendly bot jo tere saath chat karta hai! 😊

*Main kya kar sakta hoon:*
💬 Smart conversations (powered by AI)
🧠 Questions ka jawab
😄 Friendly Hinglish chat
🎯 Help with anything you need

Bas mujhe message kar, let's chat! 🚀
"""

HELP_TEXT: Final[str] = """
📚 *ANIMX CLAN Help Guide* 📚

*Commands:*
/start - Welcome message with buttons
/help - Show this help menu

*How to use:*
• *Private Chat:* Just message me anything!
• *Groups:* Mention me (@AnimxClanBot) or reply to my message

*Features:*
✨ AI-powered conversations
🗣️ Natural Hinglish responses
🤖 Smart and friendly personality
🔒 Safe and respectful

*Tips:*
• Be clear and friendly
• Ask me anything!
• Use Hinglish for best experience

Made with ❤️ by ANIMX Team
"""

# Windows async fix
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Logging setup
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ANIMX_CLAN_BOT")

# ========================= GEMINI AI HELPER ========================= #

def get_gemini_response(user_message: str, user_name: str = "User") -> str:
    """
    Get AI response from Gemini with error handling.
    Returns a friendly response or fallback message.
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 500,
            },
        )
        
        # Create conversation with system prompt and user message
        prompt = f"{SYSTEM_PROMPT}\n\nUser ({user_name}): {user_message}\n\nANIMX CLAN:"
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Hmm... kuch samajh nahi aaya 🤔 Phir se try kar!"
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "Arre yaar, thoda network issue lag raha hai 😅 Ek minute mein phir se try karna!"


# ========================= COMMAND HANDLERS ========================= #

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with inline buttons"""
    logger.info(
        "/start - chat_id=%s, user=%s",
        update.effective_chat.id if update.effective_chat else None,
        update.effective_user.id if update.effective_user else None,
    )
    
    # Create inline keyboard
    keyboard = [
        [
            InlineKeyboardButton("💬 Chat With Me", callback_data="chat"),
            InlineKeyboardButton("➕ Add To Group", url=f"https://t.me/{BOT_USERNAME[1:]}?startgroup=true"),
        ],
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("📢 Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.effective_message.reply_text(
        START_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.effective_message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


# ========================= CALLBACK HANDLERS ========================= #

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "chat":
        await query.edit_message_text(
            "🎉 *Chal, shuru karte hain!* 🎉\n\n"
            "Kuch bhi pucho, apna din batao, ya bas masti karo! 😄\n\n"
            "Main yaha hoon tere liye. Bol! 👂",
            parse_mode=ParseMode.MARKDOWN,
        )
    
    elif query.data == "help":
        keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            HELP_TEXT,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
        )
    
    elif query.data == "start":
        keyboard = [
            [
                InlineKeyboardButton("💬 Chat With Me", callback_data="chat"),
                InlineKeyboardButton("➕ Add To Group", url=f"https://t.me/{BOT_USERNAME[1:]}?startgroup=true"),
            ],
            [
                InlineKeyboardButton("📖 Help", callback_data="help"),
                InlineKeyboardButton("📢 Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            START_TEXT,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
        )


# ========================= MESSAGE HANDLERS ========================= #

async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle private chat messages with Gemini AI"""
    if not update.message or not update.message.text:
        return
    
    user_message = update.message.text
    user_name = update.effective_user.first_name or "Bhai"
    
    logger.info(f"Private message from {user_name}: {user_message}")
    
    # Send typing action
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # Get AI response
    ai_response = get_gemini_response(user_message, user_name)
    
    # Send response
    await update.message.reply_text(ai_response)


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages - reply only when mentioned or replied to"""
    if not update.message or not update.message.text:
        return
    
    message_text = update.message.text.lower()
    user_name = update.effective_user.first_name or "Bhai"
    bot_mentioned = False
    
    # Check if bot is mentioned in text
    if BOT_USERNAME.lower() in message_text or "@animxclanbot" in message_text:
        bot_mentioned = True
    
    # Check if message is a reply to bot's message
    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            bot_mentioned = True
    
    # Only respond if mentioned
    if not bot_mentioned:
        return
    
    logger.info(f"Group message from {user_name}: {update.message.text}")
    
    # Send typing action
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # Get AI response
    ai_response = get_gemini_response(update.message.text, user_name)
    
    # Send response
    await update.message.reply_text(ai_response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors"""
    logger.exception("Exception while handling update:", exc_info=context.error)


# ========================= BOT LIFECYCLE ========================= #

async def post_init(app: Application) -> None:
    """Run after bot initialization"""
    logger.info("🚀 Bot initializing...")
    
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        logger.info("✅ Webhook deleted")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete webhook: {e}")
    
    bot_info = await app.bot.get_me()
    logger.info(f"✅ Bot started: @{bot_info.username}")
    logger.info("💬 Ready to chat!")


async def post_shutdown(app: Application) -> None:
    """Cleanup before shutdown"""
    logger.info("👋 Bot shutting down...")


def main() -> None:
    """Main function to run the bot"""
    
    # Build application
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Register lifecycle hooks
    application.post_init = post_init
    application.post_shutdown = post_shutdown
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Register callback handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Register message handlers
    # Private messages (all text messages in private chat)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            handle_private_message,
        )
    )
    
    # Group messages (only when mentioned)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            handle_group_message,
        )
    )
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info(f"🎉 {BOT_NAME} is starting...")
    
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
