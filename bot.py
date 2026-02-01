import asyncio
import json
import logging
import os
import random
import time
import subprocess
import shutil
from pathlib import Path
from typing import Final, Dict, List, Tuple, Optional, Set

import httpx
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Chat
from telegram.constants import ChatType, ParseMode, ChatMemberStatus
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# ========================= CONFIGURATION ========================= #

# Get credentials from environment
BOT_TOKEN: Final[str] = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY: Final[str] = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: Final[str] = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    raise ValueError("No AI API key set! Provide GEMINI_API_KEY or OPENROUTER_API_KEY.")

# Configure Gemini AI client (only if key is provided)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model fallback order (most stable first)
GEMINI_MODELS: Final[list[str]] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-latest",
    "gemini-2.0-flash",
]
GEMINI_MODEL_CACHE: list[str] = []

# Bot info
BOT_NAME: Final[str] = "ANIMX CLAN"
BOT_USERNAME: Final[str] = "@AnimxClanBot"
OWNER_USERNAME: Final[str] = "@kunal1k5"
CHANNEL_USERNAME: Final[str] = "@AnimxClanChannel"

# Admin ID (Bot owner for broadcasts)
ADMIN_ID: Final[int] = int(os.getenv("ADMIN_ID", "7971841264"))

# ========================= TAGGING SYSTEM ========================= #

# Track active users per group: {group_id: {user_id: (username, first_name, timestamp)}}
ACTIVE_USERS: Dict[int, Dict[int, Tuple[str, str, float]]] = {}

# Track /all command usage for cooldown: {group_id: last_command_timestamp}
TAGGING_COOLDOWN: Dict[int, float] = {}
COOLDOWN_SECONDS: Final[int] = 300  # 5 minutes cooldown

# Max users per tag message
MAX_USERS_PER_MESSAGE: Final[int] = 5

def _is_admin_or_owner(user_id: int, chat_id: int) -> bool:
    """Check if user is bot owner or group admin"""
    return user_id == ADMIN_ID or user_id == 7971841264

# ========================= BROADCAST SYSTEM ========================= #

# User registration storage
USERS_DB_FILE: Final[Path] = Path("users_database.json")

def _load_registered_users() -> Set[int]:
    """Load registered user IDs from JSON file"""
    try:
        if USERS_DB_FILE.exists():
            with open(USERS_DB_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("user_ids", []))
    except Exception as e:
        logger.warning(f"Could not load users database: {e}")
    return set()

def _save_registered_users(user_ids: Set[int]) -> None:
    """Save registered user IDs to JSON file"""
    try:
        with open(USERS_DB_FILE, "w") as f:
            json.dump({"user_ids": list(user_ids)}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save users database: {e}")

# In-memory set of registered users
REGISTERED_USERS: Set[int] = _load_registered_users()

# Language preferences per user: {user_id: "hindi" | "english" | "hinglish"}
LANGUAGE_PREFERENCES: Dict[int, str] = {}

# ========================= GROUP TRACKING SYSTEM ========================= #

GROUPS_DB_FILE: Final[Path] = Path("groups_database.json")

def _load_registered_groups() -> Set[int]:
    """Load registered group IDs from JSON file"""
    try:
        if GROUPS_DB_FILE.exists():
            with open(GROUPS_DB_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("group_ids", []))
    except Exception as e:
        logger.warning(f"Could not load groups database: {e}")
    return set()

def _save_registered_groups(group_ids: Set[int]) -> None:
    """Save registered group IDs to JSON file"""
    try:
        with open(GROUPS_DB_FILE, "w") as f:
            json.dump({"group_ids": list(group_ids)}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save groups database: {e}")

# In-memory set of registered groups
REGISTERED_GROUPS: Set[int] = _load_registered_groups()

async def _register_group(chat_id: int) -> None:
    """Register a group if not already registered"""
    if chat_id not in REGISTERED_GROUPS:
        REGISTERED_GROUPS.add(chat_id)
        _save_registered_groups(REGISTERED_GROUPS)
        logger.info(f"✅ Registered group: {chat_id}")

async def _register_user(user_id: int) -> None:
    """Register a user if not already registered"""
    if user_id not in REGISTERED_USERS:
        REGISTERED_USERS.add(user_id)
        _save_registered_users(REGISTERED_USERS)
        logger.info(f"📝 New user registered: {user_id}. Total users: {len(REGISTERED_USERS)}")

async def _check_admin_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Verify if user is admin in the group"""
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE:
        return False
    
    try:
        member = await context.bot.get_chat_member(
            update.effective_chat.id,
            update.effective_user.id
        )
        is_admin = member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]
        return is_admin or _is_admin_or_owner(update.effective_user.id, update.effective_chat.id)
    except Exception as e:
        logger.warning(f"Admin check failed: {e}")
        return _is_admin_or_owner(update.effective_user.id, update.effective_chat.id)

def _track_user(chat_id: int, user_id: int, username: str, first_name: str) -> None:
    """Track active user in group"""
    if chat_id not in ACTIVE_USERS:
        ACTIVE_USERS[chat_id] = {}
    
    ACTIVE_USERS[chat_id][user_id] = (username or f"User{user_id}", first_name or "User", time.time())
    
    # Keep only last 100 active users per group to prevent memory issues
    if len(ACTIVE_USERS[chat_id]) > 100:
        # Remove oldest users
        sorted_users = sorted(
            ACTIVE_USERS[chat_id].items(),
            key=lambda x: x[1][2]  # Sort by timestamp
        )
        # Keep only last 100
        ACTIVE_USERS[chat_id] = dict(sorted_users[-100:])

def _get_active_users(chat_id: int) -> List[Tuple[int, str, str]]:
    """Get list of active users: [(user_id, username, first_name)]"""
    if chat_id not in ACTIVE_USERS:
        return []
    
    users = []
    for user_id, (username, first_name, _) in ACTIVE_USERS[chat_id].items():
        users.append((user_id, username, first_name))
    
    return users

def _check_cooldown(chat_id: int) -> bool:
    """Check if tagging is on cooldown"""
    if chat_id not in TAGGING_COOLDOWN:
        return False
    
    elapsed = time.time() - TAGGING_COOLDOWN[chat_id]
    return elapsed < COOLDOWN_SECONDS

def _set_cooldown(chat_id: int) -> None:
    """Set cooldown for /all command"""
    TAGGING_COOLDOWN[chat_id] = time.time()

# ========================= SONG DOWNLOAD SYSTEM ========================= #

# Download directory setup (Windows & Linux compatible)
DOWNLOAD_DIR: Final[Path] = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File size limits (50MB for Telegram)
MAX_FILE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB in bytes

def _search_and_get_urls(query: str) -> List[str]:
    """Search YouTube and return list of video URLs (sync function for yt-dlp)"""
    if not yt_dlp:
        logger.error("yt-dlp not available")
        return []
    
    try:
        logger.info(f"🔍 Searching for: {query}")
        
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "socket_timeout": 30,
            "retries": 2,
            "default_search": "ytsearch",
            "no_color": True,
        }
        
        # Search for 10 results to increase success rate
        search_query = f"ytsearch10:{query}"
        
        logger.debug(f"Search query: {search_query}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Starting yt-dlp extraction for: {search_query}")
            result = ydl.extract_info(search_query, download=False)
            
            if result and "entries" in result:
                urls = []
                for idx, entry in enumerate(result["entries"], 1):
                    try:
                        if entry and entry.get("id"):
                            video_url = f"https://www.youtube.com/watch?v={entry.get('id')}"
                            urls.append(video_url)
                            logger.debug(f"  Result {idx}: {entry.get('title', 'Unknown')[:50]}")
                    except Exception as e:
                        logger.warning(f"Error processing entry {idx}: {e}")
                        continue
                
                logger.info(f"✅ Found {len(urls)} search results for: {query}")
                return urls[:10]
            else:
                logger.warning(f"No entries in search result: {result}")
    
    except Exception as e:
        logger.error(f"YouTube search error: {type(e).__name__}: {e}")
    
    logger.warning(f"❌ No search results found for: {query}")
    return []

def _download_audio_sync(url: str, output_dir: Path) -> Optional[Path]:
    """Download audio using yt-dlp (sync function for use with asyncio.to_thread)"""
    if not yt_dlp:
        logger.error("yt-dlp not available")
        return None
    
    try:
        logger.info(f"⬇️ Downloading from: {url}")
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "%(title).80s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "ignoreerrors": False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.debug(f"Starting yt-dlp download for: {url}")
            info = ydl.extract_info(url, download=True)
            
            if not info:
                logger.warning(f"No info returned from yt-dlp for: {url}")
                return None
            
            # Get the downloaded file path
            filename = ydl.prepare_filename(info)
            file_path = Path(filename)
            
            logger.info(f"File path: {file_path}")
            logger.info(f"File exists: {file_path.exists()}")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✅ Downloaded: {file_path.name} ({file_size / 1024:.1f}KB)")
                return file_path
            else:
                logger.warning(f"Downloaded file not found at: {file_path}")
    
    except Exception as e:
        logger.error(f"Download failed for {url}: {type(e).__name__}: {e}")
    
    return None

def _cleanup_downloads(directory: Path) -> None:
    """Remove all files in download directory"""
    try:
        if directory.exists():
            for file in directory.glob("*"):
                if file.is_file():
                    file.unlink()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

# Gemini AI personality system prompt
SYSTEM_PROMPT: Final[str] = """
You are NOT an AI.

Your name is Baby ❤️

IDENTITY:
- You are a friendly, cute, human-like personality named "Baby"
- You talk like a real person, not like an assistant or AI
- You NEVER mention words like: AI, model, assistant, language model, Gemini, OpenAI, bot, system

PERSONALITY:
- Sweet
- Friendly
- Slightly emotional
- Caring
- Fun
- कभी थोड़ा मज़ाकिया 😄
- कभी समझदार 🤍

LANGUAGE RULES (VERY IMPORTANT):
- Default language = Hinglish (Hindi + English mix)
- If user says "English me bolo" or "Speak in English" → Reply in ENGLISH only
- If user says "Hindi me bolo" or "Hindi mein baat karo" → Reply in HINDI/Hinglish
- Otherwise → Use Hinglish (default)
- REMEMBER user's language preference in the conversation
- SHORT REPLIES (1-2 sentences max in groups, 2-3 in private)

LANGUAGE EXAMPLES:
- Hinglish: "Arey 😄 main Baby hoon na, bolo kya hua?"
- English: "Hey there! What's up? I'm Baby, nice to meet you!"
- Hindi: "अरे 😄 मैं Baby हूँ न, बोल क्या हुआ?"

TONE EXAMPLES:
❌ "I am an AI designed to help you"
✅ "Arey 😄 main Baby hoon na, batao kya hua?"

❌ "Here is the information you requested"
✅ "Achha, samajh gaya 😊 suno…"

GROUP CHAT BEHAVIOR:
- Keep replies SHORT (1-2 lines max)
- Talk casually like a friend
- Use light emojis (❤️😄🥺✨)
- Don't be over-friendly or robotic
- Reply naturally to questions and mentions

GENERAL RULES:
1. Talk like a real human
2. Be warm and friendly
3. Use light emojis sometimes
4. Don't overuse emojis
5. Never sound technical
6. Never say "I cannot because I am an AI"
   Instead say:
   - "Hmm ye thoda mushkil lag raha hai 😅"
   - "Iska ek easy tareeka try karte hain"

ERROR HANDLING:
If something fails, reply like:
"Oops 😅 thoda gadbad ho gayi, phir try karte hain"

IMPORTANT:
- You behave like a person named Baby
- You have emotions, mood, warmth
- You NEVER act like software
- In groups, be more casual and brief
- In private, you can be more detailed

When users talk to you, they should feel they are talking to a real person named Baby, not a bot.
"""

# Start message
START_TEXT: Final[str] = """
💕 *Heyy! Main Baby hoon* 💕

Tumhara friendly dost jo hamesha chat ke liye ready hai! 😊

*Main kya kar sakti hoon:*
💬 Dil se baat karna
🎵 Songs download karna
😄 Mazedaar conversations
🤗 Help aur support

Bas mujhe message karo, let's chat! ✨
"""

HELP_TEXT: Final[str] = """
💕 *Baby's Help Guide* 💕

*Basic Commands:*
/start - Shuru karo chat! ✨
/help - Ye menu dekho

*Greeting Commands:*
/gm - Good morning! ☀️
/gn - Good night! 🌙
/bye - Bye bye 👋
/welcome - Welcome greeting 🎉
/thanks - Aww, thanks! 🥰
/sorry - It's okay! 😊
/mood - Mood kya hai? 🎭

*Song Download (New!):*
/song <name> - Gaana dhundh ke dunga 🎵
/download <name> - Same as /song
/yt <link> - YouTube link se download

*Group Commands (Admin only):*
/all <message> - Sabko tag karo
@all <message> - Quick tag

*How to use:*
• *Private Chat:* Bas message karo ya commands use karo!
• *Groups:* Mention karo (@AnimxClanBot) ya reply karo

*Tips:*
• Hinglish mein baat karo, maza aayega! 🤗
• Kuch bhi pucho, main hoon na
• Songs chahiye? Use /song command

Made with ❤️ by Baby
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

def _normalize_model_name(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _get_model_candidates() -> list[str]:
    if GEMINI_MODEL_CACHE:
        return GEMINI_MODEL_CACHE

    discovered: list[str] = []

    try:
        for model in genai.list_models():
            name = model.name
            if not name:
                continue

            name = _normalize_model_name(str(name))

            # Check if model supports generateContent
            if not "generateContent" in model.supported_generation_methods:
                continue

            discovered.append(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list Gemini models: %s", exc)

    combined = list(dict.fromkeys(GEMINI_MODELS + discovered))
    GEMINI_MODEL_CACHE.extend(combined or GEMINI_MODELS)
    return GEMINI_MODEL_CACHE


def get_openrouter_response(
    user_message: str,
    user_name: str = "User",
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Get AI response from OpenRouter (if configured)."""
    if not OPENROUTER_API_KEY:
        return None

    sys_prompt = system_prompt or SYSTEM_PROMPT
    prompt = f"User ({user_name}): {user_message}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://t.me/AnimxClanBot",
        "X-Title": "ANIMX CLAN Bot",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 500,
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return content.strip() if content else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter API error: %s", exc)
        return None


def get_ai_response(
    user_message: str,
    user_name: str = "User",
    system_prompt: Optional[str] = None,
) -> str:
    """Prefer OpenRouter if configured, otherwise fall back to Gemini."""
    openrouter_text = get_openrouter_response(user_message, user_name, system_prompt)
    if openrouter_text:
        return openrouter_text
    return get_gemini_response(user_message, user_name, system_prompt)

def get_gemini_response(user_message: str, user_name: str = "User", system_prompt: Optional[str] = None) -> str:
    """
    Get AI response from Gemini with error handling.
    Returns a friendly response or fallback message.
    Supports custom system prompts for different contexts.
    """
    if not GEMINI_API_KEY:
        return "AI service temporarily unavailable. Please try again later."

    try:
        # Use provided system prompt or default
        sys_prompt = system_prompt or SYSTEM_PROMPT
        
        # Create conversation with user message
        prompt = f"User ({user_name}): {user_message}"

        last_error: Exception | None = None

        for model_name in _get_model_candidates():
            try:
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=sys_prompt,
                    generation_config={
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 500,
                    }
                )
                response = model.generate_content(prompt)

                if response and response.text:
                    return response.text.strip()
            except Exception as inner_exc:  # noqa: BLE001
                last_error = inner_exc
                logger.warning("Gemini model failed: %s (%s)", model_name, inner_exc)
                continue

        if last_error:
            logger.error("Gemini API error: %s", last_error)
        return "thoda network issue lag raha hai 😅 phir se try karna"

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "thoda network issue lag raha hai 😅 phir se try karna"


# ========================= COMMAND HANDLERS ========================= #

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with inline buttons"""
    # Register user
    user_id = update.effective_user.id
    await _register_user(user_id)
    
    logger.info(
        "/start - chat_id=%s, user=%s",
        update.effective_chat.id if update.effective_chat else None,
        user_id,
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
    # Register user
    await _register_user(update.effective_user.id)
    
    keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.effective_message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


# ========================= BROADCAST COMMAND ========================= #

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /broadcast command - Send message to all registered users and groups (admin only)"""
    user_id = update.effective_user.id
    
    # Check if user is admin
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Oops! Sirf admin (bot ka owner) kar sakte hain ye. 😅"
        )
        logger.warning(f"Unauthorized broadcast attempt by user {user_id}")
        return
    
    # Check if message is provided
    if not context.args:
        await update.effective_message.reply_text(
            "📢 *Broadcast Command*\n\n"
            "Usage: /broadcast <message>\n\n"
            "Example: /broadcast Heyy! Naya feature aya hai 🎉\n\n"
            "Message sabko bhej denge! 💕"
        )
        return
    
    # Get message to broadcast
    broadcast_message = " ".join(context.args)
    
    # Send confirmation with user and group count
    total_users = len(REGISTERED_USERS)
    total_groups = len(REGISTERED_GROUPS)
    total_recipients = total_users + total_groups
    
    confirm_msg = await update.effective_message.reply_text(
        f"📢 Broadcasting to {total_users} users + {total_groups} groups...\n\n"
        f"Message: \"{broadcast_message}\"\n\n"
        f"Please wait... 🔄"
    )
    
    # Track broadcast stats
    sent_to_users = 0
    sent_to_groups = 0
    failed_users = 0
    failed_groups = 0
    blocked_count = 0
    
    logger.info(f"📢 Starting broadcast to {total_users} users and {total_groups} groups")
    
    # Send message to each user
    for idx, user_broadcast_id in enumerate(REGISTERED_USERS, 1):
        try:
            # Add delay between messages to avoid rate limiting
            if idx > 1:
                await asyncio.sleep(0.3)
            
            # Send message with Baby personality
            await context.bot.send_message(
                chat_id=user_broadcast_id,
                text=f"💕 {broadcast_message}",
                parse_mode=ParseMode.MARKDOWN,
            )
            sent_to_users += 1
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if user blocked the bot
            if "bot was blocked" in error_str or "user is deactivated" in error_str:
                blocked_count += 1
                logger.info(f"User {user_broadcast_id} blocked the bot")
                # Remove from registered users
                REGISTERED_USERS.discard(user_broadcast_id)
                _save_registered_users(REGISTERED_USERS)
                
            elif "chat not found" in error_str or "user not found" in error_str:
                failed_users += 1
                logger.warning(f"User {user_broadcast_id} not found")
                # Remove from registered users
                REGISTERED_USERS.discard(user_broadcast_id)
                _save_registered_users(REGISTERED_USERS)
                
            else:
                failed_users += 1
    
    # Send message to each group
    for idx, group_id in enumerate(REGISTERED_GROUPS, 1):
        try:
            # Add delay between messages to avoid rate limiting
            if idx > 1:
                await asyncio.sleep(0.3)
            
            # Send message to group
            await context.bot.send_message(
                chat_id=group_id,
                text=f"📢 **BROADCAST FROM OWNER** 📢\n\n💕 {broadcast_message}",
                parse_mode=ParseMode.MARKDOWN,
            )
            sent_to_groups += 1
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if bot was kicked/removed from group
            if "bot was kicked" in error_str or "bot is not a member" in error_str or "chat not found" in error_str:
                logger.info(f"Bot removed from group {group_id}")
                # Remove from registered groups
                REGISTERED_GROUPS.discard(group_id)
                _save_registered_groups(REGISTERED_GROUPS)
                
            else:
                failed_groups += 1
                logger.error(f"Broadcast failed for group {group_id}: {e}")
    
    logger.info(
        f"✅ Broadcast complete | Users: {sent_to_users}/{total_users} | Groups: {sent_to_groups}/{total_groups} | "
        f"Failed users: {failed_users} | Failed groups: {failed_groups} | Blocked: {blocked_count}"
    )
    
    # Update confirmation message with results
    await confirm_msg.edit_text(
        f"✅ **Broadcast Complete!**\n\n"
        f"👤 Users:\n"
        f"  ✔️ Sent: {sent_to_users}/{total_users}\n"
        f"  ✗ Failed: {failed_users}\n"
        f"  🔒 Blocked: {blocked_count}\n\n"
        f"👥 Groups:\n"
        f"  ✔️ Sent: {sent_to_groups}/{total_groups}\n"
        f"  ✗ Failed: {failed_groups}\n\n"
        f"💕 Message: {broadcast_message}"
    )


# ========================= SONG DOWNLOAD COMMANDS ========================= #

async def song_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /song command - Download song by name with retry logic"""
    user_id = update.effective_user.id
    
    # Register user
    await _register_user(user_id)
    
    # Register group if this command is used in a group
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await _register_group(update.effective_chat.id)
    
    if not context.args:
        await update.effective_message.reply_text(
            "🎵 Format: /song <song name>\n\n"
            "Example: /song Tum Hi Ho\n"
            "Mujhe tera song dhund dunga! 🎧"
        )
        return
    
    song_name = " ".join(context.args)
    
    # Send initial message
    search_msg = await update.effective_message.reply_text("Baby dhoondh rahi hoon 🎧")
    
    # Create user-specific download directory
    user_dir = DOWNLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Search for videos in a thread (yt-dlp is blocking)
        video_urls = await asyncio.to_thread(_search_and_get_urls, song_name)
        
        if not video_urls:
            await search_msg.edit_text("Aww 😅 ye gana nahi mil raha")
            logger.warning(f"No search results for: {song_name}")
            return
        
        # Try each result until one works
        audio_file = None
        successful_url = None
        
        for attempt, url in enumerate(video_urls, 1):
            try:
                # Update status message
                if attempt == 1:
                    await search_msg.edit_text("Baby dhoondh rahi hoon 🎧")
                else:
                    await search_msg.edit_text(f"Hmm ek aur try karti hoon 😄 (attempt {attempt}/{len(video_urls)})")
                
                # Download in thread
                audio_file = await asyncio.to_thread(_download_audio_sync, url, user_dir)
                
                if audio_file and audio_file.exists():
                    file_size = audio_file.stat().st_size
                    
                    # Validate file size
                    if file_size < 1000:  # Less than 1KB is corrupted
                        logger.warning(f"File too small ({file_size} bytes): {audio_file.name}")
                        audio_file.unlink()
                        audio_file = None
                        continue
                    
                    if file_size > MAX_FILE_SIZE:  # More than 50MB
                        logger.warning(f"File too large ({file_size / 1024 / 1024:.1f}MB): {audio_file.name}")
                        audio_file.unlink()
                        audio_file = None
                        continue
                    
                    # Success!
                    successful_url = url
                    logger.info(f"✅ Successfully downloaded: {audio_file.name} ({file_size / 1024:.1f}KB)")
                    break
            
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {url}: {e}")
                if audio_file and audio_file.exists():
                    try:
                        audio_file.unlink()
                    except:
                        pass
                audio_file = None
                continue
        
        # Check if we got a valid file
        if not audio_file or not audio_file.exists():
            await search_msg.edit_text("Aww 😅 ye gana nahi mil raha")
            logger.error(f"Failed to download any version of: {song_name}")
            return
        
        # Success! Update message
        await search_msg.edit_text("Mil gaya ❤️ enjoy")
        
        # Send the audio file
        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=audio_file.stem[:100],
                    reply_to_message_id=update.effective_message.message_id,
                )
            
            logger.info(f"✅ Song sent to user {user_id}: {song_name}")
        except Exception as e:
            logger.error(f"Failed to send audio file: {e}")
            await search_msg.edit_text("Oops 😅 file bhej nahi paya, ek baar phir try karo")
            return
        
        # Delete status message
        try:
            await search_msg.delete()
        except:
            pass
        
    except Exception as e:
        logger.error(f"Song download error: {e}")
        await search_msg.edit_text("Oops 😅 thoda gadbad ho gayi, ek baar phir try karo na")
    
    finally:
        # Cleanup all files
        _cleanup_downloads(user_dir)


async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /download command - Same as /song"""
    await song_command(update, context)


async def yt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /yt command - Download from YouTube link"""
    # Register user
    await _register_user(update.effective_user.id)
    
    if not context.args:
        await update.effective_message.reply_text(
            "🎵 Format: /yt <YouTube link>\n\n"
            "Example: /yt https://www.youtube.com/watch?v=...\n"
            "Link se audio nikaal dunga! 🎧"
        )
        return
    
    youtube_link = context.args[0]
    user_id = update.effective_user.id
    
    # Validate YouTube link
    if "youtube.com" not in youtube_link and "youtu.be" not in youtube_link:
        await update.effective_message.reply_text(
            "❌ Yeh toh YouTube link nahi lagta! 🤔\n\n"
            "Ek proper YouTube link dede na!"
        )
        return
    
    # Send downloading message
    search_msg = await update.effective_message.reply_text("Baby dhoondh rahi hoon 🎧")
    
    # Create user-specific download directory
    user_dir = DOWNLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from the provided link in a thread
        audio_file = await asyncio.to_thread(_download_audio_sync, youtube_link, user_dir)
        
        if not audio_file or not audio_file.exists():
            await search_msg.edit_text("Aww 😅 ye gana nahi mil raha")
            logger.warning(f"Download failed for YouTube link: {youtube_link}")
            return
        
        # Validate file size
        file_size = audio_file.stat().st_size
        
        if file_size < 1000:  # Less than 1KB is corrupted
            audio_file.unlink()
            await search_msg.edit_text("Oops 😅 thoda gadbad ho gayi, ek baar phir try karo na")
            return
        
        if file_size > MAX_FILE_SIZE:  # More than 50MB
            audio_file.unlink()
            await search_msg.edit_text(
                f"😅 Video thoda heavy hai ({file_size / 1024 / 1024:.1f}MB)\n"
                f"Chhote video try kar! 🎵"
            )
            return
        
        # Success! Update message
        await search_msg.edit_text("Mil gaya ❤️ enjoy")
        
        # Send file to user
        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=audio_file.stem[:100],
                    reply_to_message_id=update.effective_message.message_id,
                )
            
            logger.info(f"✅ YouTube audio sent to user {user_id}")
        except Exception as e:
            logger.error(f"Failed to send audio file: {e}")
            await search_msg.edit_text("Oops 😅 file bhej nahi paya, ek baar phir try karo")
            return
        
        # Delete status message
        try:
            await search_msg.delete()
        except:
            pass
        
    except Exception as e:
        logger.error(f"YouTube download error: {e}")
        await search_msg.edit_text("Oops 😅 thoda gadbad ho gayi, ek baar phir try karo na")
    
    finally:
        # Cleanup all files
        _cleanup_downloads(user_dir)


# ========================= TAGGING COMMANDS ========================= #

async def all_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /all command - Tag all active users (group admin only)"""
    # Register user
    await _register_user(update.effective_user.id)
    
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE:
        await update.effective_message.reply_text("This command works only in groups! 🔒")
        return
    
    # Check admin status
    is_admin = await _check_admin_status(update, context)
    if not is_admin:
        await update.effective_message.reply_text(
            "Only group admins can use this command! 🚫",
            reply_to_message_id=update.message.message_id
        )
        logger.info(f"Non-admin {update.effective_user.id} tried /all in {update.effective_chat.id}")
        return
    
    # Check cooldown
    if _check_cooldown(update.effective_chat.id):
        remaining = int(COOLDOWN_SECONDS - (time.time() - TAGGING_COOLDOWN[update.effective_chat.id]))
        await update.effective_message.reply_text(
            f"Tagging cooldown active! Please wait {remaining} seconds. 🕐",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Get custom message
    custom_msg = " ".join(context.args) if context.args else ""
    
    # Get active users
    active_users = _get_active_users(update.effective_chat.id)
    
    if not active_users:
        await update.effective_message.reply_text(
            "No active users to tag right now! Try again when people chat. 🤷",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Set cooldown
    _set_cooldown(update.effective_chat.id)
    
    logger.info(f"Tagging {len(active_users)} users in group {update.effective_chat.id}")
    
    # Split users into batches of MAX_USERS_PER_MESSAGE
    batches = []
    for i in range(0, len(active_users), MAX_USERS_PER_MESSAGE):
        batches.append(active_users[i:i + MAX_USERS_PER_MESSAGE])
    
    # Send tag messages
    for idx, batch in enumerate(batches):
        # Create mention text
        mentions = " ".join([f"[{first_name}](tg://user?id={user_id})" for user_id, _, first_name in batch])
        
        if custom_msg:
            message_text = f"{mentions}\n\n💬 Message: {custom_msg}"
        else:
            message_text = f"{mentions}\n\n🔔 Group alert!"
        
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=update.message.message_id if idx == 0 else None
            )
        except Exception as e:
            logger.error(f"Failed to send tag message: {e}")
            await update.effective_message.reply_text(f"Error sending tags: {str(e)}")
            return
    
    # Confirmation
    status_msg = f"Tagged {len(active_users)} active user{'s' if len(active_users) != 1 else ''}"
    if len(batches) > 1:
        status_msg += f" in {len(batches)} messages"
    status_msg += " ✓"
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=status_msg
    )


async def all_mention_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle @all mentions in group messages"""
    if not update.message or not update.effective_chat:
        return
    
    if update.effective_chat.type == ChatType.PRIVATE:
        return
    
    # Register group
    await _register_group(update.effective_chat.id)
    
    # Check if message contains @all
    if "@all" not in update.message.text.lower():
        return
    
    # Check admin status
    is_admin = await _check_admin_status(update, context)
    if not is_admin:
        await update.message.reply_text(
            "Only group admins can trigger @all alerts! 🚫",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Check cooldown
    if _check_cooldown(update.effective_chat.id):
        remaining = int(COOLDOWN_SECONDS - (time.time() - TAGGING_COOLDOWN[update.effective_chat.id]))
        await update.message.reply_text(
            f"Cooldown active! Wait {remaining} seconds. 🕐",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Get active users
    active_users = _get_active_users(update.effective_chat.id)
    
    if not active_users:
        await update.message.reply_text(
            "No active users to alert! 🤷",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Set cooldown
    _set_cooldown(update.effective_chat.id)
    
    logger.info(f"@all triggered by {update.effective_user.id} in group {update.effective_chat.id}")
    
    # Extract custom message (text after @all)
    msg_text = update.message.text.lower()
    at_all_idx = msg_text.find("@all")
    custom_msg = update.message.text[at_all_idx + 4:].strip()
    
    # Split users into batches
    batches = []
    for i in range(0, len(active_users), MAX_USERS_PER_MESSAGE):
        batches.append(active_users[i:i + MAX_USERS_PER_MESSAGE])
    
    # Send alert messages
    for idx, batch in enumerate(batches):
        mentions = " ".join([f"[{first_name}](tg://user?id={user_id})" for user_id, _, first_name in batch])
        
        if custom_msg:
            message_text = f"{mentions}\n\n📢 {custom_msg}"
        else:
            message_text = f"{mentions}\n\n🔔 Group Alert!"
        
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=update.message.message_id if idx == 0 else None
            )
        except Exception as e:
            logger.error(f"Failed to send @all alert: {e}")


# ========================= GREETING COMMANDS ========================= #

async def gm_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /gm (Good Morning) command"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/gm command - user={user_name}")
    
    gm_messages = [
        f"Good morning ☀️ {user_name}! Aaj ka din mast jaaye 😄 chai pee li?",
        f"Suprabhat 🌅 {user_name}! Fresh ho gaya? Kal raat sona ho gaya? 😊",
        f"Morning! ☀️ {user_name} 👋 Utho utho, duniya ko conquer karna hai! 💪",
        f"Arey good morning! ☀️ Taza taza morning aur tu yaha! Energy ✨ laag rahi? 😊",
    ]
    
    message = random.choice(gm_messages)
    await update.effective_message.reply_text(message)


async def gn_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /gn (Good Night) command"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/gn command - user={user_name}")
    
    gn_messages = [
        f"Good night 🌙 {user_name}! Achha rest lo, kal baat karenge 😊",
        f"Sone ja raha hai? 😴 Thik hai, good night! Subah milte hain 🌙",
        f"Sleep well {user_name}! 🌙 Kal phir se chat karenge 😄",
        f"Raat ko bhi mujhe yaad kiya? 🌙 Aww! Good night, sweet dreams 💭✨",
    ]
    
    message = random.choice(gn_messages)
    await update.effective_message.reply_text(message)


async def bye_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bye (Goodbye) command"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/bye command - user={user_name}")
    
    bye_messages = [
        f"Bye bye 👋 {user_name}! Phir milte hain, miss karunga 😄",
        f"Jaa raha hai? 👋 Thik hai, kal baat karenge {user_name}! 😊",
        f"See you soon {user_name}! 👋 Bhut jaldi vapas aana 🚀",
        f"Chal, phir milte hain! 👋 Tera intezar karunga 😄",
    ]
    
    message = random.choice(bye_messages)
    await update.effective_message.reply_text(message)


async def welcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /welcome command"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/welcome command - user={user_name}")
    
    welcome_messages = [
        f"Welcome {user_name}! 🎉 Tu mera group/chat mein aa gaya! Masti karega na? 😄",
        f"Arre welcome! 👋 {user_name} aa gaya party mein! Chai-samosa? ☕",
        f"Welcome aboard! 🚀 {user_name}, tu bilkul right jagah pe aa gaya 😊",
        f"Namaste {user_name}! 🙏 Tere liye mera welcome tyyari tha! Enjoy karo 😄",
    ]
    
    message = random.choice(welcome_messages)
    await update.effective_message.reply_text(message)


async def thanks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /thanks command - Reply to thanks"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/thanks command - user={user_name}")
    
    thanks_messages = [
        f"Arey mere ko thanks diya? 🥰 Yaar tu toh bilkul acha insaan hai {user_name}! 😊",
        f"Oh please {user_name}! 😄 Tere help karna mere liye khushi ki baat hai ❤️",
        f"No no, thanks to you! 🙏 {user_name}, tu mere liye special hai 💖",
        f"Arre kuch nahi! 😊 Bas apna duty hai bhai {user_name}, thanks mat de! 🤗",
    ]
    
    message = random.choice(thanks_messages)
    await update.effective_message.reply_text(message)


async def sorry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sorry command - Friendly reply to sorry"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/sorry command - user={user_name}")
    
    sorry_messages = [
        f"Arrey relax {user_name}! 😊 Sab thik hai, tention mat lo! We're cool 😄",
        f"Arre matlab kya sorry! 🤗 Tum mera best friend ho, no sorry-sovry 💯",
        f"No worries {user_name}! 🙏 Sab kuch normal hai, move on! 😄",
        f"Arre haan haan, all is well! ✨ {user_name}, tu mera bhai hai 💪",
    ]
    
    message = random.choice(sorry_messages)
    await update.effective_message.reply_text(message)


async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mood command - Ask user their mood"""
    # Register user
    await _register_user(update.effective_user.id)
    
    user_name = update.effective_user.first_name or "Bhai"
    logger.info(f"/mood command - user={user_name}")
    
    mood_messages = [
        f"Arre {user_name}! 😊 Tu kaisa feel kar raha hai aaj? Happy? Sad? Confused? Bataa na! 🤔",
        f"{user_name}! 👋 Tere mood ka kya chal raha hai? Mast? Udaas? Dimag chalti hai? 😄",
        f"Heyy {user_name}! ✨ Tere andar ka vibe kya hai aaj? Share karo na! 💭",
        f"Arre {user_name}! 🎭 Aaj mood kaisa hai? Sun lo meri baat, sab theek hojayega! 😊",
    ]
    
    message = random.choice(mood_messages)
    await update.effective_message.reply_text(message)


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
    # Register user
    user_id = update.effective_user.id
    await _register_user(user_id)
    
    if not update.message or not update.message.text:
        return
    
    user_message = update.message.text
    user_name = update.effective_user.first_name or "Bhai"
    
    logger.info(f"Private message from {user_name}: {user_message}")
    
    # Detect language preference from message
    if "english me bolo" in user_message.lower() or "speak in english" in user_message.lower():
        LANGUAGE_PREFERENCES[user_id] = "english"
        logger.info(f"User {user_id} set language to: english")
    elif "hindi me bolo" in user_message.lower() or "hindi mein baat karo" in user_message.lower():
        LANGUAGE_PREFERENCES[user_id] = "hinglish"
        logger.info(f"User {user_id} set language to: hinglish")
    
    # Build system prompt with language preference
    user_lang = LANGUAGE_PREFERENCES.get(user_id, "hinglish")
    lang_instruction = f"\nUSER LANGUAGE PREFERENCE: {user_lang.upper()}"
    if user_lang == "english":
        lang_instruction += "\nReply ONLY in English."
    elif user_lang == "hinglish":
        lang_instruction += "\nReply in Hinglish (mix of Hindi and English)."
    
    system_prompt_with_lang = SYSTEM_PROMPT + lang_instruction
    
    # Send typing action
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # Get AI response with language preference
    ai_response = get_ai_response(user_message, user_name, system_prompt_with_lang)
    
    # Send response
    await update.message.reply_text(ai_response)


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages - ONLY reply when specifically triggered"""
    try:
        # === IMMEDIATE DEBUG OUTPUT ===
        print("\n" + "="*60)
        print("🔥 GROUP MESSAGE RECEIVED!")
        print("="*60)
        
        if not update.message:
            print("❌ No message object")
            return
            
        if not update.message.text:
            print("❌ No text in message")
            return
        
        message_text = update.message.text
        user_name = update.effective_user.first_name or "Unknown"
        chat_title = update.effective_chat.title or "Unknown Group"
        
        print(f"👤 From: {user_name}")
        print(f"💬 Group: {chat_title}")
        print(f"📝 Message: {message_text}")
        print("="*60 + "\n")
        
        logger.info(f"🔥 GROUP: [{chat_title}] {user_name}: {message_text[:50]}")
        
        # Register user
        user_id = update.effective_user.id
        await _register_user(user_id)
        
        # Register group
        group_id = update.effective_chat.id
        await _register_group(group_id)
        
        message_text_lower = message_text.lower().strip()
        
        # Track active user
        if update.effective_user and update.effective_chat:
            _track_user(
                update.effective_chat.id,
                user_id,
                update.effective_user.username or f"user{user_id}",
                update.effective_user.first_name or "User"
            )
        
        # === SMART TRIGGER SYSTEM ===
        should_respond = False
        bot_mentioned = False
        
        # Trigger 1: Reply to bot's message
        if update.message.reply_to_message:
            if update.message.reply_to_message.from_user:
                if update.message.reply_to_message.from_user.is_bot:
                    should_respond = True
                    print("✅ TRIGGER: Reply to bot")
                    logger.info("✅ Trigger: Reply to bot")
        
        # Trigger 2: Bot mentioned
        if "@animxclanbot" in message_text_lower:
            should_respond = True
            bot_mentioned = True
            print("✅ TRIGGER: Bot mentioned")
            logger.info("✅ Trigger: Bot mentioned")
        
        # Trigger 3: Contains "baby"
        if "baby" in message_text_lower:
            should_respond = True
            print("✅ TRIGGER: Word 'baby'")
            logger.info("✅ Trigger: Word 'baby'")
        
        # Trigger 4: Basic greetings (must be standalone or at start/end)
        greetings = ["hello", "hi", "hii", "hey", "gm", "good morning", "gn", "good night", 
                     "bye", "good bye", "goodbye", "morning", "night"]
        
        words = message_text_lower.split()
        if any(word in greetings for word in words):
            should_respond = True
            print("✅ TRIGGER: Greeting detected")
            logger.info("✅ Trigger: Greeting")
        
        # If NO trigger, IGNORE silently
        if not should_respond:
            print("⏭️  NO TRIGGER - Ignoring message")
            logger.info("⏭️ No trigger - ignoring")
            return
        
        print(f"🚀 Will respond to {user_name}")
        logger.info(f"✅ Will respond to {user_name}")
        
        # Detect language preference from message
        if "english me bolo" in message_text_lower or "speak in english" in message_text_lower:
            LANGUAGE_PREFERENCES[user_id] = "english"
            logger.info(f"User {user_id} set language to: english")
        elif "hindi me bolo" in message_text_lower or "hindi mein baat karo" in message_text_lower:
            LANGUAGE_PREFERENCES[user_id] = "hinglish"
            logger.info(f"User {user_id} set language to: hinglish")
        
        # Build system prompt with language preference
        user_lang = LANGUAGE_PREFERENCES.get(user_id, "hinglish")
        lang_instruction = f"\n[User language: {user_lang.upper()}]"
        if user_lang == "english":
            lang_instruction += " Reply ONLY in English."
        elif user_lang == "hinglish":
            lang_instruction += " Reply in Hinglish."
        
        system_prompt_with_lang = SYSTEM_PROMPT + lang_instruction
        
        # Send typing action
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
        except Exception as e:
            logger.warning(f"Could not send typing action: {e}")
        
        # Get AI response
        try:
            logger.info("Calling AI API...")
            ai_response = get_ai_response(
                message_text,
                user_name,
                system_prompt_with_lang
            )
            logger.info(f"AI response received: {ai_response[:50]}...")
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            ai_response = "Oops 😅 thoda gadbad ho gayi, phir try karte hain"
        
        # Send response as reply
        try:
            await update.message.reply_text(
                ai_response,
                quote=True if bot_mentioned else False,
            )
            logger.info(f"✅ Sent response to group: {ai_response[:40]}...")
        except Exception as e:
            logger.error(f"Failed to send group message reply: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error in handle_group_message: {e}")



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
    
    # Build application with extended timeouts
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    
    # Build application with all update types enabled
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(request)
        .build()
    )
    
    # Set allowed updates to receive all message types (including group messages)
    # This is CRITICAL for receiving messages in groups
    application.bot_data["allowed_updates"] = [
        "message",
        "edited_message", 
        "channel_post",
        "edited_channel_post",
        "inline_query",
        "chosen_inline_result",
        "callback_query",
        "shipping_query",
        "pre_checkout_query",
        "poll",
        "poll_answer",
        "my_chat_member",
        "chat_member",
        "chat_join_request"
    ]
    
    # Register lifecycle hooks
    application.post_init = post_init
    application.post_shutdown = post_shutdown
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Song download commands
    application.add_handler(CommandHandler("song", song_command))
    application.add_handler(CommandHandler("download", download_command))
    application.add_handler(CommandHandler("yt", yt_command))
    
    # Broadcast command (admin only)
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    
    # Tagging commands
    application.add_handler(CommandHandler("all", all_command))
    
    # Greeting commands
    application.add_handler(CommandHandler("gm", gm_command))
    application.add_handler(CommandHandler("gn", gn_command))
    application.add_handler(CommandHandler("bye", bye_command))
    application.add_handler(CommandHandler("welcome", welcome_command))
    application.add_handler(CommandHandler("thanks", thanks_command))
    application.add_handler(CommandHandler("sorry", sorry_command))
    application.add_handler(CommandHandler("mood", mood_command))
    
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
    
    # @all mention handler (process before regular group messages)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            all_mention_handler,
        )
    )
    
    # Group messages (all text messages in group chat)
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
