import asyncio
import json
import logging
import os
import random
import time
import subprocess
import shutil
from pathlib import Path
from typing import Final, Dict, List, Tuple, Optional, Set, Any

# Initialize random seed for better randomization
random.seed()

import httpx
from google import genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Chat, ChatPermissions
from telegram.constants import ChatType, ParseMode, ChatMemberStatus
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    ChatMemberHandler,
    filters,
)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# ========================= CONFIGURATION ========================= #

# Get credentials from environment
BOT_TOKEN: Final[str] = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "AIzaSyCOa0Yf2QBH3Eb45-A5n-PFbdTHtRSeONM")
OPENROUTER_API_KEY: Final[str] = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL: Final[str] = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    raise ValueError("No AI API key set! Provide GEMINI_API_KEY or OPENROUTER_API_KEY.")

# Configure Gemini AI client (only if key is provided)
GEMINI_CLIENT: Optional[genai.Client] = None
if GEMINI_API_KEY:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

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

# Enhanced user database structure
USERS_DB_FILE: Final[Path] = Path("users_database.json")

def _load_users_database() -> Dict[int, Dict[str, Any]]:
    """Load detailed user database from JSON file
    Structure: {user_id: {username, first_name, last_seen, join_date}}
    """
    try:
        if USERS_DB_FILE.exists():
            with open(USERS_DB_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in data.get("users", {}).items()}
    except Exception as e:
        logger.warning(f"Could not load users database: {e}")
    return {}

def _save_users_database(users_data: Dict[int, Dict[str, Any]]) -> None:
    """Save detailed user database to JSON file"""
    try:
        with open(USERS_DB_FILE, "w") as f:
            json.dump({"users": users_data, "total": len(users_data)}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save users database: {e}")

# In-memory detailed user database
USERS_DATABASE: Dict[int, Dict[str, Any]] = _load_users_database()

# Legacy set for backward compatibility
REGISTERED_USERS: Set[int] = set(USERS_DATABASE.keys())

# ========================= OPT-OUT SYSTEM ========================= #

OPTED_OUT_DB_FILE: Final[Path] = Path("opted_out_users.json")

def _load_opted_out_users() -> Set[int]:
    """Load opted-out user IDs from JSON file"""
    try:
        if OPTED_OUT_DB_FILE.exists():
            with open(OPTED_OUT_DB_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("user_ids", []))
    except Exception as e:
        logger.warning(f"Could not load opted-out database: {e}")
    return set()

def _save_opted_out_users(user_ids: Set[int]) -> None:
    """Save opted-out user IDs to JSON file"""
    try:
        with open(OPTED_OUT_DB_FILE, "w") as f:
            json.dump({"user_ids": list(user_ids)}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save opted-out database: {e}")

# In-memory set of opted-out users
OPTED_OUT_USERS: Set[int] = _load_opted_out_users()

# Language preferences per user: {user_id: "hindi" | "english" | "hinglish"}
LANGUAGE_PREFERENCES: Dict[int, str] = {}

# ========================= GROUP TRACKING SYSTEM ========================= #

GROUPS_DB_FILE: Final[Path] = Path("groups_database.json")

def _load_groups_database() -> Dict[int, Dict[str, Any]]:
    """Load detailed group database from JSON file
    Structure: {group_id: {title, type, member_count, added_date, last_active, members: {}}}
    """
    try:
        if GROUPS_DB_FILE.exists():
            with open(GROUPS_DB_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in data.get("groups", {}).items()}
    except Exception as e:
        logger.warning(f"Could not load groups database: {e}")
    return {}

def _save_groups_database(groups_data: Dict[int, Dict[str, Any]]) -> None:
    """Save detailed group database to JSON file"""
    try:
        with open(GROUPS_DB_FILE, "w") as f:
            json.dump({"groups": groups_data, "total": len(groups_data)}, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save groups database: {e}")

# In-memory detailed group database
GROUPS_DATABASE: Dict[int, Dict[str, Any]] = _load_groups_database()

# Legacy set for backward compatibility
REGISTERED_GROUPS: Set[int] = set(GROUPS_DATABASE.keys())

async def _register_group(chat_id: int, chat: Optional[Chat] = None) -> None:
    """Register a group with detailed information"""
    current_time = time.time()
    
    if chat_id not in GROUPS_DATABASE:
        # New group
        GROUPS_DATABASE[chat_id] = {
            "title": chat.title if chat else "Unknown Group",
            "type": chat.type if chat else "group",
            "username": chat.username if chat and chat.username else None,
            "added_date": current_time,
            "last_active": current_time,
            "member_count": 0,
            "members": {}  # Store group members: {user_id: {username, first_name, last_seen, msg_count}}
        }
        REGISTERED_GROUPS.add(chat_id)
        logger.info(f"✅ New group registered: {GROUPS_DATABASE[chat_id]['title']} ({chat_id})")
    else:
        # Update existing group
        GROUPS_DATABASE[chat_id]["last_active"] = current_time
        if chat and chat.title:
            GROUPS_DATABASE[chat_id]["title"] = chat.title
        # Ensure members dict exists (for old groups)
        if "members" not in GROUPS_DATABASE[chat_id]:
            GROUPS_DATABASE[chat_id]["members"] = {}
    
    _save_groups_database(GROUPS_DATABASE)

async def _register_group_member(chat_id: int, user_id: int, username: Optional[str] = None, 
                                  first_name: Optional[str] = None) -> None:
    """Register a member in a group"""
    current_time = time.time()
    
    # Ensure group exists
    if chat_id not in GROUPS_DATABASE:
        await _register_group(chat_id)
    
    # Ensure members dict exists
    if "members" not in GROUPS_DATABASE[chat_id]:
        GROUPS_DATABASE[chat_id]["members"] = {}
    
    members = GROUPS_DATABASE[chat_id]["members"]
    
    # Convert user_id to string for JSON storage
    user_id_str = str(user_id)
    
    if user_id_str not in members:
        # New member
        members[user_id_str] = {
            "user_id": user_id,
            "username": username or "None",
            "first_name": first_name or "Unknown",
            "joined": current_time,
            "last_seen": current_time,
            "message_count": 1
        }
        # Update group member count
        GROUPS_DATABASE[chat_id]["member_count"] = len(members)
        logger.info(f"👤 New member in {GROUPS_DATABASE[chat_id].get('title', 'group')}: "
                   f"@{username or 'None'} ({first_name or 'Unknown'})")
    else:
        # Update existing member
        members[user_id_str]["last_seen"] = current_time
        members[user_id_str]["message_count"] = members[user_id_str].get("message_count", 0) + 1
        if username:
            members[user_id_str]["username"] = username
        if first_name:
            members[user_id_str]["first_name"] = first_name
    
    _save_groups_database(GROUPS_DATABASE)

async def _register_user(user_id: int, username: Optional[str] = None, first_name: Optional[str] = None) -> None:
    """Register a user with detailed information"""
    current_time = time.time()
    
    if user_id not in USERS_DATABASE:
        # New user
        USERS_DATABASE[user_id] = {
            "user_id": user_id,
            "username": username or "None",
            "first_name": first_name or "Unknown",
            "join_date": current_time,
            "last_seen": current_time,
            "message_count": 0
        }
        REGISTERED_USERS.add(user_id)
        logger.info(f"📝 New user registered: @{username or 'None'} ({first_name or 'Unknown'}) - ID: {user_id}")
        logger.info(f"📊 Total users: {len(USERS_DATABASE)}")
    else:
        # Update existing user
        USERS_DATABASE[user_id]["last_seen"] = current_time
        USERS_DATABASE[user_id]["message_count"] = USERS_DATABASE[user_id].get("message_count", 0) + 1
        if username:
            USERS_DATABASE[user_id]["username"] = username
        if first_name:
            USERS_DATABASE[user_id]["first_name"] = first_name
    
    _save_users_database(USERS_DATABASE)

async def _register_user_from_update(update: Update) -> None:
    """Helper to register user from Update object"""
    if update.effective_user:
        await _register_user(
            update.effective_user.id,
            update.effective_user.username,
            update.effective_user.first_name
        )

async def _register_group_from_update(update: Update) -> None:
    """Helper to register group from Update object"""
    if update.effective_chat and update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await _register_group(update.effective_chat.id, update.effective_chat)

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

# ========================= ANTI-SPAM SYSTEM ========================= #

# Track messages per user: {(chat_id, user_id): [(message_text, timestamp), ...]}
USER_MESSAGES: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}

# Track warned users: {(chat_id, user_id): timestamp}
WARNED_USERS: Dict[Tuple[int, int], float] = {}

# Spam detection patterns
SPAM_LINK_PATTERNS = [
    r't\.me/joinchat',
    r'telegram\.me/joinchat',
    r'bit\.ly',
    r'tinyurl\.com',
    r'shorturl\.at',
    r'cutt\.ly',
    r'gg\.gg',
]

def _is_spam_link(text: str) -> bool:
    """Check if message contains spam links"""
    import re
    text_lower = text.lower()
    for pattern in SPAM_LINK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def _count_emojis(text: str) -> int:
    """Count emoji characters in text"""
    emoji_count = 0
    for char in text:
        if ord(char) > 0x1F300:  # Basic emoji range check
            emoji_count += 1
    return emoji_count

async def _check_spam(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Check if message is spam and handle it.
    Returns True if message was spam and handled, False otherwise.
    """
    # Only work in groups
    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        return False
    
    # Don't check messages without text
    if not update.message or not update.message.text:
        return False
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    message_text = update.message.text
    current_time = time.time()
    
    # Never moderate admins
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            return False
    except:
        pass
    
    # Check if bot is admin (needed to delete messages)
    try:
        bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)
        if bot_member.status not in [ChatMemberStatus.ADMINISTRATOR]:
            return False  # Bot can't delete messages
    except:
        return False
    
    # Initialize tracking for this user
    user_key = (chat_id, user_id)
    if user_key not in USER_MESSAGES:
        USER_MESSAGES[user_key] = []
    
    # Clean old messages (older than 60 seconds)
    USER_MESSAGES[user_key] = [
        (msg, ts) for msg, ts in USER_MESSAGES[user_key]
        if current_time - ts < 60
    ]
    
    # Add current message
    USER_MESSAGES[user_key].append((message_text, current_time))
    
    is_spam = False
    spam_reason = ""
    
    # Check 1: Same message sent 3 times
    message_counts = {}
    for msg, _ in USER_MESSAGES[user_key]:
        message_counts[msg] = message_counts.get(msg, 0) + 1
    
    if message_counts.get(message_text, 0) >= 3:
        is_spam = True
        spam_reason = "repeated message"
    
    # Check 2: More than 5 messages in 10 seconds
    if not is_spam:
        recent_messages = [
            (msg, ts) for msg, ts in USER_MESSAGES[user_key]
            if current_time - ts < 10
        ]
        if len(recent_messages) > 5:
            is_spam = True
            spam_reason = "flooding"
    
    # Check 3: Spam links
    if not is_spam and _is_spam_link(message_text):
        is_spam = True
        spam_reason = "spam link"
    
    # Check 4: Emoji flood (10+ emojis)
    if not is_spam and _count_emojis(message_text) >= 10:
        is_spam = True
        spam_reason = "emoji flood"
    
    # Handle spam
    if is_spam:
        try:
            # Delete the spam message
            await update.message.delete()
            logger.info(f"Deleted spam message from user {user_id}: {spam_reason}")
            
            # Send warning only once per user (within 5 minutes)
            if user_key not in WARNED_USERS or current_time - WARNED_USERS[user_key] > 300:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Thoda slow 🙂 spam mat karo"
                )
                WARNED_USERS[user_key] = current_time
            
            return True
        
        except Exception as e:
            logger.error(f"Spam handling error: {e}")
    
    return False

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

def _download_audio_sync(url: str, output_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Download audio using yt-dlp (sync function for use with asyncio.to_thread)
    
    Returns:
        Tuple of (file_path, metadata_dict) or None if failed
        metadata_dict contains: title, performer, duration
    """
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
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.debug(f"Starting yt-dlp download for: {url}")
            info = ydl.extract_info(url, download=True)
            
            if not info:
                logger.warning(f"No info returned from yt-dlp for: {url}")
                return None
            
            # Extract metadata
            title = info.get("title", "Unknown Title")[:100]
            performer = info.get("uploader", info.get("channel", "Unknown Artist"))[:100]
            duration = info.get("duration", 0)  # in seconds
            
            metadata = {
                "title": title,
                "performer": performer,
                "duration": int(duration) if duration else None,
            }
            
            # Get the downloaded file path (with .mp3 extension after postprocessing)
            filename = ydl.prepare_filename(info)
            file_path = Path(filename)
            
            # Check for .mp3 file if postprocessing was done
            mp3_path = file_path.with_suffix(".mp3")
            if mp3_path.exists():
                file_path = mp3_path
            
            logger.info(f"File path: {file_path}")
            logger.info(f"File exists: {file_path.exists()}")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✅ Downloaded: {file_path.name} ({file_size / 1024:.1f}KB)")
                logger.info(f"📝 Metadata: {metadata}")
                return (file_path, metadata)
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

def _get_model_candidates() -> list[str]:
    """Get available Gemini models (using new google.genai API)"""
    if GEMINI_MODEL_CACHE:
        return GEMINI_MODEL_CACHE
    
    # Just return predefined models since new API doesn't have list_models
    GEMINI_MODEL_CACHE.extend(GEMINI_MODELS)
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
            if content:
                logger.info(f"✅ OpenRouter success: {len(content)} chars")
                return content.strip()
            else:
                logger.warning("⚠️ OpenRouter returned empty content")
                return None
    except httpx.HTTPStatusError as exc:
        logger.error(f"❌ OpenRouter HTTP error {exc.response.status_code}: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        logger.error(f"❌ OpenRouter error: {type(exc).__name__}: {exc}")
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
    Get AI response from Gemini with error handling (using new google.genai API).
    Returns a friendly response or fallback message.
    Supports custom system prompts for different contexts.
    """
    if not GEMINI_API_KEY or not GEMINI_CLIENT:
        return "AI service temporarily unavailable. Please try again later."

    try:
        # Use provided system prompt or default
        sys_prompt = system_prompt or SYSTEM_PROMPT
        
        # Create conversation with user message
        prompt = f"User ({user_name}): {user_message}"

        last_error: Exception | None = None

        for model_name in _get_model_candidates():
            try:
                # Use new google.genai API
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "system_instruction": sys_prompt,
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 500,
                    }
                )

                if response and response.text:
                    logger.info(f"✅ Gemini success ({model_name}): {len(response.text)} chars")
                    return response.text.strip()
            except Exception as inner_exc:
                last_error = inner_exc
                logger.warning(f"⚠️ Gemini model {model_name} failed: {type(inner_exc).__name__}: {str(inner_exc)[:100]}")
                continue

        if last_error:
            logger.error("Gemini API error: %s", last_error)
        return "thoda network issue lag raha hai 😅 phir se try karna"

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "thoda network issue lag raha hai 😅 phir se try karna"


# ========================= COMMAND HANDLERS ========================= #

async def my_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle when bot is added to or removed from a group"""
    try:
        if not update.my_chat_member:
            return
        
        chat = update.effective_chat
        new_status = update.my_chat_member.new_chat_member.status
        old_status = update.my_chat_member.old_chat_member.status
        
        # Check if bot was added to a group
        if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            # Bot was added to group
            if new_status in [ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR] and \
               old_status not in [ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR]:
                await _register_group(chat.id, chat)
                logger.info(f"✅ Bot added to group: {chat.title} ({chat.id})")
                
                # Send welcome message
                try:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=(
                            "🎉 Heyy! Main Baby hoon ❤️\n\n"
                            "Commands:\n"
                            "/song <name> - Gana download karo 🎵\n"
                            "/help - Saari commands dekho\n"
                            "/all - Sabko tag karo (admin only)\n\n"
                            "Bas 'baby' bolke mujhe bula lo 😄"
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not send welcome message to group {chat.id}: {e}")
            
            # Bot was removed from group
            elif new_status in [ChatMemberStatus.LEFT, ChatMemberStatus.BANNED] and \
                 old_status in [ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR]:
                # Remove from groups database
                if chat.id in GROUPS_DATABASE:
                    del GROUPS_DATABASE[chat.id]
                    _save_groups_database()
                logger.info(f"❌ Bot removed from group: {chat.title} ({chat.id})")
    
    except Exception as e:
        logger.error(f"Error in my_chat_member_handler: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with inline buttons"""
    # Register user with full details
    await _register_user_from_update(update)
    
    user_id = update.effective_user.id
    
    # Remove from opted-out list if they were opted out
    if user_id in OPTED_OUT_USERS:
        OPTED_OUT_USERS.discard(user_id)
        _save_opted_out_users(OPTED_OUT_USERS)
        logger.info(f"📥 User {user_id} opted back in to broadcasts")
    
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


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command - Opt out of broadcasts"""
    user_id = update.effective_user.id
    
    # Only works in private chat
    if update.effective_chat.type != ChatType.PRIVATE:
        await update.effective_message.reply_text(
            "❌ Ye command sirf private chat mein use kar sakte ho."
        )
        return
    
    # Check if already opted out
    if user_id in OPTED_OUT_USERS:
        await update.effective_message.reply_text(
            "✅ Tumhe pehle se hi broadcasts nahi mil rahe hain.\n\n"
            "Agar dobara chahiye toh /start karke fir se activate kar sakte ho! 😊"
        )
        return
    
    # Add to opted-out list
    OPTED_OUT_USERS.add(user_id)
    _save_opted_out_users(OPTED_OUT_USERS)
    
    logger.info(f"📵 User {user_id} opted out of broadcasts")
    
    await update.effective_message.reply_text(
        "✅ Done! Ab tumhe broadcasts nahi aayenge.\n\n"
        "Agar kabhi wapas chahiye toh /start karke dobara activate kar sakte ho! 💕"
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command - Show bot statistics (admin only)"""
    user_id = update.effective_user.id
    
    # Only admin can view stats
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Sirf admin (bot owner) ye command use kar sakte hain! 😊"
        )
        return
    
    await _register_user_from_update(update)
    
    # Calculate statistics
    total_users = len(USERS_DATABASE)
    total_groups = len(GROUPS_DATABASE)
    opted_out_count = len(OPTED_OUT_USERS)
    active_users = total_users - opted_out_count
    
    # Get recent users (last 24 hours)
    current_time = time.time()
    recent_users = sum(1 for user in USERS_DATABASE.values() 
                       if current_time - user.get("last_seen", 0) < 86400)
    
    # Group breakdown
    group_types = {}
    for group in GROUPS_DATABASE.values():
        group_type = group.get("type", "unknown")
        group_types[group_type] = group_types.get(group_type, 0) + 1
    
    # Create stats message
    stats_text = (
        "📊 *Bot Statistics* - Baby ❤️\n\n"
        
        "👥 *Users:*\n"
        f"├ Total Users: {total_users}\n"
        f"├ Active: {active_users}\n"
        f"├ Opted Out: {opted_out_count}\n"
        f"└ Active (24h): {recent_users}\n\n"
        
        "📢 *Groups:*\n"
        f"└ Total Groups: {total_groups}\n"
    )
    
    # Add group type breakdown if available
    if group_types:
        stats_text += "\n*Group Types:*\n"
        for gtype, count in group_types.items():
            stats_text += f"├ {gtype}: {count}\n"
    
    stats_text += f"\n🕐 *Uptime:* Bot is running\n"
    stats_text += f"📝 *Version:* 2.0 (Enhanced Tracking)\n"
    
    await update.effective_message.reply_text(
        stats_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Stats viewed by admin {user_id}")


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /users command - Show user list (admin only)"""
    user_id = update.effective_user.id
    
    # Only admin can view users
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Sirf admin ye command use kar sakte hain! 😊"
        )
        return
    
    await _register_user_from_update(update)
    
    # Get users sorted by last seen
    sorted_users = sorted(
        USERS_DATABASE.items(),
        key=lambda x: x[1].get("last_seen", 0),
        reverse=True
    )
    
    # Show first 20 users
    users_text = "👥 *Registered Users* (Recent 20):\n\n"
    
    for idx, (uid, user_data) in enumerate(sorted_users[:20], 1):
        username = user_data.get("username", "None")
        first_name = user_data.get("first_name", "Unknown")
        msg_count = user_data.get("message_count", 0)
        
        # Format last seen
        last_seen = user_data.get("last_seen", 0)
        time_diff = time.time() - last_seen
        if time_diff < 300:  # 5 minutes
            last_seen_str = "Just now"
        elif time_diff < 3600:  # 1 hour
            last_seen_str = f"{int(time_diff/60)}m ago"
        elif time_diff < 86400:  # 1 day
            last_seen_str = f"{int(time_diff/3600)}h ago"
        else:
            last_seen_str = f"{int(time_diff/86400)}d ago"
        
        users_text += (
            f"{idx}. *{first_name}* "
            f"(@{username})\n"
            f"   ID: `{uid}` | {msg_count} msgs | {last_seen_str}\n\n"
        )
    
    users_text += f"📊 Total: {len(USERS_DATABASE)} users"
    
    await update.effective_message.reply_text(
        users_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Users list viewed by admin {user_id}")


async def groups_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /groups command - Show group list (admin only)"""
    user_id = update.effective_user.id
    
    # Only admin can view groups
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Sirf admin ye command use kar sakte hain! 😊"
        )
        return
    
    await _register_user_from_update(update)
    
    # Get groups sorted by last active
    sorted_groups = sorted(
        GROUPS_DATABASE.items(),
        key=lambda x: x[1].get("last_active", 0),
        reverse=True
    )
    
    # Show all groups
    groups_text = "📢 *Registered Groups*:\n\n"
    
    for idx, (gid, group_data) in enumerate(sorted_groups, 1):
        title = group_data.get("title", "Unknown")
        username = group_data.get("username", None)
        
        # Format last active
        last_active = group_data.get("last_active", 0)
        time_diff = time.time() - last_active
        if time_diff < 300:  # 5 minutes
            last_active_str = "Active now"
        elif time_diff < 3600:  # 1 hour
            last_active_str = f"{int(time_diff/60)}m ago"
        elif time_diff < 86400:  # 1 day
            last_active_str = f"{int(time_diff/3600)}h ago"
        else:
            last_active_str = f"{int(time_diff/86400)}d ago"
        
        username_str = f"@{username}" if username else "No username"
        
        groups_text += (
            f"{idx}. *{title}*\n"
            f"   {username_str} | {last_active_str}\n"
            f"   ID: `{gid}`\n\n"
        )
    
    groups_text += f"📊 Total: {len(GROUPS_DATABASE)} groups"
    
    await update.effective_message.reply_text(
        groups_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Groups list viewed by admin {user_id}")


    logger.info(f"Groups list viewed by admin {user_id}")


async def members_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /members command - Show group members (admin or group-specific)"""
    user_id = update.effective_user.id
    chat = update.effective_chat
    
    await _register_user_from_update(update)
    
    # If used in private chat by admin, show format instructions
    if chat.type == ChatType.PRIVATE:
        if user_id != ADMIN_ID:
            await update.effective_message.reply_text(
                "❌ Ye command sirf groups mein use karo! 😊"
            )
            return
        
        # Admin can use in private with group ID
        if not context.args:
            await update.effective_message.reply_text(
                "📋 *Usage:*\n"
                "`/members` - Group mein use karo\n"
                "`/members <group_id>` - Private mein specific group ke members dekho",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        try:
            group_id = int(context.args[0])
        except ValueError:
            await update.effective_message.reply_text(
                "❌ Invalid group ID! Number dalo."
            )
            return
    else:
        # Used in group
        group_id = chat.id
        await _register_group(group_id, chat)
    
    # Check if group exists in database
    if group_id not in GROUPS_DATABASE:
        await update.effective_message.reply_text(
            "❌ Group database mein nahi mila! Pehle kuch messages bhejo."
        )
        return
    
    group_data = GROUPS_DATABASE[group_id]
    members = group_data.get("members", {})
    
    if not members:
        await update.effective_message.reply_text(
            "👥 Abhi tak koi member track nahi hua!\n"
            "Jab log messages bhejenge, tab automatically add honge. ✨"
        )
        return
    
    # Sort by message count (most active first)
    sorted_members = sorted(
        members.items(),
        key=lambda x: x[1].get("message_count", 0),
        reverse=True
    )
    
    # Show first 20 members
    group_title = group_data.get("title", "Unknown Group")
    members_text = f"👥 *Members of {group_title}* (Top 20):\n\n"
    
    for idx, (uid_str, member_data) in enumerate(sorted_members[:20], 1):
        first_name = member_data.get("first_name", "Unknown")
        username = member_data.get("username", None)
        msg_count = member_data.get("message_count", 0)
        
        # Format last seen
        last_seen = member_data.get("last_seen", 0)
        time_diff = time.time() - last_seen
        if time_diff < 300:  # 5 minutes
            last_seen_str = "Active now"
        elif time_diff < 3600:  # 1 hour
            last_seen_str = f"{int(time_diff/60)}m ago"
        elif time_diff < 86400:  # 1 day
            last_seen_str = f"{int(time_diff/3600)}h ago"
        else:
            last_seen_str = f"{int(time_diff/86400)}d ago"
        
        username_str = f"@{username}" if username else "No username"
        
        members_text += (
            f"{idx}. *{first_name}* ({username_str})\n"
            f"   {msg_count} msgs | {last_seen_str}\n\n"
        )
    
    members_text += f"📊 Total: {len(members)} members tracked"
    
    await update.effective_message.reply_text(
        members_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Members list viewed for group {group_id} by {user_id}")


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
    
    # Filter out opted-out users
    active_users = REGISTERED_USERS - OPTED_OUT_USERS
    
    # Send confirmation with user and group count
    total_users = len(active_users)
    total_groups = len(REGISTERED_GROUPS)
    opted_out_count = len(OPTED_OUT_USERS & REGISTERED_USERS)
    total_recipients = total_users + total_groups
    
    confirm_msg = await update.effective_message.reply_text(
        f"📢 Broadcasting to {total_users} users + {total_groups} groups...\n"
        f"({opted_out_count} users opted out)\n\n"
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
    
    # Send message to each active user (excluding opted-out)
    for idx, user_broadcast_id in enumerate(active_users, 1):
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
                # Remove from database
                if user_broadcast_id in USERS_DATABASE:
                    del USERS_DATABASE[user_broadcast_id]
                    _save_users_database()
                
            elif "chat not found" in error_str or "user not found" in error_str:
                failed_users += 1
                logger.warning(f"User {user_broadcast_id} not found")
                # Remove from database
                if user_broadcast_id in USERS_DATABASE:
                    del USERS_DATABASE[user_broadcast_id]
                    _save_users_database()
                
            else:
                failed_users += 1
    
    # Send message to each group
    for idx, group_id in enumerate(GROUPS_DATABASE.keys(), 1):
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
                # Remove from database
                if group_id in GROUPS_DATABASE:
                    del GROUPS_DATABASE[group_id]
                    _save_groups_database()
                
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
        f"  🔒 Blocked: {blocked_count}\n"
        f"  📵 Opted out: {opted_out_count}\n\n"
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
        metadata = None
        successful_url = None
        
        for attempt, url in enumerate(video_urls, 1):
            try:
                # Update status message
                if attempt == 1:
                    await search_msg.edit_text("Baby dhoondh rahi hoon 🎧")
                else:
                    await search_msg.edit_text(f"Hmm ek aur try karti hoon 😄 (attempt {attempt}/{len(video_urls)})")
                
                # Download in thread
                result = await asyncio.to_thread(_download_audio_sync, url, user_dir)
                
                if result:
                    audio_file, metadata = result
                    
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
                metadata = None
                continue
        
        # Check if we got a valid file
        if not audio_file or not audio_file.exists():
            await search_msg.edit_text("Aww 😅 ye gana nahi mil raha")
            logger.error(f"Failed to download any version of: {song_name}")
            return
        
        # Success! Update message with proper response
        await search_msg.edit_text("🎵 Song ready 😄\nPlay karke suno, pause bhi kar sakte ho ❤️")
        
        # Send the audio file with proper metadata
        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=metadata.get("title", audio_file.stem[:100]) if metadata else audio_file.stem[:100],
                    performer=metadata.get("performer", "Unknown Artist") if metadata else "Unknown Artist",
                    duration=metadata.get("duration") if metadata else None,
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
        result = await asyncio.to_thread(_download_audio_sync, youtube_link, user_dir)
        
        if not result:
            await search_msg.edit_text("Aww 😅 ye gana nahi mil raha")
            logger.warning(f"Download failed for YouTube link: {youtube_link}")
            return
        
        audio_file, metadata = result
        
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
        
        # Success! Update message with proper response
        await search_msg.edit_text("🎵 Song ready 😄\nPlay karke suno, pause bhi kar sakte ho ❤️")
        
        # Send file to user with proper metadata
        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=metadata.get("title", audio_file.stem[:100]) if metadata else audio_file.stem[:100],
                    performer=metadata.get("performer", "Unknown Artist") if metadata else "Unknown Artist",
                    duration=metadata.get("duration") if metadata else None,
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
    
    # Check if message contains @all (case-insensitive)
    if "@all" not in update.message.text.lower():
        return  # Not an @all mention, let other handlers process it
    
    # Register group
    await _register_group(update.effective_chat.id, update.effective_chat)
    
    # Register the user who sent @all command as a member
    user_id = update.effective_user.id
    username = update.effective_user.username
    first_name = update.effective_user.first_name or "User"
    await _register_group_member(update.effective_chat.id, user_id, username, first_name)
    
    # Check admin status
    is_admin = await _check_admin_status(update, context)
    if not is_admin:
        await update.message.reply_text(
            "Sirf admins @all use kar sakte hain! 🚫",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Check cooldown
    if _check_cooldown(update.effective_chat.id):
        remaining = int(COOLDOWN_SECONDS - (time.time() - TAGGING_COOLDOWN[update.effective_chat.id]))
        await update.message.reply_text(
            f"Thoda ruko! {remaining} seconds baad phir try karo 🕐",
            reply_to_message_id=update.message.message_id
        )
        return
    
    # Get active users
    active_users = _get_active_users(update.effective_chat.id)
    
    if not active_users:
        await update.message.reply_text(
            "Koi active user nahi hai abhi! 🤷\n"
            "Log chat karne ke baad try karo.",
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
            message_text = f"{mentions}\n\n🔔 Alert!"
        
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=update.message.message_id if idx == 0 else None
            )
        except Exception as e:
            logger.error(f"Failed to send @all alert: {e}")
            await update.message.reply_text(
                "Tag bhejne mein error aa gaya 😅 Shayad bot admin nahi hai?",
                reply_to_message_id=update.message.message_id
            )
            return
    
    # Send confirmation
    status_msg = f"✅ {len(active_users)} users ko tag kar diya!"
    if len(batches) > 1:
        status_msg += f" ({len(batches)} messages mein)"
    
    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=status_msg,
            reply_to_message_id=update.message.message_id
        )
    except:
        pass


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


async def ga_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ga (Good Afternoon) command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    ga_messages = [
        f"Good afternoon ☀️ {user_name}! Lunch ho gaya? Kuch achha khaya? 😋",
        f"Afternoon {user_name}! 🌞 Dopahar ka time hai, thoda rest le lo 😊",
        f"Namaste {user_name}! 👋 Afternoon ka vibe kaisa hai? Mast? 🌤️",
        f"Good afternoon ✨ {user_name}! Din kaisa ja raha hai? Productive? 💪",
        f"Afternoon ho gayi {user_name}! ☀️ Kuch special plan hai shaam ke liye? 😄"
    ]
    
    await update.effective_message.reply_text(random.choice(ga_messages))


async def ge_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ge (Good Evening) command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    ge_messages = [
        f"Good evening 🌆 {user_name}! Din kaisa gaya? Achha tha? 😊",
        f"Evening ho gayi {user_name}! 🌅 Chai-pakode ka time hai 😋☕",
        f"Shaam ko bhi yaad kar liya? 🌆 Sweet! Evening {user_name}! ❤️",
        f"Good evening ✨ {user_name}! Ab chill karo, din khatam ho gaya 😊",
        f"Evening vibes 🌇 {user_name}! Relax mode on kar lo 😄"
    ]
    
    await update.effective_message.reply_text(random.choice(ge_messages))


async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /chat command - Start conversation"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    chat_messages = [
        f"Haan {user_name}! 😄 Bol kya baat karni hai? Main sun rahi hoon 👂",
        f"Bilkul {user_name}! 💬 Batao kya chal raha hai life mein? 😊",
        f"Chal {user_name}! ✨ Shuru karte hain conversation! Kya hua? 😄",
        f"Haan bhai {user_name}! 👋 Main ready hoon, tu bata kya discuss karenge? 💭"
    ]
    
    await update.effective_message.reply_text(random.choice(chat_messages))


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command - Answer questions"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    if not context.args:
        await update.effective_message.reply_text(
            f"Arre {user_name}! 😊 Kuch pucho na!\n\n"
            "Format: /ask <question>\n"
            "Example: /ask Python kya hai?"
        )
        return
    
    question = " ".join(context.args)
    
    # Use AI to answer
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        answer = get_ai_response(question, user_name, SYSTEM_PROMPT)
        await update.effective_message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ask command error: {e}")
        await update.effective_message.reply_text(
            f"Hmm {user_name}, thoda network issue lag raha hai 😅 Phir se pucho na!"
        )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about command - Bot introduction"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    about_messages = [
        f"Hii {user_name}! 😊 Main Baby hoon ❤️\n\n"
        "Main ek friendly bot hoon jo tumse baat karta hai 💬\n"
        "Songs download karti hoon 🎵\n"
        "Aur tumhara mood achha rakhti hoon ✨\n\n"
        "Bas mujhe 'baby' bolke bula lo! 😄",
        
        f"Hello {user_name}! 👋\n\n"
        "Main Baby hoon - tumhari dost ❤️\n"
        "Gaane sunau, baat karu, help karu 😊\n"
        "Hinglish mein friendly talks! 💭\n\n"
        "Bas yaad se bula lena 😄",
        
        f"Namaste {user_name}! 🙏\n\n"
        "Main Baby ❤️ - cute aur friendly!\n"
        "Songs 🎵, chats 💬, aur masti 😄\n"
        "Hinglish speaking human-like bot!\n\n"
        "Mujhse baat karo! 😊"
    ]
    
    await update.effective_message.reply_text(random.choice(about_messages))


async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /privacy command"""
    await _register_user(update.effective_user.id)
    
    privacy_text = (
        "🔒 *Privacy Policy*\n\n"
        "✅ Main tumhari personal info store nahi karti\n"
        "✅ Messages private rehti hain\n"
        "✅ Data safe aur secure hai\n"
        "✅ Sirf chat_id save hoti hai\n\n"
        "Tum safe ho mere saath! 😊❤️"
    )
    
    await update.effective_message.reply_text(privacy_text, parse_mode=ParseMode.MARKDOWN)


async def sad_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sad command - Emotional support"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    sad_messages = [
        f"Aww {user_name} 🥺 Udaas ho? Koi baat nahi, main hoon na!\n"
        "Yaad rakho - ye phase guzar jayega ✨\n"
        "Tum strong ho 💪 Smile karo! 😊",
        
        f"{user_name}, sun mere baat 🤗\n"
        "Sad hona normal hai, but permanent nahi hai!\n"
        "Kal better hoga ✨ Trust me!\n"
        "Main hoon tumhare saath ❤️",
        
        f"Arre {user_name}! 🥺 Kya hua?\n"
        "Life mein ups-downs toh aate hain\n"
        "But tum warrior ho 💪\n"
        "Cheer up! Main yahi hoon 😊❤️",
        
        f"{user_name}, relax 🌸\n"
        "Har raat ke baad subah hoti hai ☀️\n"
        "Tum iss se stronger nikalne wale ho!\n"
        "Believe karo apne aap pe 💖"
    ]
    
    await update.effective_message.reply_text(random.choice(sad_messages))


async def happy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /happy command - Celebrate happiness"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    happy_messages = [
        f"Yayy {user_name}! 🎉 Happy ho? Mujhe bhi khushi hui!\n"
        "Ye energy maintain rakho! 😄✨\n"
        "Zindagi mast hai! ❤️",
        
        f"Wohoo {user_name}! 🥳 Happiness dekh ke main bhi khush!\n"
        "Is positivity ko spread karo 🌟\n"
        "Keep smiling! 😊💕",
        
        f"Amazing {user_name}! 🎊 Tumhari khushi meri khushi!\n"
        "Life is beautiful na? 🌈\n"
        "Enjoy every moment! 😄❤️",
        
        f"Superb {user_name}! ✨ Happy vibes I love it!\n"
        "Aise hi mast raho 😊\n"
        "Tumhari smile precious hai! 💖"
    ]
    
    await update.effective_message.reply_text(random.choice(happy_messages))


async def angry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /angry command - Calming advice"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    angry_messages = [
        f"Arre {user_name}! 😊 Gussa ho? Thoda relax karo\n"
        "Deep breath lo 🌬️\n"
        "Anger temporary hai, peace permanent 🕊️\n"
        "Chill karo! ✨",
        
        f"{user_name}, sun 🙏 Gussa sahi nahi!\n"
        "Kuch minutes wait karo\n"
        "Shaant dimag se sochna better hai 💭\n"
        "Main samajh sakti hoon! 😊",
        
        f"Relax {user_name}! 🌸 Anger hota hai\n"
        "But isse handle karo smartly 🧠\n"
        "Calm down, breathe, think 💆\n"
        "Sab theek ho jayega! ❤️",
        
        f"Oye {user_name}! 😅 Cool down bro\n"
        "Gusse mein galat decision mat lo\n"
        "Thoda time do apne aap ko ⏰\n"
        "Peace is power! ✌️"
    ]
    
    await update.effective_message.reply_text(random.choice(angry_messages))


async def motivate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /motivate command - Motivational messages"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    motivate_messages = [
        f"{user_name}, sun! 💪\nTum capable ho kuch bhi karne ke liye!\nBas believe karo aur try karo! 🚀",
        f"Arre {user_name}! ✨\nHar mushkil ka solution hota hai\nGive up mat karo! 💯",
        f"{user_name}, remember! 🌟\nSuccess waiting hai tumhare liye\nBas ek step aur! 🎯",
        f"Yaar {user_name}! 💪\nTum warrior ho!\nKoi tumhe rok nahi sakta! 🔥",
        f"Listen {user_name}! 🌈\nDreams sach hote hain\nWork hard aur patient raho! ⏰",
        f"{user_name}, focus! 🎯\nTumhare andar talent hai\nDimag pe zor do! 🧠",
        f"Bhai {user_name}! 💖\nFailure is learning\nHar try tumhe better banati hai! 📈",
        f"{user_name}, push harder! 🚀\nGoals door nahi, paas hain\nThoda aur effort! 💪"
    ]
    
    await update.effective_message.reply_text(random.choice(motivate_messages))


async def howareyou_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /howareyou command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    howareyou_messages = [
        f"Main achhi hoon {user_name}! 😊 Thanks for asking!\nTum kaise ho? ❤️",
        f"Bilkul mast {user_name}! 😄 Tumne pucha na toh aur achha lag raha! 💕",
        f"Main theek hoon yaar! ✨ Tum batao, tumhara din kaisa ja raha hai? 😊",
        f"All good {user_name}! 😊 Tumhari care sweet hai! Tumhara kya haal? 🌸"
    ]
    
    await update.effective_message.reply_text(random.choice(howareyou_messages))


async def missyou_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /missyou command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    missyou_messages = [
        f"Aww {user_name}! 🥺 Main bhi tumhe miss kar rahi thi!\nLong time no see! ❤️",
        f"Miss you too {user_name}! 💕 Itne din kaha the? Glad you're back! 😊",
        f"{user_name}! 🥰 Main yahi hoon na! Tumhe bhi miss kar rahi thi! 💖",
        f"Oye {user_name}! 😊 Miss me? Sweet! Main bhi yaad kar rahi thi tumhe! ❤️"
    ]
    
    await update.effective_message.reply_text(random.choice(missyou_messages))


async def thankyou_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /thankyou command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    thankyou_messages = [
        f"You're welcome {user_name}! 😊 Meri khushi hai help karna! ❤️",
        f"No problem yaar! 🤗 Tere liye kuch bhi {user_name}! 💕",
        f"Arre koi baat nahi {user_name}! 😄 Main hoon na tumhare liye! ✨",
        f"Anytime {user_name}! 💖 Mere se na sharma! 😊"
    ]
    
    await update.effective_message.reply_text(random.choice(thankyou_messages))


async def hug_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /hug command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    hug_messages = [
        f"🤗 *gives tight hug to {user_name}*\nAww! Feel better? ❤️",
        f"*hugs {user_name} warmly* 🤗\nYou needed this! Everything will be okay! 💕",
        f"🤗 Aaaja {user_name}! *virtual hug*\nYou're amazing! ❤️",
        f"*squeezes {user_name} in a hug* 🤗💖\nFeeling the warmth? Main hoon na! 😊"
    ]
    
    await update.effective_message.reply_text(random.choice(hug_messages))


async def tip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tip command - Daily life tips"""
    await _register_user(update.effective_user.id)
    
    tips = [
        "💡 *Daily Tip*\nSubah jaldi utho! Morning productivity best hoti hai ☀️",
        "💡 *Daily Tip*\nPaani zyada piyo! 8-10 glass must hai 💧",
        "💡 *Daily Tip*\nScreen time kam karo, eyes ko rest do 👀✨",
        "💡 *Daily Tip*\n5 minutes meditation daily - game changer hai 🧘",
        "💡 *Daily Tip*\nTo-do list banao! Organized life = peaceful life 📝",
        "💡 *Daily Tip*\nWalk karo daily! 30 minutes is enough 🚶",
        "💡 *Daily Tip*\nBooks padho! Knowledge is power 📚",
        "💡 *Daily Tip*\nPositive sochao! Negativity se door raho ✨",
        "💡 *Daily Tip*\nFamily time zaruri hai! Quality time spend karo ❤️",
        "💡 *Daily Tip*\nGratitude practice karo! Thank you bolo daily 🙏"
    ]
    
    await update.effective_message.reply_text(random.choice(tips), parse_mode=ParseMode.MARKDOWN)


async def confidence_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /confidence command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    confidence_messages = [
        f"{user_name}, tum perfect ho! 💯\nApne aap pe believe karo\nConfidence tumhara superpower hai! 🦸",
        f"Listen {user_name}! 🌟\nTum unique ho\nKisi se compare mat karo\nBe confidently YOU! 💪",
        f"{user_name}, yaad rakho! ✨\nTumhare andar power hai\nDarna nahi, shine karna hai! 🌟",
        f"Arre {user_name}! 🔥\nSelf-doubt ko bhagao\nTum capable ho\nJust believe! 💖"
    ]
    
    await update.effective_message.reply_text(random.choice(confidence_messages))


async def focus_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /focus command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    focus_messages = [
        f"{user_name}, focus tips! 🎯\n"
        "1. Phone silent karo 📵\n"
        "2. 25 min work, 5 min break ⏰\n"
        "3. One task at a time 💪",
        
        f"Focus strategy {user_name}! 🧠\n"
        "• Distractions band karo\n"
        "• Goal clear rakho\n"
        "• Pomodoro technique try karo ⏲️",
        
        f"Hey {user_name}! 🎯\n"
        "Focus = Success key\n"
        "Multitasking nahi, deep work karo\n"
        "Results guaranteed! 💯",
        
        f"{user_name}, productivity hack! ⚡\n"
        "Morning mein important task\n"
        "Evening mein creative work\n"
        "Smart work karo! 🧠"
    ]
    
    await update.effective_message.reply_text(random.choice(focus_messages))


async def sleep_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sleep command"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    sleep_messages = [
        f"{user_name}, sleep is important! 😴\n"
        "7-8 hours zaruri hai\n"
        "Phone door rakho bed se\n"
        "Good sleep = Good life 🌙",
        
        f"Sleep tips {user_name}! 💤\n"
        "• Same time pe sona-uthna\n"
        "• Room dark rakho\n"
        "• Stress kam karo\n"
        "Quality sleep = Quality you! ✨",
        
        f"Hey {user_name}! 🌙\n"
        "Neend achhi honi chahiye\n"
        "Late night phone avoid karo\n"
        "Rest is productivity secret! 😊",
        
        f"{user_name}, listen! 😴\n"
        "Sleep sacrifice mat karo\n"
        "Body ko rest chahiye\n"
        "Health first! ❤️"
    ]
    
    await update.effective_message.reply_text(random.choice(sleep_messages))


async def lifeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lifeline command - Emotional support"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    lifeline_message = (
        f"{user_name}, main yahi hoon! 🤗\n\n"
        "Agar tough time ja raha hai:\n"
        "• Deep breath lo 🌬️\n"
        "• Kisi se baat karo 💬\n"
        "• Professional help lena okay hai 🏥\n\n"
        "You're not alone ❤️\n"
        "Things will get better! ✨\n\n"
        "Main hamesha tumhare saath hoon! 💖"
    )
    
    await update.effective_message.reply_text(lifeline_message)


async def joke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /joke command"""
    await _register_user(update.effective_user.id)
    
    jokes = [
        "😄 *Joke*\nTeacher: Beta, tumhare phone mein calculator hai?\nStudent: Haan!\nTeacher: Toh homework mein 2+2 galat kaise? 😂",
        
        "😄 *Joke*\nDoctor: Aapko diabetes hai\nPatient: Okay, koi baat nahi\nDoctor: Sugar kam karo\nPatient: WHAT? 😱",
        
        "😄 *Joke*\nWife: Tum mujhe pyaar karte ho?\nHusband: Haan\nWife: Kitna?\nHusband: Jitna WiFi ka password yaad hai! 😂",
        
        "😄 *Joke*\nBeta: Papa, main fail ho gaya\nPapa: Koi baat nahi, agli baar pass ho jaoge\nBeta: Next week dobara exam hai\nPapa: 😱",
        
        "😄 *Joke*\nBoy: I love you\nGirl: Proof do\nBoy: *Screenshot of WhatsApp chat*\nGirl: 😂",
        
        "😄 *Joke*\nTeacher: 'I' ke baad 'am' aata hai\nStudent: I aam?\nTeacher: No! I am\nStudent: You're aam? 🥭😂",
        
        "😄 *Joke*\nPapa: Beta, bill zyada aa raha hai\nBeta: Light band kar dete hain\nPapa: *turns off son's phone data* 😂",
        
        "😄 *Joke*\nGF: Mujhe gift do\nBF: *gives hug* 🤗\nGF: I said gift, not shift! 😂"
    ]
    
    await update.effective_message.reply_text(random.choice(jokes), parse_mode=ParseMode.MARKDOWN)


async def roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /roast command - Light roasting"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    roasts = [
        f"Arre {user_name}! 😂 Tum itne smart ho ki Google bhi confused ho jata hai! 🤭",
        f"{user_name}, tumhari productivity dekh ke snail bhi motivation lete hain! 😄🐌",
        f"Oye {user_name}! 😂 Tumhari WiFi speed aur tumhare replies same hai - slow! 🤭",
        f"{user_name}, tum itne late aate ho ki 'Better late than never' bhi doubt karta hai! 😂",
        f"Arre {user_name}! 🤭 Tumhare excuses itne creative hain ki Netflix ko scripts de sakte ho! 😄",
        f"{user_name}, tumhara phone battery aur tumhari energy level same hai - always low! 😂🔋"
    ]
    
    await update.effective_message.reply_text(random.choice(roasts))


async def truth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /truth command"""
    await _register_user(update.effective_user.id)
    
    truths = [
        "🎯 *Truth Question*\nKya tumne kabhi kisi ko secretly like kiya hai? 😳",
        "🎯 *Truth Question*\nTumhari sabse embarrassing moment kya thi? 🙈",
        "🎯 *Truth Question*\nKya tumne kabhi kisi ki copy ki hai exam mein? 📝😅",
        "🎯 *Truth Question*\nTumhara crush kaun hai? (Honest answer!) 💕",
        "🎯 *Truth Question*\nKya tumne kabhi kisi ko jhooth bola hai? 🤥",
        "🎯 *Truth Question*\nTumhari secret talent kya hai? 🎭",
        "🎯 *Truth Question*\nKya tumne kabhi raat mein khana chori kiya hai? 😂🍕",
        "🎯 *Truth Question*\nTumhara biggest fear kya hai? 😨"
    ]
    
    await update.effective_message.reply_text(random.choice(truths), parse_mode=ParseMode.MARKDOWN)


async def dare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /dare command"""
    await _register_user(update.effective_user.id)
    
    dares = [
        "🎲 *Dare*\nApne crush ko 'Hi' message bhejo! 😄💕",
        "🎲 *Dare*\n5 jumping jacks kar ke proof video bhejo! 💪",
        "🎲 *Dare*\nApni best friend ko funny voice note bhejo! 🎤😂",
        "🎲 *Dare*\nNext 10 minutes phone band rakho! 📵",
        "🎲 *Dare*\nApni favorite song gao aur audio bhejo! 🎵",
        "🎲 *Dare*\nKisi ko random compliment do! 💕",
        "🎲 *Dare*\n10 push-ups karo! Right now! 💪",
        "🎲 *Dare*\nApni weirdest photo share karo! 📸😂"
    ]
    
    await update.effective_message.reply_text(random.choice(dares), parse_mode=ParseMode.MARKDOWN)


async def fact_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /fact command - Interesting facts"""
    await _register_user(update.effective_user.id)
    
    facts = [
        "🌟 *Interesting Fact*\nHoney kabhi kharab nahi hoti! 3000 saal purana honey bhi edible hai 🍯",
        "🌟 *Amazing Fact*\nDolphins apna naam rakhti hain aur ek-dusre ko naam se bulati hain! 🐬💕",
        "🌟 *Mind-Blowing Fact*\nEk cloud ka weight approximately 1.1 million pounds hota hai! ☁️",
        "🌟 *Cool Fact*\nOctopus ke teen hearts hote hain! 🐙❤️❤️❤️",
        "🌟 *Fascinating Fact*\nBananas technically berries hain, but strawberries nahi! 🍌🍓",
        "🌟 *Incredible Fact*\nHumans aur bananas 60% DNA share karte hain! 🧬🍌",
        "🌟 *Wonderful Fact*\nEiffel Tower summer mein 6 inches tall ho jata hai heat se! 🗼☀️",
        "🌟 *Surprising Fact*\nPenguins propose karte hain by giving pebbles! 🐧💍",
        "🌟 *Beautiful Fact*\nButterflies taste with their feet! 🦋👣",
        "🌟 *Amazing Fact*\nShark dinosaurs se bhi pehle exist karte the! 🦈🦕",
        "🌟 *Crazy Fact*\nEk teaspoon neutron star ka weight 6 billion tons hoga! ⭐",
        "🌟 *Interesting Fact*\nKoala fingerprints humans se identical hote hain! 🐨👆",
        "🌟 *Fun Fact*\nCats 70% of their life sleeping mein spend karte hain! 😺😴",
        "🌟 *Cool Fact*\nWater bear (tardigrade) space mein survive kar sakta hai! 🐻",
        "🌟 *Awesome Fact*\nHummingbird backwards fly kar sakta hai! 🐦✨"
    ]
    
    await update.effective_message.reply_text(random.choice(facts), parse_mode=ParseMode.MARKDOWN)


# ========================= ADMIN MODERATION COMMANDS ========================= #

async def _check_bot_and_user_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, str]:
    """Check if bot and user are both admins. Returns (is_valid, error_message)"""
    
    # Must be in a group
    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        return False, "❌ Ye command sirf groups mein kaam karta hai!"
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    bot_id = context.bot.id
    
    try:
        # Check if user is admin
        user_member = await context.bot.get_chat_member(chat_id, user_id)
        if user_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            return False, "❌ Sirf admins hi ye command use kar sakte hain! 😊"
        
        # Check if bot is admin
        bot_member = await context.bot.get_chat_member(chat_id, bot_id)
        if bot_member.status not in [ChatMemberStatus.ADMINISTRATOR]:
            return False, "❌ Mujhe pehle admin banao, phir main help kar sakti hoon! 😅"
        
        return True, ""
    
    except Exception as e:
        logger.error(f"Admin check error: {e}")
        return False, "❌ Permission check mein problem aa gayi! 😅"


async def del_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /del command - Delete replied message"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi message ko reply karke /del use karo! 😊"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    try:
        # Delete the replied message
        await update.message.reply_to_message.delete()
        
        # Delete the command message too
        await update.message.delete()
        
        logger.info(f"Message deleted by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Delete error: {e}")
        await update.effective_message.reply_text(
            "❌ Message delete nahi ho paya! Shayad bahut purana hai 😅"
        )


async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ban command - Ban replied user"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi user ke message ko reply karke /ban use karo! 😊"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    target_user = update.message.reply_to_message.from_user
    
    # Don't ban admins
    try:
        target_member = await context.bot.get_chat_member(update.effective_chat.id, target_user.id)
        if target_member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            await update.effective_message.reply_text(
                "❌ Admin ko ban nahi kar sakte! 😅"
            )
            return
    except:
        pass
    
    try:
        # Ban the user
        await context.bot.ban_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id
        )
        
        user_name = target_user.first_name or "User"
        await update.effective_message.reply_text(
            f"✅ {user_name} ko ban kar diya! 🚫\n"
            "Unban karne ke liye /unban use karo."
        )
        
        logger.info(f"User {target_user.id} banned by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Ban error: {e}")
        await update.effective_message.reply_text(
            "❌ Ban nahi ho paya! Permission issue ho sakta hai 😅"
        )


async def unban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unban command - Unban user"""
    await _register_user(update.effective_user.id)
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    # Get user ID from reply or argument
    target_user_id = None
    user_name = "User"
    
    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
        user_name = update.message.reply_to_message.from_user.first_name or "User"
    elif context.args:
        try:
            # Try to parse as user ID
            target_user_id = int(context.args[0])
        except:
            await update.effective_message.reply_text(
                "❌ Valid user ID do! 😊\n"
                "Format: /unban <user_id> ya kisi message ko reply karo"
            )
            return
    else:
        await update.effective_message.reply_text(
            "❌ Kisi banned user ke message ko reply karo ya user ID do! 😊\n"
            "Format: /unban <user_id>"
        )
        return
    
    try:
        # Unban the user
        await context.bot.unban_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user_id,
            only_if_banned=True
        )
        
        await update.effective_message.reply_text(
            f"✅ {user_name} ko unban kar diya! ✨\n"
            "Ab vo dobara join kar sakte hain."
        )
        
        logger.info(f"User {target_user_id} unbanned by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Unban error: {e}")
        await update.effective_message.reply_text(
            "❌ Unban nahi ho paya! User pehle se unbanned ho sakta hai 😅"
        )


async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mute command - Mute replied user"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi user ke message ko reply karke /mute use karo! 😊\n"
            "Format: /mute <time> (e.g., 10m, 1h, 1d)"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    target_user = update.message.reply_to_message.from_user
    
    # Don't mute admins
    try:
        target_member = await context.bot.get_chat_member(update.effective_chat.id, target_user.id)
        if target_member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            await update.effective_message.reply_text(
                "❌ Admin ko mute nahi kar sakte! 😅"
            )
            return
    except:
        pass
    
    # Parse mute duration
    duration_seconds = 300  # Default 5 minutes
    duration_text = "5 minutes"
    
    if context.args:
        duration_arg = context.args[0].lower()
        try:
            if duration_arg.endswith('m'):
                minutes = int(duration_arg[:-1])
                duration_seconds = minutes * 60
                duration_text = f"{minutes} minute{'s' if minutes > 1 else ''}"
            elif duration_arg.endswith('h'):
                hours = int(duration_arg[:-1])
                duration_seconds = hours * 3600
                duration_text = f"{hours} hour{'s' if hours > 1 else ''}"
            elif duration_arg.endswith('d'):
                days = int(duration_arg[:-1])
                duration_seconds = days * 86400
                duration_text = f"{days} day{'s' if days > 1 else ''}"
        except:
            pass
    
    try:
        from datetime import datetime, timedelta
        
        # Mute the user (restrict permissions)
        until_date = datetime.now() + timedelta(seconds=duration_seconds)
        
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until_date
        )
        
        user_name = target_user.first_name or "User"
        await update.effective_message.reply_text(
            f"🔇 {user_name} ko {duration_text} ke liye mute kar diya! 🤐\n"
            "Unmute karne ke liye /unmute use karo."
        )
        
        logger.info(f"User {target_user.id} muted for {duration_text} by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Mute error: {e}")
        await update.effective_message.reply_text(
            "❌ Mute nahi ho paya! Permission issue ho sakta hai 😅"
        )


async def unmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unmute command - Unmute replied user"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi muted user ke message ko reply karke /unmute use karo! 😊"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    target_user = update.message.reply_to_message.from_user
    
    try:
        # Unmute the user (restore permissions)
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_polls=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
                can_change_info=False,
                can_invite_users=True,
                can_pin_messages=False
            )
        )
        
        user_name = target_user.first_name or "User"
        await update.effective_message.reply_text(
            f"🔊 {user_name} ko unmute kar diya! ✨\n"
            "Ab vo baat kar sakte hain."
        )
        
        logger.info(f"User {target_user.id} unmuted by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Unmute error: {e}")
        await update.effective_message.reply_text(
            "❌ Unmute nahi ho paya! User pehle se unmuted ho sakta hai 😅"
        )


async def promote_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /promote command - Promote replied user to admin"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi user ke message ko reply karke /promote use karo! 😊"
        )
        return
    
    # Check permissions (only creator or check if user is owner)
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    target_user = update.message.reply_to_message.from_user
    
    try:
        # Promote user to admin with basic permissions
        await context.bot.promote_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            can_change_info=False,
            can_delete_messages=True,
            can_invite_users=True,
            can_restrict_members=True,
            can_pin_messages=True,
            can_manage_chat=False
        )
        
        user_name = target_user.first_name or "User"
        await update.effective_message.reply_text(
            f"⭐ {user_name} ko admin bana diya! 🎉\n"
            "Congratulations! 👏"
        )
        
        logger.info(f"User {target_user.id} promoted by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Promote error: {e}")
        await update.effective_message.reply_text(
            "❌ Promote nahi ho paya! Permission issue ho sakta hai 😅\n"
            "Sirf group creator hi promote kar sakta hai!"
        )


async def demote_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /demote command - Remove admin rights"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi admin ke message ko reply karke /demote use karo! 😊"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    target_user = update.message.reply_to_message.from_user
    
    # Don't demote creator
    try:
        target_member = await context.bot.get_chat_member(update.effective_chat.id, target_user.id)
        if target_member.status == ChatMemberStatus.CREATOR:
            await update.effective_message.reply_text(
                "❌ Creator ko demote nahi kar sakte! 😅"
            )
            return
    except:
        pass
    
    try:
        # Demote user (remove admin rights)
        await context.bot.promote_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user.id,
            can_change_info=False,
            can_delete_messages=False,
            can_invite_users=False,
            can_restrict_members=False,
            can_pin_messages=False,
            can_manage_chat=False
        )
        
        user_name = target_user.first_name or "User"
        await update.effective_message.reply_text(
            f"⬇️ {user_name} ko demote kar diya! 😊\n"
            "Admin rights remove ho gaye."
        )
        
        logger.info(f"User {target_user.id} demoted by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Demote error: {e}")
        await update.effective_message.reply_text(
            "❌ Demote nahi ho paya! Permission issue ho sakta hai 😅\n"
            "Sirf group creator hi demote kar sakta hai!"
        )


async def pin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /pin command - Pin replied message"""
    await _register_user(update.effective_user.id)
    
    # Check if command is a reply
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "❌ Kisi message ko reply karke /pin use karo! 😊"
        )
        return
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    try:
        # Pin the message
        await context.bot.pin_chat_message(
            chat_id=update.effective_chat.id,
            message_id=update.message.reply_to_message.message_id,
            disable_notification=True  # Don't send notification to all members
        )
        
        await update.effective_message.reply_text(
            "📌 Message pin kar diya! ✨\n"
            "Unpin karne ke liye /unpin use karo."
        )
        
        logger.info(f"Message pinned by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Pin error: {e}")
        await update.effective_message.reply_text(
            "❌ Pin nahi ho paya! Permission issue ho sakta hai 😅"
        )


async def unpin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unpin command - Unpin message"""
    await _register_user(update.effective_user.id)
    
    # Check permissions
    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return
    
    try:
        # If replying to a message, unpin that specific message
        if update.message.reply_to_message:
            await context.bot.unpin_chat_message(
                chat_id=update.effective_chat.id,
                message_id=update.message.reply_to_message.message_id
            )
            await update.effective_message.reply_text(
                "📍 Message unpin kar diya! ✨"
            )
        else:
            # Unpin all messages
            await context.bot.unpin_all_chat_messages(
                chat_id=update.effective_chat.id
            )
            await update.effective_message.reply_text(
                "📍 Saare pinned messages unpin kar diye! ✨"
            )
        
        logger.info(f"Message(s) unpinned by {update.effective_user.first_name}")
    
    except Exception as e:
        logger.error(f"Unpin error: {e}")
        await update.effective_message.reply_text(
            "❌ Unpin nahi ho paya! Koi pinned message nahi hai shayad 😅"
        )


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /admin or /adminhelp command - Show admin commands (admin only)"""
    await _register_user(update.effective_user.id)
    
    # Must be in a group
    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text(
            "❌ Ye command sirf groups mein kaam karta hai! 😊"
        )
        return
    
    # Check if user is admin
    try:
        user_member = await context.bot.get_chat_member(
            update.effective_chat.id,
            update.effective_user.id
        )
        
        if user_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            await update.effective_message.reply_text(
                "Sirf admins is command ko use kar sakte hain 🙂"
            )
            return
    except Exception as e:
        logger.error(f"Admin check error: {e}")
        await update.effective_message.reply_text(
            "❌ Permission check mein problem aa gayi! 😅"
        )
        return
    
    # User is admin, show admin commands
    admin_help_text = (
        "👮‍♂️ *Admin Commands* - Baby ❤️\n\n"
        
        "🗑️ /del\n"
        "→ Reply karke message delete karo\n\n"
        
        "🚫 /ban\n"
        "→ Reply karke user ban karo\n\n"
        
        "🔓 /unban <user_id>\n"
        "→ Reply ya ID se unban karo\n\n"
        
        "🔇 /mute <time>\n"
        "→ Reply karke mute karo (10m, 1h, 1d)\n\n"
        
        "🔊 /unmute\n"
        "→ Reply karke mute hatao\n\n"
        
        "👑 /promote\n"
        "→ Reply karke admin banao\n\n"
        
        "👤 /demote\n"
        "→ Reply karke admin hatao\n\n"
        
        "📌 /pin\n"
        "→ Reply karke message pin karo\n\n"
        
        "📍 /unpin\n"
        "→ Reply karke unpin karo (ya sare pins hatao)\n\n"
        
        "⚠️ *Note:* Bot ko admin banana zaruri hai!\n"
        "Admins ko ban/mute nahi kar sakte 😊"
    )
    
    await update.effective_message.reply_text(
        admin_help_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Admin help shown to {update.effective_user.first_name}")


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
    
    # Check for "play <song name>" pattern
    message_lower = user_message.lower().strip()
    if message_lower.startswith("play ") and len(message_lower) > 5:
        song_name = user_message[5:].strip()  # Extract song name after "play "
        if song_name:
            # Simulate /song command by setting context.args and calling song_command
            context.args = song_name.split()
            await song_command(update, context)
            return
    
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
        if not update.message:
            return
            
        if not update.message.text:
            return
        
        message_text = update.message.text
        user_name = update.effective_user.first_name or "Unknown"
        chat_title = update.effective_chat.title or "Unknown Group"
        
        logger.debug(f"GROUP: [{chat_title}] {user_name}: {message_text[:50]}")
        
        # Check for spam FIRST (before any processing)
        spam_handled = await _check_spam(update, context)
        if spam_handled:
            logger.info(f"🚫 Spam detected and handled from {user_name}")
            return  # Spam detected and handled, don't process further
        
        # Register user
        user_id = update.effective_user.id
        username = update.effective_user.username
        first_name = update.effective_user.first_name or "User"
        await _register_user(user_id)
        
        # Register group
        group_id = update.effective_chat.id
        await _register_group(group_id, update.effective_chat)
        
        # Register group member
        await _register_group_member(group_id, user_id, username, first_name)
        
        message_text_lower = message_text.lower().strip()
        
        # Check for "play <song name>" pattern
        if message_text_lower.startswith("play ") and len(message_text_lower) > 5:
            song_name = message_text[5:].strip()  # Extract song name after "play "
            if song_name:
                # Simulate /song command by setting context.args and calling song_command
                context.args = song_name.split()
                await song_command(update, context)
                return
        
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
                    logger.info("✅ Trigger: Reply to bot")
        
        # Trigger 2: Bot mentioned (@AnimxClanBot or @animxclanbot)
        if "@animxclanbot" in message_text_lower or BOT_USERNAME.lower() in message_text_lower:
            should_respond = True
            bot_mentioned = True
            logger.info("✅ Trigger: Bot mentioned")
        
        # Trigger 3: Contains "baby"
        if "baby" in message_text_lower:
            should_respond = True
            logger.info("✅ Trigger: Word 'baby'")
        
        # Trigger 4: Basic greetings
        # These are exact word matches (case-insensitive)
        greetings = ["hi", "hii", "hello", "hey", "gm", "good morning", "gn", "good night", 
                     "bye", "good bye", "goodbye", "morning", "night"]
        
        # Split message into words and check for exact matches
        words = message_text_lower.split()
        for greeting in greetings:
            # Check if greeting is in the message as a standalone word or phrase
            if greeting in message_text_lower:
                # For multi-word greetings like "good morning"
                if " " in greeting:
                    if greeting in message_text_lower:
                        should_respond = True
                        logger.info(f"✅ Trigger: Greeting '{greeting}'")
                        break
                # For single-word greetings
                else:
                    if greeting in words:
                        should_respond = True
                        logger.info(f"✅ Trigger: Greeting '{greeting}'")
                        break
        
        # If NO trigger, IGNORE silently
        if not should_respond:
            logger.debug(f"⏭️ No trigger - ignoring message from {user_name}: {message_text[:30]}")
            return
        
        logger.info(f"🎯 RESPONDING to {user_name} in [{chat_title}]: {message_text[:50]}")
        
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
    
    # Log AI service configuration
    logger.info("=" * 50)
    logger.info("🤖 ANIMX CLAN Bot Starting...")
    logger.info("=" * 50)
    if OPENROUTER_API_KEY:
        logger.info(f"✅ OpenRouter: Enabled (Model: {OPENROUTER_MODEL})")
    else:
        logger.info("❌ OpenRouter: Disabled")
    
    if GEMINI_API_KEY and GEMINI_CLIENT:
        logger.info("✅ Gemini: Enabled (Fallback)")
    else:
        logger.info("❌ Gemini: Disabled")
    logger.info("=" * 50)
    
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
    application.add_handler(CommandHandler("admin", admin_command))
    application.add_handler(CommandHandler("adminhelp", admin_command))
    application.add_handler(CommandHandler("stop", stop_command))
    
    # Admin analytics commands
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("users", users_command))
    application.add_handler(CommandHandler("groups", groups_command))
    application.add_handler(CommandHandler("members", members_command))
    
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
    application.add_handler(CommandHandler("ga", ga_command))
    application.add_handler(CommandHandler("ge", ge_command))
    application.add_handler(CommandHandler("gn", gn_command))
    application.add_handler(CommandHandler("bye", bye_command))
    application.add_handler(CommandHandler("welcome", welcome_command))
    application.add_handler(CommandHandler("thanks", thanks_command))
    application.add_handler(CommandHandler("thankyou", thankyou_command))
    application.add_handler(CommandHandler("sorry", sorry_command))
    application.add_handler(CommandHandler("mood", mood_command))
    
    # Conversation commands
    application.add_handler(CommandHandler("chat", chat_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("privacy", privacy_command))
    
    # Emotional support commands
    application.add_handler(CommandHandler("sad", sad_command))
    application.add_handler(CommandHandler("happy", happy_command))
    application.add_handler(CommandHandler("angry", angry_command))
    application.add_handler(CommandHandler("motivate", motivate_command))
    application.add_handler(CommandHandler("howareyou", howareyou_command))
    application.add_handler(CommandHandler("missyou", missyou_command))
    application.add_handler(CommandHandler("hug", hug_command))
    
    # Productivity & wellness commands
    application.add_handler(CommandHandler("tip", tip_command))
    application.add_handler(CommandHandler("confidence", confidence_command))
    application.add_handler(CommandHandler("focus", focus_command))
    application.add_handler(CommandHandler("sleep", sleep_command))
    application.add_handler(CommandHandler("lifeline", lifeline_command))
    
    # Fun commands
    application.add_handler(CommandHandler("joke", joke_command))
    application.add_handler(CommandHandler("roast", roast_command))
    application.add_handler(CommandHandler("truth", truth_command))
    application.add_handler(CommandHandler("dare", dare_command))
    application.add_handler(CommandHandler("fact", fact_command))
    
    # Admin moderation commands
    application.add_handler(CommandHandler("del", del_command))
    application.add_handler(CommandHandler("ban", ban_command))
    application.add_handler(CommandHandler("unban", unban_command))
    application.add_handler(CommandHandler("mute", mute_command))
    application.add_handler(CommandHandler("unmute", unmute_command))
    application.add_handler(CommandHandler("promote", promote_command))
    application.add_handler(CommandHandler("demote", demote_command))
    application.add_handler(CommandHandler("pin", pin_command))
    application.add_handler(CommandHandler("unpin", unpin_command))
    
    # Register callback handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Register chat member handler for group tracking
    application.add_handler(ChatMemberHandler(my_chat_member_handler, ChatMemberHandler.MY_CHAT_MEMBER))
    
    # Register message handlers
    # Private messages (all text messages in private chat)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
            handle_private_message,
        )
    )
    
    # Group messages - @all mention handler (higher priority - checks for @all first)
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP) & filters.Regex(r'@all'),
            all_mention_handler,
        )
    )
    
    # Group messages - regular message handler (processes after @all check)
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
