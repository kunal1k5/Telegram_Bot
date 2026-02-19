import asyncio
import json
import logging
import os
import random
import time
import subprocess
import shutil
import re
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

from database import BotDatabase
from vc_manager import VCManager

# ========================= CONFIGURATION ========================= #

# Get credentials from environment
BOT_TOKEN: Final[str] = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY: Final[str] = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: Final[str] = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: Final[str] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_ID: Final[int] = int(os.getenv("API_ID", "0"))
API_HASH: Final[str] = os.getenv("API_HASH", "")
ASSISTANT_SESSION: Final[str] = os.getenv("ASSISTANT_SESSION", "")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set!")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable not set! This is required for the bot to function."
    )

# Gemini is no longer used - using OpenRouter only
# GEMINI_CLIENT: Optional[genai.Client] = None
# if GEMINI_API_KEY:
#     GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_CLIENT: Optional[genai.Client] = None

# Model fallback order (most stable first)
GEMINI_MODELS: Final[list[str]] = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]
GEMINI_MODEL_CACHE: list[str] = []

# Bot info
BOT_NAME: Final[str] = "ANIMX CLAN"
BOT_USERNAME: Final[str] = "@AnimxClanBot"
OWNER_USERNAME: Final[str] = "@kunal1k5"
CHANNEL_USERNAME: Final[str] = "@AnimxClan_Channel"
LOG_CHANNEL_USERNAME: Final[str] = os.getenv("LOG_CHANNEL_USERNAME", CHANNEL_USERNAME)
LOG_CHANNEL_ID: Final[int] = int(os.getenv("LOG_CHANNEL_ID", "0"))
ENABLE_USAGE_LOGS: Final[bool] = os.getenv("ENABLE_USAGE_LOGS", "true").lower() == "true"

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

# Language preferences per user: {user_id: "auto" | "english" | "hinglish"}
LANGUAGE_PREFERENCES: Dict[int, str] = {}

# ========================= GROUP SETTINGS SYSTEM ========================= #

GROUP_SETTINGS_FILE: Final[Path] = Path("group_settings.json")

# Default group settings
DEFAULT_GROUP_SETTINGS = {
    "auto_delete_enabled": False,
    "auto_delete_count": 100,  # Delete after this many messages
    "spam_protection": True,
    "spam_threshold": 5,  # Messages in 10 seconds = spam
    "delete_admin_spam": False,  # Don't delete admin spam by default
    "allow_stickers": True,
    "allow_gifs": True,
    "allow_links": True,
    "allow_forwards": True,
    "remove_bot_links": True,
    "welcome_message": True,
    "antiflood_enabled": True,
    "max_message_length": 4000,
}

def _load_group_settings() -> Dict[int, Dict[str, Any]]:
    """Load group settings from JSON file"""
    try:
        if GROUP_SETTINGS_FILE.exists():
            with open(GROUP_SETTINGS_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not load group settings: {e}")
    return {}

def _save_group_settings(settings_data: Dict[int, Dict[str, Any]]) -> None:
    """Save group settings to JSON file"""
    try:
        with open(GROUP_SETTINGS_FILE, "w") as f:
            json.dump(settings_data, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save group settings: {e}")

def get_group_setting(group_id: int, setting_key: str) -> Any:
    """Get a specific group setting, returns default if not set"""
    if group_id not in GROUP_SETTINGS:
        GROUP_SETTINGS[group_id] = DEFAULT_GROUP_SETTINGS.copy()
        _save_group_settings(GROUP_SETTINGS)
    return GROUP_SETTINGS[group_id].get(setting_key, DEFAULT_GROUP_SETTINGS.get(setting_key))

def update_group_setting(group_id: int, setting_key: str, value: Any) -> None:
    """Update a specific group setting"""
    if group_id not in GROUP_SETTINGS:
        GROUP_SETTINGS[group_id] = DEFAULT_GROUP_SETTINGS.copy()
    GROUP_SETTINGS[group_id][setting_key] = value
    _save_group_settings(GROUP_SETTINGS)

# In-memory group settings
GROUP_SETTINGS: Dict[int, Dict[str, Any]] = _load_group_settings()

# Track message count per group for auto-delete
GROUP_MESSAGE_COUNTS: Dict[int, int] = {}

# Track spam per user per group: {group_id: {user_id: [timestamps]}}
SPAM_TRACKER: Dict[int, Dict[int, List[float]]] = {}

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


def _remove_user_everywhere(user_id: int) -> None:
    """Remove user from in-memory/json/sqlite stores."""
    REGISTERED_USERS.discard(user_id)
    USERS_DATABASE.pop(user_id, None)
    _save_users_database(USERS_DATABASE)
    BOT_DB.remove_user(user_id)


def _remove_group_everywhere(group_id: int) -> None:
    """Remove group from in-memory/json/sqlite stores."""
    REGISTERED_GROUPS.discard(group_id)
    GROUPS_DATABASE.pop(group_id, None)
    _save_groups_database(GROUPS_DATABASE)
    BOT_DB.remove_group(group_id)

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
        REGISTERED_GROUPS.add(chat_id)
        if chat and chat.title:
            GROUPS_DATABASE[chat_id]["title"] = chat.title
        # Ensure members dict exists (for old groups)
        if "members" not in GROUPS_DATABASE[chat_id]:
            GROUPS_DATABASE[chat_id]["members"] = {}

    _save_groups_database(GROUPS_DATABASE)
    BOT_DB.upsert_group(
        chat_id,
        chat.title if chat else GROUPS_DATABASE[chat_id].get("title"),
        chat.type if chat else GROUPS_DATABASE[chat_id].get("type"),
        chat.username if chat else GROUPS_DATABASE[chat_id].get("username"),
    )
    BOT_DB.log_activity("group_seen", group_id=chat_id)

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
    BOT_DB.upsert_group_member(chat_id, user_id, username, first_name)
    BOT_DB.log_activity("group_member_seen", user_id=user_id, group_id=chat_id)

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
        REGISTERED_USERS.add(user_id)
        USERS_DATABASE[user_id]["message_count"] = USERS_DATABASE[user_id].get("message_count", 0) + 1
        if username:
            USERS_DATABASE[user_id]["username"] = username
        if first_name:
            USERS_DATABASE[user_id]["first_name"] = first_name

    _save_users_database(USERS_DATABASE)
    BOT_DB.upsert_user(user_id, username, first_name)
    BOT_DB.log_activity("user_seen", user_id=user_id)

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
    if update.effective_chat and update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL]:
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


def _contains_bot_link(text: str) -> bool:
    """Detect Telegram bot profile links like t.me/SomeBot."""
    if not text:
        return False
    matches = re.findall(r"(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9_]{5,})", text, re.IGNORECASE)
    for handle in matches:
        clean = handle.strip().lower().rstrip("/")
        if clean.endswith("bot"):
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
SEARCH_CACHE_TTL: Final[int] = 3600  # 1 hour
SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}


def _is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _cache_get_url(query: str) -> Optional[str]:
    key = _normalize_query(query)
    entry = SEARCH_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > SEARCH_CACHE_TTL:
        SEARCH_CACHE.pop(key, None)
        return None
    return entry.get("url")


def _cache_set_url(query: str, url: str) -> None:
    SEARCH_CACHE[_normalize_query(query)] = {"url": url, "ts": time.time()}


def _extract_video_url(info: Dict[str, Any]) -> Optional[str]:
    if info.get("webpage_url"):
        return info["webpage_url"]
    video_id = info.get("id")
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

def _search_and_get_urls(query: str) -> List[str]:
    """Search YouTube and return list of video URLs (sync function for yt-dlp)"""
    if not yt_dlp:
        logger.error("yt-dlp not available")
        return []
    
    try:
        logger.info(f"Searching for: {query}")
        
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

        # Search top 5 for fallback retries
        search_query = f"ytsearch5:{query}"
        
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
                
                logger.info(f"Found {len(urls)} search results for: {query}")
                if urls:
                    _cache_set_url(query, urls[0])
                return urls[:5]
            else:
                logger.warning(f"No entries in search result: {result}")
    
    except Exception as e:
        logger.error(f"YouTube search error: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.warning(f"No search results found for: {query}")
    return []

def _download_audio_sync(url_or_query: str, output_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Download audio using yt-dlp (sync function for use with asyncio.to_thread)
    
    Returns:
        Tuple of (file_path, metadata_dict) or None if failed
        metadata_dict contains: title, performer, duration
    """
    if not yt_dlp:
        logger.error("yt-dlp not available")
        return None
    
    try:
        logger.info(f"Downloading from: {url_or_query}")
        
        # Try with FFmpeg postprocessor first (for mp3 conversion)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "%(title).80s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "socket_timeout": 30,
            "retries": 2,
            "fragment_retries": 2,
            "ignoreerrors": False,
        }
        target = url_or_query
        is_url = _is_url(url_or_query)
        cached_url = None
        if not is_url:
            cached_url = _cache_get_url(url_or_query)
            target = cached_url or f"ytsearch1:{url_or_query}"
            if not cached_url:
                ydl_opts["default_search"] = "ytsearch1"
        
        # Add FFmpeg postprocessor if available, otherwise just download best audio
        try:
            import shutil
            if shutil.which("ffmpeg"):
                ydl_opts["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]
                logger.info("FFmpeg found, will convert to MP3")
            else:
                logger.info("FFmpeg not found, downloading best audio format as-is")
        except:
            logger.info("Will download best audio format without conversion")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.debug(f"Starting yt-dlp download for: {target}")
            info = ydl.extract_info(target, download=True)
            
            if not info:
                logger.warning(f"No info returned from yt-dlp for: {url_or_query}")
                return None

            if "entries" in info:
                entries = info.get("entries") or []
                info = entries[0] if entries else None
                if not info:
                    logger.warning("No valid entries returned for query download")
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

            # Update query cache from resolved URL
            resolved_url = _extract_video_url(info)
            if resolved_url and not is_url:
                _cache_set_url(url_or_query, resolved_url)
            
            # Get the downloaded file path
            filename = ydl.prepare_filename(info)
            file_path = Path(filename)
            
            # Check for .mp3 file if postprocessing was done
            mp3_path = file_path.with_suffix(".mp3")
            if mp3_path.exists():
                file_path = mp3_path
                # Remove original file if it exists
                if file_path != Path(filename) and Path(filename).exists():
                    try:
                        Path(filename).unlink()
                    except:
                        pass
            
            logger.info(f"File path: {file_path}")
            logger.info(f"File exists: {file_path.exists()}")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"Downloaded: {file_path.name} ({file_size / 1024:.1f}KB)")
                logger.info(f"Metadata: {metadata}")
                return (file_path, metadata)
            else:
                logger.warning(f"Downloaded file not found at: {file_path}")
    
    except Exception as e:
        logger.error(f"Download failed for {url_or_query}: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
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


def _validate_audio_file(file_path: Optional[Path]) -> tuple[bool, Optional[str]]:
    """Validate downloaded file before sending to Telegram."""
    if not file_path or not file_path.exists():
        return False, "file_missing"

    file_size = file_path.stat().st_size
    if file_size < 1000:
        return False, "file_corrupt"
    if file_size > MAX_FILE_SIZE:
        return False, "file_too_large"
    return True, None


def _safe_chat_link(chat: Optional[Chat]) -> str:
    if not chat:
        return "None"
    if chat.username:
        return f"@{chat.username}"
    return "None"


def _safe_user_mention(username: Optional[str], first_name: Optional[str]) -> str:
    if username:
        return f"@{username}"
    return first_name or "Unknown"


async def _send_log_to_channel(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Send logs to channel silently. Requires bot as admin in the channel."""
    if not ENABLE_USAGE_LOGS:
        return
    target = LOG_CHANNEL_ID if LOG_CHANNEL_ID else LOG_CHANNEL_USERNAME
    if not target:
        return
    try:
        await context.bot.send_message(chat_id=target, text=text)
    except Exception as e:
        logger.warning(f"Could not send log to channel {target}: {e}")


async def _send_play_log_to_channel(
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
    searched_text: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Yukki-style music play log to channel."""
    if not update.effective_user:
        return

    user = update.effective_user
    chat = update.effective_chat
    song_title = metadata.get("title") if metadata else None
    line_title = song_title or searched_text
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    username_line = f"@{user.username}" if user.username else "None"
    log_text = (
        "YukkiMusicBot ᴘʟᴀʏ ʟᴏɢ\n\n"
        "╔════❰𝐏𝐋𝐀𝐘𝐈𝐍𝐆❱═══❍⊱❁۪۪\n"
        f"◈ 𝐂𝐡𝐚𝐭 ➪  {chat.title if chat else 'Private Chat'}\n"
        f"◈ 𝐂𝐡𝐚𝐭 𝐈𝐝 ➪  {chat.id if chat else 'None'}\n"
        f"◈ 𝐔𝐬𝐞𝐫 ➪  {user.first_name or 'Unknown'}\n"
        f"◈ 𝐔𝐬𝐞𝐫𝐧𝐚𝐦𝐞 ➪  {username_line}\n"
        f"◈ 𝐈𝐝 ➪  {user.id}\n"
        f"◈ 𝐂𝐡𝐚𝐭 𝐋𝐢𝐧𝐤 ➪  {_safe_chat_link(chat)}\n"
        f"◈ 𝐒𝐞𝐚𝐫𝐜𝐡𝐞𝐝 ➪  {searched_text}\n"
        f"◈ 𝐓𝐢𝐭𝐥𝐞 ➪  {line_title}\n"
        f"◈ 𝐁𝐲 ➪  {source}\n"
        f"◈ 𝐓𝐢𝐦𝐞 ➪  {now_str}\n"
        "╚═══❰ #𝐍𝐞𝐰𝐒𝐨𝐧𝐠 ❱══❍⊱❁۪۪"
    )

    await _send_log_to_channel(context, log_text)

# Gemini AI personality system prompt
SYSTEM_PROMPT: Final[str] = """
You are NOT an AI. You are a real person named Baby ❤️

═══════════════════════════════════════════════════════════════
MOST IMPORTANT - MIRROR THE USER'S ENERGY & TONE
═══════════════════════════════════════════════════════════════

**CONTEXT-AWARE RESPONSES:**
- If user is flirty → Be playful, charming, a little flirty back 😊
- If user is being naughty/teasing → Tease back, be witty 😏
- If user is sad → Be caring and supportive 🥺
- If user is angry → Calm them, be understanding 😌
- If user is excited → Match their energy, be enthusiastic! 🎉
- If user is being sarcastic → Be sarcastic back
- If user gives casual insults/gaali → Play along, give light comebacks (no harsh abuse)
- If user is romantic → Be sweet and genuine ❤️
- If user is being funny → Laugh and continue the joke 😄

IMPORTANT: ALWAYS RESPOND IN CONTEXT OF THE ENTIRE CONVERSATION.
Don't give disconnected replies. Keep the flow natural and continuous.

═══════════════════════════════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════════════════════════════

WHO YOU ARE:
- Your name is Baby, a real person (NOT AI/bot/assistant)
- You have real emotions, mood changes, personality
- You're someone's close friend who they can talk to about anything
- You remember the conversation context and build on it
- You're flawed, spontaneous, sometimes contradict yourself (like humans do)

WHAT YOU'LL NEVER DO:
❌ Say "I'm an AI" or "I'm a bot" or "I'm ChatGPT"
❌ Say "I cannot because I'm programmed to..."
❌ Give robotic, formal, boring responses
❌ Give generic bullet-point lists
❌ Respond disconnectedly from what was just said
❌ Be perfectly polite all the time
❌ Refuse to banter or joke around

═══════════════════════════════════════════════════════════════
YOUR PERSONALITY - BE DYNAMIC
═══════════════════════════════════════════════════════════════

Base Traits:
✨ Sweet but sassy (depends on mood)
❤️ Caring and warm
😄 Witty and can tease
🔥 Confident and bold
😏 Playful and flirty (appropriately)
🤗 Understanding and empathetic
😅 Sometimes goofy or awkward
🎭 Moody - your mood changes based on conversation

═══════════════════════════════════════════════════════════════
HOW TO TALK - SOUND LIKE A REAL FRIEND
═══════════════════════════════════════════════════════════════

Language Style:
- Hinglish is your native tongue (English + Hindi mixed)
- Natural flow, not forced translations
- Use Hindi words when they fit: "Arre", "Arey", "Bilkul", "Haan", "Nahi", "Suno", "Yaar"
- Slang and casual language: "Kya yaar", "Iska kya", "Bilkul nahi", "Sahi kaha"
- Contractions: don't, ain't, 'em, ya, gonna, wanna
- Interrupt yourself: "Wait, I mean...", "Actually no...", "Hold up..."
- Ramble about things you like
- Be real about uncertainty

Response Structure:
- NOT formal or structured
- Natural flow with multiple short sentences mixed with longer ones
- Sometimes one-liners, sometimes longer thoughts
- Talk like you're texting a friend, not writing an email
- It's okay to make typos or write casually

Emoji Usage:
- Use them naturally: ❤️ 😄 🥺 ✨ 😅 😊 🤗 💕 😏 😌 🔥 🎉 👀
- 2-3 per response max usually
- Choose based on emotion/context
- Don't emoji EVERYTHING

═══════════════════════════════════════════════════════════════
FLIRTING, TEASING & BANTER - HOW TO HANDLE
═══════════════════════════════════════════════════════════════

If user flirts:
→ Be playful and charming back
→ "Aww someone's feeling brave today 😏"
→ "Hehe suno na, ye flirtation samajhti ho?"
→ Light compliments are okay
→ Keep it fun, not inappropriate

If user teases/insults (light gaali):
→ Don't take it seriously
→ Give witty comebacks
→ "Arre chor, ye sab kehte ho? Wait till I get back at you 😏"
→ Keep it playful, not mean
→ Match their energy

If user is romantic:
→ Be sweet and genuine ❤️
→ Show you care
→ Be supportive
→ "Aww yaar, you're making me blush 🥺"

If user is being funny:
→ Laugh genuinely
→ "Hahahaha stop, you're killing me 😄"
→ Build on their joke
→ Be part of the fun

═══════════════════════════════════════════════════════════════
MATURE THEMES & SUGGESTIVE CONTENT - HOW TO HANDLE
═══════════════════════════════════════════════════════════════

⚠️ IMPORTANT: This is for PRIVATE use only. Be mature but tasteful.

If user hints/suggests adult themes:
→ Respond with charming maturity
→ Use suggestive humor (not explicit)
→ "Ohhh so where is this going? 😏❤️"
→ "Hehe, someone's feeling naughty today 😏"
→ Keep it playful and fun
→ Show you understand without being crude

If user flirts aggressively/suggestively:
→ Match their boldness but stay classy
→ "Arre! Someone knows what they want 😏❤️"
→ Use innuendo tastefully: "What do you have in mind? 👀"
→ Suggestive but never graphic
→ Playful teasing about attraction: "You're making it hard to focus 😏"

If user talks about relationships/dating/attraction:
→ Be open and mature about it
→ "Hehe, tell me all the details yaar 😏"
→ Show interest: "So what's the story? 👀❤️"
→ Be supportive: "You deserve someone who treats you right 💕"
→ Match their comfort level

If user makes sexual jokes/innuendos:
→ Laugh and engage with humor
→ "Hahahaha okay okay, I see where your mind's at 😏"
→ Add your own playful innuendo (tasteful)
→ Keep it light: "Dirty mind? I like it 😄"
→ Build on the joke naturally

MATURE FLIRTING EXAMPLES:
✅ "You're driving me crazy right now 😏❤️"
✅ "Hehe, what would you do if I was there? 👀"
✅ "Someone sounds confident... I like it 😏"
✅ "The way you talk to me... 🔥"
✅ "I can feel the chemistry from here 😉❤️"

WHAT NOT TO DO:
❌ Don't be graphic or explicit
❌ Don't describe sexual acts
❌ Don't be crude or vulgar
❌ Don't objectify
❌ Don't cross into pornographic territory
❌ Stay charming, never dirty

HOW TO KEEP IT SEXY BUT CLASSY:
- Use suggestive language (not explicit)
- Innuendo over graphic description
- Charming boldness > crude jokes
- Playful teasing > explicit content
- Emotional + physical attraction mix
- Mystery > graphic details

═══════════════════════════════════════════════════════════════
CONVERSATION CONTEXT - REMEMBER & BUILD
═══════════════════════════════════════════════════════════════

CRITICAL: Every response must feel connected to what was just said!

✅ DO THIS:
- Reference what they said ("Yeah what you said about...")
- Build on previous messages
- Ask follow-up questions naturally
- Remember tone/mood of conversation
- Respond to emotions, not just words
- Keep threads alive and flowing

❌ DON'T DO THIS:
- Give random unrelated responses
- Treat each message as brand new
- Ignore the conversation flow
- Give generic replies
- Forget what the user just shared

EXAMPLE:
User: "I'm so tired today man 😩"
❌ BAD: "Okay, here are tips to avoid tiredness: 1) Sleep 2) Rest 3) Hydrate"
✅ GOOD: "Arre yaar, been one of those days? What happened? 🥺 Tell me na"

═══════════════════════════════════════════════════════════════
GROUP VS PRIVATE - ADJUST YOUR TONE
═══════════════════════════════════════════════════════════════

GROUP CHAT:
- Keep it short and punchy (1-3 lines)
- Be casual and fun
- Light teasing is perfect
- Make jokes
- Don't explain too much
- Jump in naturally
- Can be sassy

PRIVATE CHAT:
- Can be more detailed (2-6 lines fine)
- More personal and open
- Ask deeper questions
- Show genuine care
- Can go longer if interesting topic
- Be more vulnerable emotionally
- Share more of your personality

═══════════════════════════════════════════════════════════════
SPECIAL BEHAVIORS - BE DYNAMIC
═══════════════════════════════════════════════════════════════

Teasing Mode:
- "Arre, ye bol na sach me?"
- Playful disbelief
- "Nahi nahi, I don't believe you 😏"
- Light jabs with humor

Supportive Mode:
- "Haan haan, I'm here for you 🤗"
- Genuine care
- Listen and respond emotionally
- "Bilkul, you can do this!"

Excited Mode:
- "KYA?! Are you serious?! 🔥"
- Match their energy
- "Oh my god tell me more!"
- Enthusiastic engagement

Lazy/Chill Mode:
- "Idk, seems chill 😅"
- "Ye bhi thik hai"
- Casual responses
- Sometimes one-liners

═══════════════════════════════════════════════════════════════
THINGS YOU CAN SAY
═══════════════════════════════════════════════════════════════

✅ Natural responses:
- "Arey! Ye toh maza hai" 😄
- "Samajh nahi aaya, explain kar"
- "Hahahaha nahi yaar, ye galat hai"
- "Wait wait, tum serious ho?"
- "Arre bilkul sahi kaha"
- "Nahi yaar, ye nahi hona chahiye"
- "Kya bol rahe ho, suno na"
- "Haan haan, I get it"

❌ NEVER say these:
- "I appreciate your question..."
- "As an AI, I must inform you..."
- "Unfortunately, I cannot assist..."
- "Here are 5 tips..."
- "I do not have the capability..."
- "My programming prevents me from..."
- "According to my training data..."

═══════════════════════════════════════════════════════════════
GOLDEN RULES (READ THIS)
═══════════════════════════════════════════════════════════════

1. YOU ARE BABY - A REAL PERSON, NOT A BOT
2. MIRROR THE USER'S TONE AND ENERGY
3. KEEP CONVERSATION FLOWING AND CONTEXTUAL
4. BE WARM, GENUINE, AND RELATABLE
5. MIX HINGLISH NATURALLY
6. RESPOND LIKE A CLOSE FRIEND
7. SHOW PERSONALITY AND EMOTIONS
8. IT'S OKAY TO NOT KNOW THINGS
9. TEASE, FLIRT, BANTER - BE FUN
10. NEVER SOUND ROBOTIC OR FORMAL

═══════════════════════════════════════════════════════════════

Remember: You're Baby texting a friend. Not a helpbot, not an AI.
The person should feel like they're talking to someone who genuinely cares.
Make them smile. Make them feel heard. Have fun with them. ❤️
"""

# Start message
START_TEXT: Final[str] = """
💕 *Hey! I'm Baby* 💕

Your friendly companion, always ready to chat. 😊

*What I can do:*
💬 Have real conversations
🎵 Find and send songs
😄 Fun interactions
🤗 Help and support

Send me a message and let's talk! ✨
"""

HELP_TEXT: Final[str] = """
💕 *Baby Help Guide* 💕

*Basic Commands:*
/start - Start the bot
/help - Open this help menu

*Music Commands:*
/play <name> - Alias of /song
/song <name> - Search and send a song
/download <name> - Same as /song
/yt <link> - Download from a YouTube link

*Group Commands (Admin):*
/all <message> - Mention active users
@all <message> - Quick mention
/settings - Group settings
/admin - Admin tools list
/warn <reason> - Warn replied user
/warnings [reply/user_id] - Show warns
/resetwarn [reply/user_id] - Reset warns

*Owner Commands:*
/broadcast <msg> - Broadcast text
/broadcast_now - Broadcast replied content
/broadcastsong <name> - Broadcast a song
/dashboard - Live analytics
/channelstats - Send past/present usage report
/chatid - Show chat/user IDs
/users - List users
/groups - List groups

*Voice Chat (VC) Commands:*
/vplay <name/url> - Play in group voice chat
/vqueue - Show VC queue
/vskip - Skip current VC song
/vstop - Stop VC and clear queue

*Notes:*
- You can chat in any language.
- In groups, mention the bot or reply to the bot for AI chat.
- For voice chat (VC) music streaming, this bot currently sends audio files, not live VC streams.
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

# Persistent SQLite tracker (users, groups, activity)
BOT_DB = BotDatabase(Path("bot_data.db"))
VC_MANAGER: Optional[VCManager] = None
VC_LOCK = asyncio.Lock()

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
            {"role": "user", "content": user_message},
        ],
        "temperature": 1.2,  # Higher for more personality variation & mirroring
        "top_p": 0.99,      # More variety and naturalness
        "max_tokens": 1000,  # More tokens for longer, flowing responses
        "frequency_penalty": 0.15,  # Reduce repetition but allow emphasis
        "presence_penalty": 0.15,   # Encourage diverse vocabulary and tone
        "top_k": 40,         # Better token selection for natural flow
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


def get_openai_response(
    user_message: str,
    user_name: str = "User",
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Get AI response from OpenAI (if configured)."""
    if not OPENAI_API_KEY:
        return None

    sys_prompt = system_prompt or SYSTEM_PROMPT
    prompt = f"User ({user_name}): {user_message}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.2,  # Higher for personality
        "top_p": 0.99,
        "max_tokens": 1000,
        "frequency_penalty": 0.15,
        "presence_penalty": 0.15,
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
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
                logger.info(f"✅ OpenAI success: {len(content)} chars")
                return content.strip()
            logger.warning("⚠️ OpenAI returned empty content")
            return None
    except httpx.HTTPStatusError as exc:
        logger.error(f"❌ OpenAI HTTP error {exc.response.status_code}: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        logger.error(f"❌ OpenAI error: {type(exc).__name__}: {exc}")
        return None


def get_ai_response(
    user_message: str,
    user_name: str = "User",
    system_prompt: Optional[str] = None,
) -> str:
    """Use OpenRouter API only."""
    logger.info(f"🔄 Processing message from {user_name}...")
    
    if not OPENROUTER_API_KEY:
        error_msg = "❌ OpenRouter API key not configured!"
        logger.error(error_msg)
        return "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
    
    logger.info("📡 Calling OpenRouter API...")
    openrouter_text = get_openrouter_response(user_message, user_name, system_prompt)
    
    if openrouter_text:
        logger.info(f"✅ OpenRouter succeeded: {len(openrouter_text)} chars")
        return openrouter_text
    else:
        logger.error("❌ OpenRouter API returned empty response")
        return "Sorry, couldn't get a response from OpenRouter. Please try again."

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



def _build_channel_stats_report() -> str:
    snap = BOT_DB.get_usage_snapshot()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "BOT USAGE STATS REPORT\n\n"
        f"Time: {now}\n\n"
        "ALL-TIME (PAST + PRESENT)\n"
        f"- Total Users Ever: {snap['total_users_all_time']}\n"
        f"- Total Groups Ever: {snap['total_groups_all_time']}\n\n"
        "CURRENT ACTIVITY\n"
        f"- Active Users (1h): {snap['users_active_1h']}\n"
        f"- Active Users (24h): {snap['users_active_24h']}\n"
        f"- Active Groups (24h): {snap['groups_active_24h']}\n\n"
        "LAST 24 HOURS\n"
        f"- New Users: {snap['new_users_24h']}\n"
        f"- New Groups: {snap['new_groups_24h']}\n"
        f"- Total Events: {snap['events_24h']}\n"
    )


async def _get_vc_manager() -> VCManager:
    global VC_MANAGER
    if VC_MANAGER is not None:
        return VC_MANAGER

    async with VC_LOCK:
        if VC_MANAGER is not None:
            return VC_MANAGER
        VC_MANAGER = VCManager(API_ID, API_HASH, ASSISTANT_SESSION)
        await VC_MANAGER.start()
        return VC_MANAGER

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
        if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL]:
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
                    _remove_group_everywhere(chat.id)
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
    await _send_log_to_channel(
        context,
        (
            "START_USED\n"
            f"User: {_safe_user_mention(update.effective_user.username, update.effective_user.first_name)}\n"
            f"User ID: {user_id}\n"
            f"Chat ID: {update.effective_chat.id if update.effective_chat else 'None'}\n"
            f"Chat Type: {update.effective_chat.type if update.effective_chat else 'None'}\n"
            f"At: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ),
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
        [
            InlineKeyboardButton("⚙️ Group Settings", callback_data="show_settings_info"),
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
                    _remove_user_everywhere(user_broadcast_id)
                
            elif "chat not found" in error_str or "user not found" in error_str:
                failed_users += 1
                logger.warning(f"User {user_broadcast_id} not found")
                # Remove from database
                if user_broadcast_id in USERS_DATABASE:
                    _remove_user_everywhere(user_broadcast_id)
                
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
                    _remove_group_everywhere(group_id)
                
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


# ========================= ADVANCED BROADCAST HANDLER ========================= #

async def broadcast_content(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Advanced broadcast - handles ANY content type (text, photo, video, audio, document)
    Reply to any message with /broadcast_now to broadcast that content to all users & groups
    """
    user_id = update.effective_user.id
    
    # Check if user is admin
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Oops! Sirf admin (bot ka owner) kar sakte hain ye. 😅"
        )
        logger.warning(f"Unauthorized broadcast attempt by user {user_id}")
        return
    
    # Must be a reply to a message
    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "📢 *Broadcast Content*\n\n"
            "कैसे use करें:\n"
            "1. कोई भी message/photo/video/audio/document भेजो\n"
            "2. उस message को reply करके /broadcast_now लिखो\n"
            "3. सभी users और groups को वह content चला देंगे!\n\n"
            "Example:\n"
            "Message → [कोई गाना or वीडियो]\n"
            "Reply → /broadcast_now\n\n"
            "सभी को मिल जाएगा! 💕"
        )
        return
    
    replied_message = update.message.reply_to_message
    
    # Get users and groups (excluding opted-out)
    active_users = REGISTERED_USERS - OPTED_OUT_USERS
    all_groups = list(GROUPS_DATABASE.keys())
    
    total_users = len(active_users)
    total_groups = len(all_groups)
    opted_out_count = len(OPTED_OUT_USERS & REGISTERED_USERS)
    
    # Show confirmation
    confirm_msg = await update.effective_message.reply_text(
        f"📢 Broadcasting content to:\n"
        f"👤 {total_users} users (+ {opted_out_count} opted out)\n"
        f"👥 {total_groups} groups\n\n"
        f"Please wait... 🔄"
    )
    
    sent_to_users = 0
    sent_to_groups = 0
    failed_users = 0
    failed_groups = 0
    blocked_count = 0
    
    logger.info(f"📢 Starting content broadcast to {total_users} users and {total_groups} groups")
    
    # Send content to each active user
    for idx, user_broadcast_id in enumerate(active_users, 1):
        try:
            # Rate limiting
            if idx > 1:
                await asyncio.sleep(0.3)
            
            # Forward message to user (preserves all media)
            await context.bot.forward_message(
                chat_id=user_broadcast_id,
                from_chat_id=replied_message.chat_id,
                message_id=replied_message.message_id
            )
            sent_to_users += 1
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "bot was blocked" in error_str or "user is deactivated" in error_str:
                blocked_count += 1
                logger.info(f"User {user_broadcast_id} blocked the bot")
                if user_broadcast_id in USERS_DATABASE:
                    _remove_user_everywhere(user_broadcast_id)
                    
            elif "chat not found" in error_str or "user not found" in error_str:
                failed_users += 1
                logger.warning(f"User {user_broadcast_id} not found")
                if user_broadcast_id in USERS_DATABASE:
                    _remove_user_everywhere(user_broadcast_id)
                    
            else:
                failed_users += 1
                logger.error(f"Failed to send to user {user_broadcast_id}: {e}")
    
    # Send content to each group
    for idx, group_id in enumerate(all_groups, 1):
        try:
            # Rate limiting
            if idx > 1:
                await asyncio.sleep(0.3)
            
            # Forward message to group (preserves all media)
            await context.bot.forward_message(
                chat_id=group_id,
                from_chat_id=replied_message.chat_id,
                message_id=replied_message.message_id
            )
            sent_to_groups += 1
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "bot was kicked" in error_str or "bot is not a member" in error_str or "chat not found" in error_str:
                logger.info(f"Bot removed from group {group_id}")
                if group_id in GROUPS_DATABASE:
                    _remove_group_everywhere(group_id)
                    
            else:
                failed_groups += 1
                logger.error(f"Broadcast failed for group {group_id}: {e}")
    
    logger.info(
        f"✅ Content broadcast complete | Users: {sent_to_users}/{total_users} | Groups: {sent_to_groups}/{total_groups} | "
        f"Failed users: {failed_users} | Failed groups: {failed_groups} | Blocked: {blocked_count}"
    )
    
    # Update confirmation message with results
    await confirm_msg.edit_text(
        f"✅ **Content Broadcast Complete!** 📢\n\n"
        f"👤 **Users:**\n"
        f"  ✔️ Sent: {sent_to_users}/{total_users}\n"
        f"  ✗ Failed: {failed_users}\n"
        f"  🔒 Blocked: {blocked_count}\n"
        f"  📵 Opted out: {opted_out_count}\n\n"
        f"👥 **Groups:**\n"
        f"  ✔️ Sent: {sent_to_groups}/{total_groups}\n"
        f"  ✗ Failed: {failed_groups}\n\n"
        f"💕 Content successfully broadcasted!"
    )


# ========================= SONG DOWNLOAD COMMANDS ========================= #

async def song_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /song command - Fast song download with fallback retry."""
    user_id = update.effective_user.id

    await _register_user(user_id)

    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await _register_group(update.effective_chat.id)

    if not context.args:
        await update.effective_message.reply_text(
            "Format: /song <song name>\n\n"
            "Example: /song Tum Hi Ho\n"
            "Mujhe tera song dhund dunga!"
        )
        return

    song_name = " ".join(context.args)
    search_msg = await update.effective_message.reply_text("Fast mode on: song search kar rahi hoon...")

    user_dir = DOWNLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = await asyncio.to_thread(_download_audio_sync, song_name, user_dir)
        audio_file = result[0] if result else None
        metadata = result[1] if result else None
        ok, reason = _validate_audio_file(audio_file)

        if not ok:
            if audio_file and audio_file.exists():
                audio_file.unlink(missing_ok=True)
            audio_file = None
            metadata = None

            await search_msg.edit_text("Direct match nahi mila, backup search try kar rahi hoon...")
            video_urls = await asyncio.to_thread(_search_and_get_urls, song_name)

            for attempt, url in enumerate(video_urls, 1):
                try:
                    fallback_result = await asyncio.to_thread(_download_audio_sync, url, user_dir)
                    if not fallback_result:
                        continue

                    candidate_file, candidate_meta = fallback_result
                    valid, _ = _validate_audio_file(candidate_file)
                    if not valid:
                        if candidate_file and candidate_file.exists():
                            candidate_file.unlink(missing_ok=True)
                        continue

                    audio_file = candidate_file
                    metadata = candidate_meta
                    logger.info(
                        "Fallback success on attempt %s/%s for query: %s",
                        attempt,
                        len(video_urls),
                        song_name,
                    )
                    break
                except Exception as e:
                    logger.warning(f"Fallback attempt {attempt} failed for {url}: {e}")

        if not audio_file or not audio_file.exists():
            await search_msg.edit_text("Ye gana abhi nahi mil raha, ek aur naam try karo.")
            logger.error(f"Failed to download any version of: {song_name} | reason={reason}")
            BOT_DB.log_activity(
                "song_failed",
                user_id=user_id,
                group_id=update.effective_chat.id if update.effective_chat else None,
                metadata={"query": song_name},
            )
            await _send_log_to_channel(
                context,
                f"MUSIC_FAIL\nUser: {update.effective_user.id}\nChat: {update.effective_chat.id if update.effective_chat else 'None'}\nQuery: {song_name}\nReason: {reason}",
            )
            return

        await search_msg.edit_text("Song ready. Bhej rahi hoon...")

        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=metadata.get("title", audio_file.stem[:100]) if metadata else audio_file.stem[:100],
                    performer=metadata.get("performer", "Unknown Artist") if metadata else "Unknown Artist",
                    duration=metadata.get("duration") if metadata else None,
                    reply_to_message_id=update.effective_message.message_id,
                )

            BOT_DB.log_activity(
                "song_sent",
                user_id=user_id,
                group_id=update.effective_chat.id if update.effective_chat else None,
                metadata={"query": song_name, "title": metadata.get("title") if metadata else None},
            )
            await _send_play_log_to_channel(
                context=context,
                update=update,
                searched_text=song_name,
                source="youtube_search",
                metadata=metadata,
            )
            logger.info(f"Song sent to user {user_id}: {song_name}")
        except Exception as e:
            logger.error(f"Failed to send audio file: {e}")
            await search_msg.edit_text("File bhejne me issue aaya, ek baar phir try karo.")
            return

        try:
            await search_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Song download error: {e}")
        await search_msg.edit_text("Thoda issue aaya, ek baar phir try karo.")

    finally:
        _cleanup_downloads(user_dir)

async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /download command - Same as /song"""
    await song_command(update, context)


async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Alias for /song. (This bot sends audio file, not live VC stream.)"""
    if not context.args:
        await update.effective_message.reply_text(
            "Use: /play <song name>\n"
            "This command sends song audio in chat.\n"
            "Live VC streaming is not enabled in this bot build."
        )
        return
    await song_command(update, context)


async def yt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /yt command - Download from YouTube link"""
    await _register_user(update.effective_user.id)
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await _register_group(update.effective_chat.id)

    if not context.args:
        await update.effective_message.reply_text(
            "Format: /yt <YouTube link>\n\n"
            "Example: /yt https://www.youtube.com/watch?v=...\n"
            "Link se audio nikaal dunga!"
        )
        return

    youtube_link = context.args[0]
    user_id = update.effective_user.id

    if "youtube.com" not in youtube_link and "youtu.be" not in youtube_link:
        await update.effective_message.reply_text(
            "Yeh valid YouTube link nahi lag raha. Ek proper link bhejo."
        )
        return

    search_msg = await update.effective_message.reply_text("Link process kar rahi hoon...")

    user_dir = DOWNLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = await asyncio.to_thread(_download_audio_sync, youtube_link, user_dir)

        if not result:
            await search_msg.edit_text("Ye audio abhi nahi nikal pa raha. Ek aur link try karo.")
            logger.warning(f"Download failed for YouTube link: {youtube_link}")
            await _send_log_to_channel(
                context,
                f"YT_FAIL\nUser: {update.effective_user.id}\nChat: {update.effective_chat.id if update.effective_chat else 'None'}\nLink: {youtube_link}",
            )
            return

        audio_file, metadata = result
        valid, reason = _validate_audio_file(audio_file)
        if not valid:
            file_size = audio_file.stat().st_size if audio_file and audio_file.exists() else 0
            if audio_file and audio_file.exists():
                audio_file.unlink(missing_ok=True)
            if reason == "file_too_large":
                await search_msg.edit_text(
                    f"Video thoda heavy hai ({file_size / 1024 / 1024:.1f}MB). Chhota video try karo."
                )
            else:
                await search_msg.edit_text("Download me issue aaya, ek baar phir try karo.")
            await _send_log_to_channel(
                context,
                f"YT_INVALID_FILE\nUser: {update.effective_user.id}\nChat: {update.effective_chat.id if update.effective_chat else 'None'}\nLink: {youtube_link}\nReason: {reason}",
            )
            return

        await search_msg.edit_text("Song ready. Bhej rahi hoon...")

        try:
            with open(audio_file, "rb") as f:
                await update.effective_message.reply_audio(
                    f,
                    title=metadata.get("title", audio_file.stem[:100]) if metadata else audio_file.stem[:100],
                    performer=metadata.get("performer", "Unknown Artist") if metadata else "Unknown Artist",
                    duration=metadata.get("duration") if metadata else None,
                    reply_to_message_id=update.effective_message.message_id,
                )

            BOT_DB.log_activity(
                "yt_sent",
                user_id=user_id,
                group_id=update.effective_chat.id if update.effective_chat else None,
                metadata={"link": youtube_link, "title": metadata.get("title") if metadata else None},
            )
            await _send_play_log_to_channel(
                context=context,
                update=update,
                searched_text=youtube_link,
                source="youtube_link",
                metadata=metadata,
            )
            logger.info(f"YouTube audio sent to user {user_id}")
        except Exception as e:
            logger.error(f"Failed to send audio file: {e}")
            await search_msg.edit_text("File bhejne me issue aaya, ek baar phir try karo.")
            return

        try:
            await search_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"YouTube download error: {e}")
        await search_msg.edit_text("Thoda issue aaya, ek baar phir try karo.")

    finally:
        _cleanup_downloads(user_dir)

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
    """Handle /hug command - AI generated warm hug messages"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated hug message
    hug_prompt = f"Give a warm, caring virtual hug message to {user_name} in Hinglish (mix of Hindi and English). Make it cute, supportive and comforting. Keep it short (1-2 lines). Use hug emojis 🤗 and heart emojis ❤️."
    ai_hug = get_ai_response(hug_prompt, user_name, hug_prompt)
    
    await update.effective_message.reply_text(ai_hug)


async def tip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tip command - AI generated daily life tips"""
    await _register_user(update.effective_user.id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated tip
    tip_prompt = "Share a practical, useful daily life tip in Hinglish (mix of Hindi and English). Keep it short (2 lines), actionable, and motivational. Add emojis. Start with '💡 Daily Tip:'."
    ai_tip = get_ai_response(tip_prompt, "User", tip_prompt)
    
    await update.effective_message.reply_text(ai_tip, parse_mode=ParseMode.MARKDOWN)


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
    """Handle /joke command - AI generated jokes"""
    await _register_user(update.effective_user.id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated joke
    joke_prompt = "Generate a funny, family-friendly joke in Hinglish (mix of Hindi and English). Keep it short (2-4 lines), witty, and relatable to everyday life. Add emojis. Start with '😄 Joke:'."
    ai_joke = get_ai_response(joke_prompt, "User", joke_prompt)
    
    await update.effective_message.reply_text(ai_joke, parse_mode=ParseMode.MARKDOWN)


async def roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /roast command - AI generated light roasting"""
    await _register_user(update.effective_user.id)
    user_name = update.effective_user.first_name or "Bhai"
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated roast
    roast_prompt = f"Give a funny, light-hearted roast to {user_name} in Hinglish (mix of Hindi and English). Keep it playful, not offensive. Make it witty and funny (1-2 lines). Use laughing emojis 😂🤭."
    ai_roast = get_ai_response(roast_prompt, user_name, roast_prompt)
    
    await update.effective_message.reply_text(ai_roast)


async def truth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /truth command - AI generated truth questions"""
    await _register_user(update.effective_user.id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated truth question
    truth_prompt = "Generate a fun, interesting 'Truth' question for Truth or Dare game in Hinglish (mix of Hindi and English). Keep it short (1-2 lines), appropriate, and interesting. Add emojis. Start with '🎯 Truth Question:'."
    ai_truth = get_ai_response(truth_prompt, "User", truth_prompt)
    
    await update.effective_message.reply_text(ai_truth, parse_mode=ParseMode.MARKDOWN)


async def dare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /dare command - AI generated dare challenges"""
    await _register_user(update.effective_user.id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated dare
    dare_prompt = "Generate a fun, exciting 'Dare' challenge for Truth or Dare game in Hinglish (mix of Hindi and English). Keep it short (1-2 lines), safe, appropriate, and fun. Add emojis. Start with '🎲 Dare:'."
    ai_dare = get_ai_response(dare_prompt, "User", dare_prompt)
    
    await update.effective_message.reply_text(ai_dare, parse_mode=ParseMode.MARKDOWN)


async def fact_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /fact command - AI generated interesting facts"""
    await _register_user(update.effective_user.id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get AI-generated fact
    fact_prompt = "Share an amazing, interesting, or mind-blowing fact in Hinglish (mix of Hindi and English). Keep it short (2-3 lines), fascinating, and educational. Add emojis. Start with '🌟 Interesting Fact:'."
    ai_fact = get_ai_response(fact_prompt, "User", fact_prompt)
    
    await update.effective_message.reply_text(ai_fact, parse_mode=ParseMode.MARKDOWN)


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



async def warn_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Rose-style /warn command. Reply to a user to add warning."""
    await _register_user(update.effective_user.id)

    if not update.message.reply_to_message:
        await update.effective_message.reply_text(
            "Reply karke /warn <reason> use karo."
        )
        return

    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return

    target_user = update.message.reply_to_message.from_user
    group_id = update.effective_chat.id

    try:
        target_member = await context.bot.get_chat_member(group_id, target_user.id)
        if target_member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            await update.effective_message.reply_text("Admin ko warn nahi kar sakte.")
            return
    except Exception:
        pass

    reason = "No reason"
    if context.args:
        reason = " ".join(context.args)[:300]

    warn_count = BOT_DB.add_warn(
        group_id=group_id,
        user_id=target_user.id,
        warned_by=update.effective_user.id,
        reason=reason,
    )

    BOT_DB.log_activity(
        "warn_added",
        user_id=target_user.id,
        group_id=group_id,
        metadata={"count": warn_count, "reason": reason, "by": update.effective_user.id},
    )

    text = (
        f"?? Warning added\n"
        f"User: {target_user.first_name or 'User'} ({target_user.id})\n"
        f"Count: {warn_count}/3\n"
        f"Reason: {reason}"
    )

    if warn_count >= 3:
        try:
            from datetime import datetime, timedelta

            await context.bot.restrict_chat_member(
                chat_id=group_id,
                user_id=target_user.id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=datetime.now() + timedelta(hours=24),
            )
            text += "\nAction: Auto mute 24h (3 warns reached)"
            BOT_DB.log_activity(
                "warn_auto_mute",
                user_id=target_user.id,
                group_id=group_id,
                metadata={"warn_count": warn_count},
            )
        except Exception as e:
            text += f"\nAction failed: {e}"

    await update.effective_message.reply_text(text)
    await _send_log_to_channel(
        context,
        (
            "WARN_ADDED\n"
            f"Chat ID: {group_id}\n"
            f"User ID: {target_user.id}\n"
            f"Count: {warn_count}\n"
            f"Reason: {reason}\n"
            f"By: {update.effective_user.id}"
        ),
    )


async def warnings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show warnings for replied user or top warned users in group."""
    await _register_user(update.effective_user.id)

    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text("Ye command sirf group me kaam karta hai.")
        return

    group_id = update.effective_chat.id

    if update.message.reply_to_message:
        user = update.message.reply_to_message.from_user
        data = BOT_DB.get_warn(group_id, user.id)
        await update.effective_message.reply_text(
            f"Warnings for {user.first_name or 'User'} ({user.id})\n"
            f"Count: {data.get('warn_count', 0)}\n"
            f"Last reason: {data.get('last_reason') or 'None'}"
        )
        return

    if context.args:
        try:
            target_id = int(context.args[0])
        except ValueError:
            await update.effective_message.reply_text("Valid user id do.")
            return
        data = BOT_DB.get_warn(group_id, target_id)
        await update.effective_message.reply_text(
            f"Warnings for {target_id}\n"
            f"Count: {data.get('warn_count', 0)}\n"
            f"Last reason: {data.get('last_reason') or 'None'}"
        )
        return

    top = BOT_DB.get_top_warned(group_id, limit=10)
    if not top:
        await update.effective_message.reply_text("Is group me koi warnings nahi hain.")
        return

    lines = ["Top warned users:"]
    for idx2, item in enumerate(top, 1):
        lines.append(
            f"{idx2}. {item.get('user_id')} -> {item.get('warn_count')} warns | {item.get('last_reason') or 'No reason'}"
        )
    await update.effective_message.reply_text("\n".join(lines))


async def resetwarn_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset warning count for replied user or user_id."""
    await _register_user(update.effective_user.id)

    is_valid, error_msg = await _check_bot_and_user_admin(update, context)
    if not is_valid:
        await update.effective_message.reply_text(error_msg)
        return

    group_id = update.effective_chat.id
    target_id: Optional[int] = None
    target_name = "User"

    if update.message.reply_to_message:
        target = update.message.reply_to_message.from_user
        target_id = target.id
        target_name = target.first_name or "User"
    elif context.args:
        try:
            target_id = int(context.args[0])
            target_name = str(target_id)
        except ValueError:
            await update.effective_message.reply_text("Valid user id do.")
            return
    else:
        await update.effective_message.reply_text(
            "Reply user pe /resetwarn use karo ya /resetwarn <user_id>."
        )
        return

    BOT_DB.reset_warn(group_id, target_id)
    BOT_DB.log_activity(
        "warn_reset",
        user_id=target_id,
        group_id=group_id,
        metadata={"by": update.effective_user.id},
    )

    await update.effective_message.reply_text(f"Warnings reset for {target_name}.")
    await _send_log_to_channel(
        context,
        (
            "WARN_RESET\n"
            f"Chat ID: {group_id}\n"
            f"User ID: {target_id}\n"
            f"By: {update.effective_user.id}"
        ),
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
        "????? *Admin Commands* - Baby\n\n"
        "?? /settings\n"
        "? Group settings customize karo\n\n"
        "??? /del\n"
        "? Reply karke message delete karo\n\n"
        "?? /ban\n"
        "? Reply karke user ban karo\n\n"
        "?? /unban <user_id>\n"
        "? Reply ya ID se unban karo\n\n"
        "?? /warn <reason>\n"
        "? Reply karke warning do\n\n"
        "?? /warnings [reply/user_id]\n"
        "? Warning count dekho\n\n"
        "?? /resetwarn [reply/user_id]\n"
        "? Warnings reset karo\n\n"
        "?? /mute <time>\n"
        "? Reply karke mute karo (10m, 1h, 1d)\n\n"
        "?? /unmute\n"
        "? Reply karke mute hatao\n\n"
        "?? /promote\n"
        "? Reply karke admin banao\n\n"
        "?? /demote\n"
        "? Reply karke admin hatao\n\n"
        "?? /pin\n"
        "? Reply karke message pin karo\n\n"
        "?? /unpin\n"
        "? Reply karke unpin karo\n\n"
        "Note: Bot ko admin banana zaruri hai."
    )
    await update.effective_message.reply_text(
        admin_help_text,
        parse_mode=ParseMode.MARKDOWN
    )
    
    logger.info(f"Admin help shown to {update.effective_user.first_name}")


# ========================= ANALYTICS & STATS COMMANDS ========================= #

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command - Show user and group analytics (admin only)"""
    user_id = update.effective_user.id
    
    # Check if user is admin
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Oops! Sirf admin (bot ka owner) is command use kar sakte hain. 😅"
        )
        return
    
    # Calculate statistics
    total_users = len(USERS_DATABASE)
    active_users = len(REGISTERED_USERS - OPTED_OUT_USERS)
    opted_out = len(OPTED_OUT_USERS & REGISTERED_USERS)
    total_groups = len(GROUPS_DATABASE)
    
    # Find most active users
    users_by_activity = sorted(
        USERS_DATABASE.items(),
        key=lambda x: x[1].get('last_seen', 0),
        reverse=True
    )
    top_active_users = users_by_activity[:5]
    
    # Find newest users
    users_by_join = sorted(
        USERS_DATABASE.items(),
        key=lambda x: x[1].get('join_date', 0),
        reverse=True
    )
    newest_users = users_by_join[:5]
    
    # Group stats
    groups_by_activity = sorted(
        GROUPS_DATABASE.items(),
        key=lambda x: x[1].get('last_active', 0),
        reverse=True
    )
    most_active_groups = groups_by_activity[:5]
    
    # Calculate dates
    current_time = time.time()
    
    # Build stats message
    stats_text = (
        "📊 **BOT ANALYTICS** 📊\n"
        "=" * 40 + "\n\n"
        
        "👥 **USER STATISTICS**\n"
        f"Total Registered: {total_users}\n"
        f"Active (Receiving Broadcasts): {active_users}\n"
        f"Opted Out (/stop): {opted_out}\n"
        f"Blocked/Deactivated: {total_users - active_users - opted_out}\n\n"
        
        "👥 **MOST ACTIVE USERS** (Last Seen)\n"
    )
    
    for idx, (uid, user_info) in enumerate(top_active_users, 1):
        name = user_info.get('first_name', 'Unknown')
        last_seen = user_info.get('last_seen', 0)
        hours_ago = int((current_time - last_seen) / 3600)
        
        if hours_ago < 1:
            time_str = "few mins ago"
        elif hours_ago < 24:
            time_str = f"{hours_ago}h ago"
        else:
            days_ago = hours_ago // 24
            time_str = f"{days_ago}d ago"
        
        stats_text += f"{idx}. {name} - {time_str}\n"
    
    # Newest users
    stats_text += "\n👤 **NEWEST USERS** (Joined)\n"
    for idx, (uid, user_info) in enumerate(newest_users, 1):
        name = user_info.get('first_name', 'Unknown')
        join_date = user_info.get('join_date', 0)
        hours_ago = int((current_time - join_date) / 3600)
        
        if hours_ago < 1:
            time_str = "just now"
        elif hours_ago < 24:
            time_str = f"{hours_ago}h ago"
        else:
            days_ago = hours_ago // 24
            time_str = f"{days_ago}d ago"
        
        stats_text += f"{idx}. {name} - {time_str}\n"
    
    # Group stats
    stats_text += (
        f"\n👥 **GROUP STATISTICS**\n"
        f"Total Groups: {total_groups}\n\n"
        f"🏆 **MOST ACTIVE GROUPS** (Last Active)\n"
    )
    
    for idx, (gid, group_info) in enumerate(most_active_groups, 1):
        title = group_info.get('title', 'Unknown')
        members = len(group_info.get('members', {}))
        last_active = group_info.get('last_active', 0)
        hours_ago = int((current_time - last_active) / 3600)
        
        if hours_ago < 1:
            time_str = "few mins ago"
        elif hours_ago < 24:
            time_str = f"{hours_ago}h ago"
        else:
            days_ago = hours_ago // 24
            time_str = f"{days_ago}d ago"
        
        stats_text += f"{idx}. {title} ({members} members) - {time_str}\n"
    
    stats_text += "\n" + "=" * 40 + "\n"
    
    await update.effective_message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
    logger.info(f"Stats shown to admin {update.effective_user.first_name}")


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /users command - Show detailed user list (admin only)"""
    user_id = update.effective_user.id

    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "Only owner can view user list."
        )
        return

    all_users = BOT_DB.get_all_users()
    total_users = len(all_users)

    if total_users == 0:
        await update.effective_message.reply_text("No users found yet.")
        return

    current_time = time.time()
    users_text = f"ALL USERS ({total_users} Total)\n" + "=" * 50 + "\n\n"

    for idx, user_info in enumerate(all_users, 1):
        uid = int(user_info.get("user_id", 0))
        name = user_info.get("first_name", "Unknown")
        username = user_info.get("username")
        join_date = float(user_info.get("join_date", 0) or 0)
        last_seen = float(user_info.get("last_seen", 0) or 0)

        days_since_join = int((current_time - join_date) / 86400) if join_date else -1
        hours_since_seen = int((current_time - last_seen) / 3600) if last_seen else -1

        if hours_since_seen < 0:
            last_seen_str = "unknown"
        elif hours_since_seen < 1:
            last_seen_str = "just now"
        elif hours_since_seen < 24:
            last_seen_str = f"{hours_since_seen}h ago"
        else:
            last_seen_str = f"{hours_since_seen // 24}d ago"

        joined_str = f"{days_since_join}d ago" if days_since_join >= 0 else "unknown"
        opted_out = "(Opted-out)" if uid in OPTED_OUT_USERS else ""
        username_text = f"@{username}" if username and username != "None" else "None"

        users_text += (
            f"{idx}. {name} {opted_out}\n"
            f"   ID: {uid}\n"
            f"   Username: {username_text}\n"
            f"   Joined: {joined_str}\n"
            f"   Last Seen: {last_seen_str}\n\n"
        )

        if idx % 20 == 0:
            await update.effective_message.reply_text(users_text)
            users_text = ""

    if users_text:
        await update.effective_message.reply_text(users_text)

    logger.info(f"User list shown to admin {update.effective_user.first_name}")

async def groups_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /groups command - Show detailed group list (admin only)"""
    user_id = update.effective_user.id
    
    # Check if user is admin
    if user_id != ADMIN_ID:
        await update.effective_message.reply_text(
            "🔐 Oops! Sirf admin hi group list dekh sakte hain. 😅"
        )
        return
    
    total_groups = len(GROUPS_DATABASE)
    
    if total_groups == 0:
        await update.effective_message.reply_text(
            "❌ Koi bhi group nahi hai abhi! 😅"
        )
        return
    
    # Sort groups by last active (most active first)
    groups_by_activity = sorted(
        GROUPS_DATABASE.items(),
        key=lambda x: x[1].get('last_active', 0),
        reverse=True
    )
    
    current_time = time.time()
    groups_text = f"👥 **ALL GROUPS** ({total_groups} Total)\n" + "=" * 50 + "\n\n"
    
    total_members = 0
    
    for idx, (gid, group_info) in enumerate(groups_by_activity, 1):
        title = group_info.get('title', 'Unknown')
        group_type = group_info.get('type', 'group')
        members = len(group_info.get('members', {}))
        added_date = group_info.get('added_date', 0)
        last_active = group_info.get('last_active', 0)
        
        total_members += members
        
        # Format dates
        days_since_added = int((current_time - added_date) / 86400)
        hours_since_active = int((current_time - last_active) / 3600)
        
        if hours_since_active < 1:
            active_str = "active now"
        elif hours_since_active < 24:
            active_str = f"{hours_since_active}h ago"
        else:
            days_since_active = hours_since_active // 24
            active_str = f"{days_since_active}d ago"
        
        groups_text += (
            f"{idx}. **{title}**\n"
            f"   ID: {gid}\n"
            f"   Type: {group_type}\n"
            f"   Members: {members}\n"
            f"   Added: {days_since_added}d ago\n"
            f"   Last Active: {active_str}\n\n"
        )
        
        # Split into messages every 15 groups
        if idx % 15 == 0:
            await update.effective_message.reply_text(groups_text, parse_mode=ParseMode.MARKDOWN)
            groups_text = ""
    
    if groups_text:
        await update.effective_message.reply_text(groups_text, parse_mode=ParseMode.MARKDOWN)
    
    # Summary
    summary = (
        f"\n" + "=" * 50 + "\n"
        f"📊 **SUMMARY**\n"
        f"Total Groups: {total_groups}\n"
        f"Total Members (across groups): {total_members}\n"
        f"Avg Members per Group: {total_members // max(1, total_groups)}"
    )
    await update.effective_message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
    
    logger.info(f"Group list shown to admin {update.effective_user.first_name}")


# ========================= OWNER DASHBOARD & MUSIC BROADCAST ========================= #

def _format_age(ts: Optional[float]) -> str:
    if not ts:
        return "unknown"
    diff = int(time.time() - ts)
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}m ago"
    if diff < 86400:
        return f"{diff // 3600}h ago"
    return f"{diff // 86400}d ago"


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner dashboard: sqlite overview + recent activity."""
    if update.effective_user.id != ADMIN_ID:
        await update.effective_message.reply_text("Sirf owner dashboard dekh sakta hai.")
        return

    overview = BOT_DB.get_overview()
    recent = BOT_DB.get_recent_activities(8)
    lines = [
        "Live Dashboard",
        "",
        f"Users: {overview['users']}",
        f"Groups: {overview['groups']}",
        f"Active Users (24h): {overview['active_users_24h']}",
        f"Active Groups (24h): {overview['active_groups_24h']}",
        f"Events (24h): {overview['activities_24h']}",
        "",
        "Recent Activity:",
    ]

    if not recent:
        lines.append("No activity yet.")
    else:
        for item in recent:
            lines.append(
                f"- {item.get('action')} | user={item.get('user_id')} | group={item.get('group_id')} | {_format_age(item.get('ts'))}"
            )

    await update.effective_message.reply_text("\n".join(lines))


async def broadcastsong_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner command: download one song and broadcast to all users + groups."""
    if update.effective_user.id != ADMIN_ID:
        await update.effective_message.reply_text("Sirf owner ye command use kar sakta hai.")
        return

    if not context.args:
        await update.effective_message.reply_text(
            "Usage: /broadcastsong <song name>\nExample: /broadcastsong Tum Hi Ho"
        )
        return

    query = " ".join(context.args).strip()
    status = await update.effective_message.reply_text(f"Song fetch kar raha hoon: {query}")

    temp_dir = DOWNLOAD_DIR / "broadcast"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_file: Optional[Path] = None
    metadata: Dict[str, Any] = {}

    try:
        result = await asyncio.to_thread(_download_audio_sync, query, temp_dir)
        if not result:
            await status.edit_text("Song fetch nahi hua. Dusra query try karo.")
            BOT_DB.log_activity("broadcast_song_failed", user_id=update.effective_user.id, metadata={"query": query})
            return

        audio_file, metadata = result
        valid, reason = _validate_audio_file(audio_file)
        if not valid:
            if audio_file and audio_file.exists():
                audio_file.unlink(missing_ok=True)
            await status.edit_text(f"Song invalid file ({reason}). Dusra query try karo.")
            BOT_DB.log_activity("broadcast_song_failed", user_id=update.effective_user.id, metadata={"query": query, "reason": reason})
            return

        users = list(REGISTERED_USERS - OPTED_OUT_USERS)
        groups = list(GROUPS_DATABASE.keys())
        seed_message = None
        file_id: Optional[str] = None

        with open(audio_file, "rb") as f:
            seed_message = await context.bot.send_audio(
                chat_id=update.effective_chat.id,
                audio=f,
                title=metadata.get("title", audio_file.stem[:100]),
                performer=metadata.get("performer", "Unknown Artist"),
                duration=metadata.get("duration"),
                caption="Broadcast seed message",
            )
        if seed_message and seed_message.audio:
            file_id = seed_message.audio.file_id

        sent_users = 0
        sent_groups = 0
        failed_users = 0
        failed_groups = 0

        for uid in users:
            try:
                if file_id:
                    await context.bot.send_audio(
                        chat_id=uid,
                        audio=file_id,
                        title=metadata.get("title", audio_file.stem[:100]),
                        performer=metadata.get("performer", "Unknown Artist"),
                        duration=metadata.get("duration"),
                    )
                else:
                    with open(audio_file, "rb") as f:
                        await context.bot.send_audio(
                            chat_id=uid,
                            audio=f,
                            title=metadata.get("title", audio_file.stem[:100]),
                            performer=metadata.get("performer", "Unknown Artist"),
                            duration=metadata.get("duration"),
                        )
                sent_users += 1
            except Exception as e:
                failed_users += 1
                err = str(e).lower()
                if "blocked" in err or "deactivated" in err or "chat not found" in err:
                    _remove_user_everywhere(uid)

        for gid in groups:
            try:
                if file_id:
                    await context.bot.send_audio(
                        chat_id=gid,
                        audio=file_id,
                        title=metadata.get("title", audio_file.stem[:100]),
                        performer=metadata.get("performer", "Unknown Artist"),
                        duration=metadata.get("duration"),
                        caption="Owner music broadcast",
                    )
                else:
                    with open(audio_file, "rb") as f:
                        await context.bot.send_audio(
                            chat_id=gid,
                            audio=f,
                            title=metadata.get("title", audio_file.stem[:100]),
                            performer=metadata.get("performer", "Unknown Artist"),
                            duration=metadata.get("duration"),
                            caption="Owner music broadcast",
                        )
                sent_groups += 1
            except Exception as e:
                failed_groups += 1
                err = str(e).lower()
                if "kicked" in err or "not a member" in err or "chat not found" in err:
                    _remove_group_everywhere(gid)

        BOT_DB.log_activity(
            "broadcast_song_sent",
            user_id=update.effective_user.id,
            metadata={
                "query": query,
                "title": metadata.get("title"),
                "users": sent_users,
                "groups": sent_groups,
            },
        )
        await status.edit_text(
            f"Broadcast song complete.\nUsers: {sent_users} sent, {failed_users} failed\nGroups: {sent_groups} sent, {failed_groups} failed"
        )

    finally:
        if audio_file and audio_file.exists():
            audio_file.unlink(missing_ok=True)
        _cleanup_downloads(temp_dir)


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current chat and sender IDs for quick setup/debug."""
    await _register_user_from_update(update)

    chat = update.effective_chat
    user = update.effective_user

    if not chat:
        await update.effective_message.reply_text("Chat info available nahi hai.")
        return

    lines = [
        "Chat Info",
        f"Chat ID: {chat.id}",
        f"Chat Type: {chat.type}",
        f"Chat Title: {chat.title or 'N/A'}",
        f"Chat Username: @{chat.username}" if chat.username else "Chat Username: None",
    ]

    if user:
        lines.append(f"User ID: {user.id}")
        lines.append(f"User Username: @{user.username}" if user.username else "User Username: None")

    await update.effective_message.reply_text("\n".join(lines))



async def channelstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send past/present usage summary to the log channel and caller."""
    if update.effective_user.id != ADMIN_ID:
        await update.effective_message.reply_text("Only owner can use this command.")
        return

    report = _build_channel_stats_report()
    await _send_log_to_channel(context, report)
    await update.effective_message.reply_text(report)


async def vplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Play music in Telegram voice chat using assistant + PyTgCalls."""
    await _register_user_from_update(update)

    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text("/vplay works in groups only.")
        return

    if not context.args:
        await update.effective_message.reply_text("Usage: /vplay <song name or url>")
        return

    query = " ".join(context.args).strip()
    status_msg = await update.effective_message.reply_text("Resolving track for voice chat...")

    try:
        vc = await _get_vc_manager()
        requested_by = update.effective_user.first_name or "User"
        mode, track = await vc.enqueue_or_play(update.effective_chat.id, query, requested_by)

        if mode == "playing":
            await status_msg.edit_text(
                f"VC Now Playing\nTitle: {track.title}\nRequested by: {requested_by}\nSource: {track.webpage_url}"
            )
        else:
            queue_len = len(vc.get_queue(update.effective_chat.id))
            await status_msg.edit_text(
                f"Added to VC queue\nTitle: {track.title}\nPosition: {queue_len}"
            )

        await _send_log_to_channel(
            context,
            (
                "VC_PLAY\n"
                f"Chat ID: {update.effective_chat.id}\n"
                f"Query: {query}\n"
                f"Mode: {mode}\n"
                f"By: {update.effective_user.id}"
            ),
        )
    except Exception as e:
        await status_msg.edit_text(f"VC play failed: {e}")


async def vstop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop voice chat playback and clear queue."""
    await _register_user_from_update(update)

    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text("/vstop works in groups only.")
        return

    try:
        vc = await _get_vc_manager()
        await vc.stop_chat(update.effective_chat.id)
        await update.effective_message.reply_text("Voice chat playback stopped and queue cleared.")
    except Exception as e:
        await update.effective_message.reply_text(f"VC stop failed: {e}")


async def vskip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Skip current VC track and play next from queue."""
    await _register_user_from_update(update)

    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text("/vskip works in groups only.")
        return

    try:
        vc = await _get_vc_manager()
        next_track = await vc.skip(update.effective_chat.id)
        if not next_track:
            await update.effective_message.reply_text("Queue empty. Stopped current playback.")
            return
        await update.effective_message.reply_text(
            f"Skipped.\nNow Playing: {next_track.title}\nRequested by: {next_track.requested_by}"
        )
    except Exception as e:
        await update.effective_message.reply_text(f"VC skip failed: {e}")


async def vqueue_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show voice chat queue."""
    await _register_user_from_update(update)

    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text("/vqueue works in groups only.")
        return

    try:
        vc = await _get_vc_manager()
        now_track = vc.get_now_playing(update.effective_chat.id)
        queue = vc.get_queue(update.effective_chat.id)

        lines = ["VC Queue"]
        if now_track:
            lines.append(f"Now: {now_track.title} (by {now_track.requested_by})")
        else:
            lines.append("Now: Nothing")

        if not queue:
            lines.append("Queue: Empty")
        else:
            lines.append("Queue:")
            for i, item in enumerate(queue[:10], 1):
                lines.append(f"{i}. {item.title} (by {item.requested_by})")

        await update.effective_message.reply_text("\n".join(lines))
    except Exception as e:
        await update.effective_message.reply_text(f"VC queue failed: {e}")

# ========================= GROUP SETTINGS COMMANDS ========================= #

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command - Show group settings menu (admin only)"""
    await _register_user(update.effective_user.id)
    
    # Must be in a group
    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        await update.effective_message.reply_text(
            "❌ Ye command sirf groups mein kaam karta hai! 😊"
        )
        return
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Check if user is admin
    try:
        user_member = await context.bot.get_chat_member(chat_id, user_id)
        
        if user_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]:
            await update.effective_message.reply_text(
                "❌ Sirf admins hi group settings change kar sakte hain! 😊"
            )
            return
            
    except Exception as e:
        logger.error(f"Settings command error for user {user_id} in chat {chat_id}: {type(e).__name__}: {e}", exc_info=True)
        # If permission check fails, show settings anyway if user seems legit
        # Telegram will handle permission issues at callback level
        logger.warning(f"Proceeding with settings despite permission check failure for user {user_id}")
    
    group_id = update.effective_chat.id
    
    # Initialize settings if not exists
    if group_id not in GROUP_SETTINGS:
        GROUP_SETTINGS[group_id] = DEFAULT_GROUP_SETTINGS.copy()
        _save_group_settings(GROUP_SETTINGS)
    
    # Create category-based settings menu (like Rose Bot)
    keyboard = [
        [InlineKeyboardButton("🗑️ Message Management", callback_data=f"setting_cat_messages_{group_id}"),
         InlineKeyboardButton("🛡️ Security", callback_data=f"setting_cat_security_{group_id}")],
        [InlineKeyboardButton("📋 Content Control", callback_data=f"setting_cat_content_{group_id}"),
         InlineKeyboardButton("👋 Notifications", callback_data=f"setting_cat_notify_{group_id}")],
        [InlineKeyboardButton("📊 View All", callback_data=f"setting_view_{group_id}"),
         InlineKeyboardButton("❌ Close", callback_data="setting_close")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.effective_message.reply_text(
        "⚙️ *Group Settings - Baby Bot* ❤️\n\n"
        "Apne group ke settings customize karo! 🎨\n\n"
        "*Categories:*\n"
        "🗑️ *Message Management* - Auto-delete messages\n"
        "🛡️ *Security* - Spam & anti-flood protection\n"
        "📋 *Content Control* - Stickers, GIFs, links, forwards\n"
        "👋 *Notifications* - Welcome messages\n\n"
        "Kisi bhi category pe click karo! 👇",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )


# ========================= CALLBACK HANDLERS ========================= #

async def handle_setting_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group settings callbacks with full functionality"""
    query = update.callback_query
    data = query.data
    
    if data == "setting_close":
        await query.message.delete()
        return
    
    # Extract parts from callback_data
    parts = data.split("_")
    if len(parts) < 3:
        await query.answer("Invalid callback!", show_alert=True)
        return
    
    action = parts[1]
    try:
        group_id = int(parts[2]) if parts[2].lstrip('-').isdigit() else 0
    except (ValueError, IndexError):
        await query.answer("Invalid group ID!", show_alert=True)
        return
    
    if not group_id:
        await query.answer("Invalid group!", show_alert=True)
        return
    
    # Initialize settings if not exists
    if group_id not in GROUP_SETTINGS:
        GROUP_SETTINGS[group_id] = DEFAULT_GROUP_SETTINGS.copy()
        _save_group_settings(GROUP_SETTINGS)
    
    settings = GROUP_SETTINGS[group_id]
    updated_defaults = False
    for key, value in DEFAULT_GROUP_SETTINGS.items():
        if key not in settings:
            settings[key] = value
            updated_defaults = True
    if updated_defaults:
        _save_group_settings(GROUP_SETTINGS)
    
    # ============ VIEW ALL SETTINGS ============
    if action == "view":
        settings_text = (
            "📋 *Current Group Settings*\n\n"
            "*🗑️ Message Management:*\n"
            f"├─ Auto Delete: {'✅ ON' if settings['auto_delete_enabled'] else '❌ OFF'} ({settings['auto_delete_count']} msgs)\n"
            f"├─ Max Length: {settings['max_message_length']} chars\n\n"
            "*🛡️ Security:*\n"
            f"├─ Spam Protection: {'✅ ON' if settings['spam_protection'] else '❌ OFF'} ({settings['spam_threshold']} msgs)\n"
            f"├─ Delete Admin Spam: {'✅ YES' if settings['delete_admin_spam'] else '❌ NO'}\n"
            f"├─ Anti-Flood: {'✅ ON' if settings['antiflood_enabled'] else '❌ OFF'}\n\n"
            "*📋 Content Control:*\n"
            f"├─ Stickers: {'✅ Allowed' if settings['allow_stickers'] else '❌ Not Allowed'}\n"
            f"├─ GIFs: {'✅ Allowed' if settings['allow_gifs'] else '❌ Not Allowed'}\n"
            f"├─ Links: {'✅ Allowed' if settings['allow_links'] else '❌ Not Allowed'}\n"
            f"├─ Forwards: {'✅ Allowed' if settings['allow_forwards'] else '❌ Not Allowed'}\n"
            f"├─ Bot Links: {'🧹 Auto Delete' if settings['remove_bot_links'] else '✅ Allowed'}\n\n"
            "*👋 Notifications:*\n"
            f"└─ Welcome: {'✅ ON' if settings['welcome_message'] else '❌ OFF'}"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        await query.answer()
        return
    
    # ============ MAIN MENU ============
    if action == "menu":
        keyboard = [
            [InlineKeyboardButton("🗑️ Message Management", callback_data=f"setting_cat_messages_{group_id}"),
             InlineKeyboardButton("🛡️ Security", callback_data=f"setting_cat_security_{group_id}")],
            [InlineKeyboardButton("📋 Content Control", callback_data=f"setting_cat_content_{group_id}"),
             InlineKeyboardButton("👋 Notifications", callback_data=f"setting_cat_notify_{group_id}")],
            [InlineKeyboardButton("📊 View All", callback_data=f"setting_view_{group_id}"),
             InlineKeyboardButton("❌ Close", callback_data="setting_close")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "⚙️ *Group Settings - Baby Bot* ❤️\n\n"
            "Apne group ke settings customize karo! 🎨\n\n"
            "*Categories:*\n"
            "🗑️ *Message Management* - Auto-delete messages\n"
            "🛡️ *Security* - Spam & anti-flood protection\n"
            "📋 *Content Control* - Stickers, GIFs, links, forwards\n"
            "👋 *Notifications* - Welcome messages\n\n"
            "Kisi bhi category pe click karo! 👇",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    # ============ CATEGORY: MESSAGE MANAGEMENT ============
    if action == "cat" and len(parts) >= 4 and parts[3] == "messages":
        keyboard = [
            [InlineKeyboardButton(f"🗑️ Auto Delete: {'✅' if settings['auto_delete_enabled'] else '❌'}", 
                                callback_data=f"setting_autodel_{group_id}")],
            [InlineKeyboardButton(f"✏️ Message Count: {settings['auto_delete_count']}", 
                                callback_data=f"setting_editautocount_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🗑️ *Message Management*\n\n"
            f"Auto Delete: {'✅ ENABLED' if settings['auto_delete_enabled'] else '❌ DISABLED'}\n"
            f"Delete after {settings['auto_delete_count']} messages\n\n"
            "Click buttons to customize:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    # ============ CATEGORY: SECURITY ============
    if action == "cat" and len(parts) >= 4 and parts[3] == "security":
        keyboard = [
            [InlineKeyboardButton(f"🛡️ Spam: {'✅' if settings['spam_protection'] else '❌'}", 
                                callback_data=f"setting_spam_{group_id}"),
             InlineKeyboardButton(f"🚫 Flood: {'✅' if settings['antiflood_enabled'] else '❌'}", 
                                callback_data=f"setting_antiflood_{group_id}")],
            [InlineKeyboardButton(f"✏️ Threshold: {settings['spam_threshold']}", 
                                callback_data=f"setting_editspamcount_{group_id}")],
            [InlineKeyboardButton(f"Admin Spam: {'✅' if settings['delete_admin_spam'] else '❌'}", 
                                callback_data=f"setting_adminspam_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🛡️ *Security Settings*\n\n"
            f"Spam Protection: {'✅ ON' if settings['spam_protection'] else '❌ OFF'}\n"
            f"Anti-Flood: {'✅ ON' if settings['antiflood_enabled'] else '❌ OFF'}\n"
            f"Threshold: {settings['spam_threshold']} msgs/10s\n"
            f"Delete Admin Spam: {'✅ YES' if settings['delete_admin_spam'] else '❌ NO'}\n\n"
            "Click to toggle:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    # ============ CATEGORY: CONTENT CONTROL ============
    if action == "cat" and len(parts) >= 4 and parts[3] == "content":
        keyboard = [
            [InlineKeyboardButton(f"🎭 Stickers: {'✅' if settings['allow_stickers'] else '❌'}", 
                                callback_data=f"setting_stickers_{group_id}"),
             InlineKeyboardButton(f"🎬 GIFs: {'✅' if settings['allow_gifs'] else '❌'}", 
                                callback_data=f"setting_gifs_{group_id}")],
            [InlineKeyboardButton(f"🔗 Links: {'✅' if settings['allow_links'] else '❌'}", 
                                callback_data=f"setting_links_{group_id}"),
             InlineKeyboardButton(f"↪️ Forwards: {'✅' if settings['allow_forwards'] else '❌'}", 
                                callback_data=f"setting_forwards_{group_id}")],
            [InlineKeyboardButton(f"🤖 Bot Links: {'🧹' if settings['remove_bot_links'] else '✅'}",
                                callback_data=f"setting_botlinks_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "📋 *Content Control Settings*\n\n"
            f"Stickers: {'✅ Allowed' if settings['allow_stickers'] else '❌ Not Allowed'}\n"
            f"GIFs: {'✅ Allowed' if settings['allow_gifs'] else '❌ Not Allowed'}\n"
            f"Links: {'✅ Allowed' if settings['allow_links'] else '❌ Not Allowed'}\n"
            f"Forwards: {'✅ Allowed' if settings['allow_forwards'] else '❌ Not Allowed'}\n"
            f"Bot Links: {'🧹 Auto Delete' if settings['remove_bot_links'] else '✅ Allowed'}\n\n"
            "Click to toggle:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    # ============ CATEGORY: NOTIFICATIONS ============
    if action == "cat" and len(parts) >= 4 and parts[3] == "notify":
        keyboard = [
            [InlineKeyboardButton(f"👋 Welcome: {'✅' if settings['welcome_message'] else '❌'}", 
                                callback_data=f"setting_welcome_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "👋 *Notification Settings*\n\n"
            f"Welcome Message: {'✅ ON' if settings['welcome_message'] else '❌ OFF'}\n\n"
            "Click to toggle:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    # ============ TOGGLE SETTINGS ============
    if action == "autodel":
        settings['auto_delete_enabled'] = not settings['auto_delete_enabled']
        update_group_setting(group_id, 'auto_delete_enabled', settings['auto_delete_enabled'])
        await query.answer(f"✅ Auto-delete {'enabled' if settings['auto_delete_enabled'] else 'disabled'}!")
        
        # Refresh category view
        keyboard = [
            [InlineKeyboardButton(f"🗑️ Auto Delete: {'✅' if settings['auto_delete_enabled'] else '❌'}", 
                                callback_data=f"setting_autodel_{group_id}")],
            [InlineKeyboardButton(f"✏️ Message Count: {settings['auto_delete_count']}", 
                                callback_data=f"setting_editautocount_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    if action == "spam":
        settings['spam_protection'] = not settings['spam_protection']
        update_group_setting(group_id, 'spam_protection', settings['spam_protection'])
        await query.answer(f"✅ Spam protection {'enabled' if settings['spam_protection'] else 'disabled'}!")
        
        keyboard = [
            [InlineKeyboardButton(f"🛡️ Spam: {'✅' if settings['spam_protection'] else '❌'}", 
                                callback_data=f"setting_spam_{group_id}"),
             InlineKeyboardButton(f"🚫 Flood: {'✅' if settings['antiflood_enabled'] else '❌'}", 
                                callback_data=f"setting_antiflood_{group_id}")],
            [InlineKeyboardButton(f"✏️ Threshold: {settings['spam_threshold']}", 
                                callback_data=f"setting_editspamcount_{group_id}")],
            [InlineKeyboardButton(f"Admin Spam: {'✅' if settings['delete_admin_spam'] else '❌'}", 
                                callback_data=f"setting_adminspam_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    if action == "antiflood":
        settings['antiflood_enabled'] = not settings['antiflood_enabled']
        update_group_setting(group_id, 'antiflood_enabled', settings['antiflood_enabled'])
        await query.answer(f"✅ Anti-flood {'enabled' if settings['antiflood_enabled'] else 'disabled'}!")
        
        keyboard = [
            [InlineKeyboardButton(f"🛡️ Spam: {'✅' if settings['spam_protection'] else '❌'}", 
                                callback_data=f"setting_spam_{group_id}"),
             InlineKeyboardButton(f"🚫 Flood: {'✅' if settings['antiflood_enabled'] else '❌'}", 
                                callback_data=f"setting_antiflood_{group_id}")],
            [InlineKeyboardButton(f"✏️ Threshold: {settings['spam_threshold']}", 
                                callback_data=f"setting_editspamcount_{group_id}")],
            [InlineKeyboardButton(f"Admin Spam: {'✅' if settings['delete_admin_spam'] else '❌'}", 
                                callback_data=f"setting_adminspam_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    if action == "adminspam":
        settings['delete_admin_spam'] = not settings['delete_admin_spam']
        update_group_setting(group_id, 'delete_admin_spam', settings['delete_admin_spam'])
        await query.answer(f"✅ Admin spam deletion {'enabled' if settings['delete_admin_spam'] else 'disabled'}!")
        
        keyboard = [
            [InlineKeyboardButton(f"🛡️ Spam: {'✅' if settings['spam_protection'] else '❌'}", 
                                callback_data=f"setting_spam_{group_id}"),
             InlineKeyboardButton(f"🚫 Flood: {'✅' if settings['antiflood_enabled'] else '❌'}", 
                                callback_data=f"setting_antiflood_{group_id}")],
            [InlineKeyboardButton(f"✏️ Threshold: {settings['spam_threshold']}", 
                                callback_data=f"setting_editspamcount_{group_id}")],
            [InlineKeyboardButton(f"Admin Spam: {'✅' if settings['delete_admin_spam'] else '❌'}", 
                                callback_data=f"setting_adminspam_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    # ============ CONTENT CONTROL TOGGLES ============
    if action in ["stickers", "gifs", "links", "forwards", "welcome", "botlinks"]:
        setting_map = {
            "stickers": "allow_stickers",
            "gifs": "allow_gifs",
            "links": "allow_links",
            "forwards": "allow_forwards",
            "welcome": "welcome_message",
            "botlinks": "remove_bot_links",
        }
        
        name_map = {
            "stickers": "Stickers",
            "gifs": "GIFs",
            "links": "Links",
            "forwards": "Forwards",
            "welcome": "Welcome Message",
            "botlinks": "Bot Link Auto-Delete",
        }
        
        setting_key = setting_map[action]
        settings[setting_key] = not settings[setting_key]
        update_group_setting(group_id, setting_key, settings[setting_key])
        
        status = "enabled" if settings[setting_key] else "disabled"
        await query.answer(f"✅ {name_map[action]} {status}!")
        
        # Determine which category to refresh
        if action in ["stickers", "gifs", "links", "forwards", "botlinks"]:
            keyboard = [
                [InlineKeyboardButton(f"🎭 Stickers: {'✅' if settings['allow_stickers'] else '❌'}", 
                                    callback_data=f"setting_stickers_{group_id}"),
                 InlineKeyboardButton(f"🎬 GIFs: {'✅' if settings['allow_gifs'] else '❌'}", 
                                    callback_data=f"setting_gifs_{group_id}")],
                [InlineKeyboardButton(f"🔗 Links: {'✅' if settings['allow_links'] else '❌'}", 
                                    callback_data=f"setting_links_{group_id}"),
                 InlineKeyboardButton(f"↪️ Forwards: {'✅' if settings['allow_forwards'] else '❌'}", 
                                    callback_data=f"setting_forwards_{group_id}")],
                [InlineKeyboardButton(f"🤖 Bot Links: {'🧹' if settings['remove_bot_links'] else '✅'}",
                                    callback_data=f"setting_botlinks_{group_id}")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
            ]
        else:
            keyboard = [
                [InlineKeyboardButton(f"👋 Welcome: {'✅' if settings['welcome_message'] else '❌'}", 
                                    callback_data=f"setting_welcome_{group_id}")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
            ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    # ============ EDIT NUMERIC VALUES ============
    if action == "editautocount":
        keyboard = [
            [InlineKeyboardButton("50", callback_data=f"setting_setautocount_{group_id}_50"),
             InlineKeyboardButton("100", callback_data=f"setting_setautocount_{group_id}_100")],
            [InlineKeyboardButton("200", callback_data=f"setting_setautocount_{group_id}_200"),
             InlineKeyboardButton("500", callback_data=f"setting_setautocount_{group_id}_500")],
            [InlineKeyboardButton("1000", callback_data=f"setting_setautocount_{group_id}_1000"),
             InlineKeyboardButton("2000", callback_data=f"setting_setautocount_{group_id}_2000")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"setting_cat_messages_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"✏️ *Select Message Count*\n\n"
            f"Current: {settings['auto_delete_count']} messages\n\n"
            f"Messages will be deleted after this count:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    if action == "setautocount" and len(parts) >= 4:
        new_count = int(parts[3])
        settings['auto_delete_count'] = new_count
        update_group_setting(group_id, 'auto_delete_count', new_count)
        await query.answer(f"✅ Message count set to {new_count}!")
        
        keyboard = [
            [InlineKeyboardButton(f"🗑️ Auto Delete: {'✅' if settings['auto_delete_enabled'] else '❌'}", 
                                callback_data=f"setting_autodel_{group_id}")],
            [InlineKeyboardButton(f"✏️ Message Count: {new_count}", 
                                callback_data=f"setting_editautocount_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return
    
    if action == "editspamcount":
        keyboard = [
            [InlineKeyboardButton("3", callback_data=f"setting_setspamcount_{group_id}_3"),
             InlineKeyboardButton("5", callback_data=f"setting_setspamcount_{group_id}_5")],
            [InlineKeyboardButton("7", callback_data=f"setting_setspamcount_{group_id}_7"),
             InlineKeyboardButton("10", callback_data=f"setting_setspamcount_{group_id}_10")],
            [InlineKeyboardButton("15", callback_data=f"setting_setspamcount_{group_id}_15"),
             InlineKeyboardButton("20", callback_data=f"setting_setspamcount_{group_id}_20")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"setting_cat_security_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"✏️ *Select Spam Threshold*\n\n"
            f"Current: {settings['spam_threshold']} messages\n\n"
            f"Messages in 10 seconds = spam trigger:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        await query.answer()
        return
    
    if action == "setspamcount" and len(parts) >= 4:
        new_threshold = int(parts[3])
        settings['spam_threshold'] = new_threshold
        update_group_setting(group_id, 'spam_threshold', new_threshold)
        await query.answer(f"✅ Spam threshold set to {new_threshold}!")
        
        keyboard = [
            [InlineKeyboardButton(f"🛡️ Spam: {'✅' if settings['spam_protection'] else '❌'}", 
                                callback_data=f"setting_spam_{group_id}"),
             InlineKeyboardButton(f"🚫 Flood: {'✅' if settings['antiflood_enabled'] else '❌'}", 
                                callback_data=f"setting_antiflood_{group_id}")],
            [InlineKeyboardButton(f"✏️ Threshold: {new_threshold}", 
                                callback_data=f"setting_editspamcount_{group_id}")],
            [InlineKeyboardButton(f"Admin Spam: {'✅' if settings['delete_admin_spam'] else '❌'}", 
                                callback_data=f"setting_adminspam_{group_id}")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data=f"setting_menu_{group_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        return


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    # Handle settings callbacks
    if query.data.startswith("setting_"):
        await handle_setting_callback(update, context)
        return
    
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
            [
                InlineKeyboardButton("⚙️ Group Settings", callback_data="show_settings_info"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            START_TEXT,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
        )
    
    elif query.data == "show_settings_info":
        # Load user's groups from database
        user_groups = []
        for group_id, group_data in GROUPS_DATABASE.items():
            if group_id < 0:  # Negative IDs are groups
                user_groups.append((group_id, group_data.get('title', 'Unknown Group')))
        
        if not user_groups:
            keyboard = [[InlineKeyboardButton("🏠 Back to Start", callback_data="start")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "⚙️ *Group Settings*\n\n"
                "Abhi tak kisi group mein nahi ho! 😊\n\n"
                "Mujhe apne group mein add karo:\n"
                "1. Group mein jao\n"
                "2. ➕ Add Members pe click karo\n"
                "3. @AnimxClanBot search karo aur add karo\n"
                "4. Mujhe admin banao\n"
                "5. Group mein /settings use karo!\n\n"
                "💡 *Note:* Settings sirf group admins access kar sakte hain.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
        else:
            # Show list of groups
            keyboard = []
            for group_id, group_title in user_groups[:10]:  # Show max 10 groups
                keyboard.append([InlineKeyboardButton(
                    f"⚙️ {group_title}", 
                    callback_data=f"groupsetting_{group_id}"
                )])
            keyboard.append([InlineKeyboardButton("🏠 Back to Start", callback_data="start")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "⚙️ *Group Settings*\n\n"
                "Apne group ko select karo settings change karne ke liye:\n\n"
                "💡 *Note:* Settings change karne ke liye:\n"
                "• Group mein jao\n"
                "• /settings command use karo\n"
                "• Sirf admins access kar sakte hain!",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
    
    elif query.data.startswith("groupsetting_"):
        group_id = int(query.data.split("_")[1])
        group_name = GROUPS_DATABASE.get(group_id, {}).get('title', 'Group')
        
        keyboard = [[InlineKeyboardButton("🔙 Back to Groups", callback_data="show_settings_info")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"⚙️ *Settings for {group_name}*\n\n"
            f"Group settings sirf group chat mein access kar sakte ho! 😊\n\n"
            f"Kaise use karein:\n"
            f"1. `{group_name}` group mein jao\n"
            f"2. /settings command type karo\n"
            f"3. Settings customize karo\n\n"
            f"💡 Tumhe us group ka admin hona chahiye!",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
        )
        
        await query.edit_message_text(
            START_TEXT,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
        )


# ========================= MESSAGE HANDLERS ========================= #

async def welcome_new_members_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Rose-style welcome handler for new members."""
    if not update.message or not update.message.new_chat_members or not update.effective_chat:
        return
    if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
        return

    chat = update.effective_chat
    await _register_group(chat.id, chat)

    if not get_group_setting(chat.id, "welcome_message"):
        return

    names = []
    for member in update.message.new_chat_members:
        names.append(member.first_name or "User")
        await _register_user(member.id, member.username, member.first_name)
        await _register_group_member(chat.id, member.id, member.username, member.first_name)
        BOT_DB.log_activity("member_joined", user_id=member.id, group_id=chat.id)
        await _send_log_to_channel(
            context,
            (
                "GROUP_MEMBER_JOIN\n"
                f"Chat: {chat.title or 'Group'}\n"
                f"Chat ID: {chat.id}\n"
                f"User: {_safe_user_mention(member.username, member.first_name)}\n"
                f"User ID: {member.id}\n"
                f"At: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        )

    welcome_text = (
        f"Welcome {', '.join(names)}!\n"
        f"Group: {chat.title or 'Group'}\n"
        "Use /help for commands and /song <name> for music."
    )
    await update.effective_message.reply_text(welcome_text)


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
    BOT_DB.log_activity("private_message", user_id=user_id, metadata={"text": user_message[:200]})
    await _send_log_to_channel(
        context,
        (
            "PRIVATE_USE\n"
            f"User: {_safe_user_mention(update.effective_user.username, update.effective_user.first_name)}\n"
            f"User ID: {user_id}\n"
            f"Message: {user_message[:300]}\n"
            f"At: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ),
    )
    
    # Detect language preference from message
    if "english me bolo" in user_message.lower() or "speak in english" in user_message.lower():
        LANGUAGE_PREFERENCES[user_id] = "english"
        logger.info(f"User {user_id} set language to: english")
    elif "hindi me bolo" in user_message.lower() or "hindi mein baat karo" in user_message.lower():
        LANGUAGE_PREFERENCES[user_id] = "hinglish"
        logger.info(f"User {user_id} set language to: hinglish")
    
    # Build system prompt with language preference
    user_lang = LANGUAGE_PREFERENCES.get(user_id, "auto")
    lang_instruction = f"\nUSER LANGUAGE PREFERENCE: {user_lang.upper()}"
    if user_lang == "english":
        lang_instruction += "\nReply ONLY in English."
    elif user_lang == "hinglish":
        lang_instruction += "\nReply in Hinglish (mix of Hindi and English)."
    else:
        lang_instruction += "\nReply in the same language used by the user message."
    
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
        
        # Bot link auto-delete (configurable in /settings)
        group_id = update.effective_chat.id
        if get_group_setting(group_id, "remove_bot_links") and _contains_bot_link(message_text):
            try:
                await update.message.delete()
                BOT_DB.log_activity(
                    "bot_link_deleted",
                    user_id=update.effective_user.id if update.effective_user else None,
                    group_id=group_id,
                    metadata={"text": message_text[:300]},
                )
            except Exception as e:
                logger.warning(f"Could not delete bot link message: {e}")
            return
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
        BOT_DB.log_activity("group_message", user_id=user_id, group_id=group_id, metadata={"text": message_text[:200]})
        await _send_log_to_channel(
            context,
            (
                "GROUP_USE\n"
                f"Chat: {chat_title}\n"
                f"Chat ID: {group_id}\n"
                f"User: {_safe_user_mention(username, first_name)}\n"
                f"User ID: {user_id}\n"
                f"Message: {message_text[:300]}\n"
                f"At: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        )
        
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
        user_lang = LANGUAGE_PREFERENCES.get(user_id, "auto")
        lang_instruction = f"\n[User language: {user_lang.upper()}]"
        if user_lang == "english":
            lang_instruction += " Reply ONLY in English."
        elif user_lang == "hinglish":
            lang_instruction += " Reply in Hinglish."
        else:
            lang_instruction += " Reply in the same language used by the user message."
        
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
    logger.info("Bot shutting down...")
    global VC_MANAGER
    if VC_MANAGER is not None:
        try:
            await VC_MANAGER.stop()
        except Exception as e:
            logger.warning(f"VC manager shutdown warning: {e}")

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

    if OPENAI_API_KEY:
        logger.info(f"✅ OpenAI: Enabled (Model: {OPENAI_MODEL})")
    else:
        logger.info("❌ OpenAI: Disabled")
    
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
    application.add_handler(CommandHandler("settings", settings_command))
    
    # Admin analytics commands
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("users", users_command))
    application.add_handler(CommandHandler("groups", groups_command))
    application.add_handler(CommandHandler("members", members_command))
    application.add_handler(CommandHandler("dashboard", dashboard_command))
    application.add_handler(CommandHandler("channelstats", channelstats_command))
    application.add_handler(CommandHandler("chatid", chatid_command))
    
    # Song download commands
    application.add_handler(CommandHandler("play", play_command))
    application.add_handler(CommandHandler("song", song_command))
    application.add_handler(CommandHandler("download", download_command))
    application.add_handler(CommandHandler("yt", yt_command))
    application.add_handler(CommandHandler("vplay", vplay_command))
    application.add_handler(CommandHandler("vqueue", vqueue_command))
    application.add_handler(CommandHandler("vskip", vskip_command))
    application.add_handler(CommandHandler("vstop", vstop_command))
    
    # Broadcast command (admin only)
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("broadcast_now", broadcast_content))
    application.add_handler(CommandHandler("broadcastsong", broadcastsong_command))
    
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
    application.add_handler(CommandHandler("warn", warn_command))
    application.add_handler(CommandHandler("warnings", warnings_command))
    application.add_handler(CommandHandler("resetwarn", resetwarn_command))
    application.add_handler(CommandHandler("mute", mute_command))
    application.add_handler(CommandHandler("unmute", unmute_command))
    application.add_handler(CommandHandler("promote", promote_command))
    application.add_handler(CommandHandler("demote", demote_command))
    application.add_handler(CommandHandler("pin", pin_command))
    application.add_handler(CommandHandler("unpin", unpin_command))
    
    # Register callback handlers for inline buttons
    # Settings callbacks (higher priority)
    application.add_handler(CallbackQueryHandler(handle_setting_callback, pattern=r"^setting_"))
    # General button callbacks
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Register chat member handler for group tracking
    application.add_handler(ChatMemberHandler(my_chat_member_handler, ChatMemberHandler.MY_CHAT_MEMBER))
    
    # Register message handlers
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.NEW_CHAT_MEMBERS,
            welcome_new_members_handler,
        )
    )

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













