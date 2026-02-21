from __future__ import annotations

from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def get_dynamic_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "🌅 Good Morning"
    if 12 <= hour < 18:
        return "🌞 Good Afternoon"
    if 18 <= hour < 23:
        return "🌙 Good Evening"
    return "🌌 Late Night Vibes"


def premium_start_caption(user_name: str = "Music Lover") -> str:
    greeting = get_dynamic_greeting()
    safe_name = (user_name or "Music Lover").strip()
    return (
        f"✨ <b>HEY BABY {safe_name}</b> NICE TO MEET YOU 🌹\n\n"
        "◎ <b>THIS IS 『ANIMX MUSIC』</b>\n\n"
        "➤ A premium designed music player bot for Telegram groups & channels.\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🎧 HD Voice Chat Streaming\n"
        "🚀 Fast • Smart • Always Active\n"
        "💬 Chat Naturally Like a Friend\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"{greeting} 💖"
    )


def premium_start_buttons(
    bot_username: str,
    channel_username: str,
    support_url: str = "https://t.me",
    source_url: str = "https://github.com",
    contact_url: str = "https://t.me",
) -> InlineKeyboardMarkup:
    _ = channel_username
    _ = support_url
    _ = source_url
    _ = contact_url
    bot_name = bot_username.lstrip("@")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("💬 Chat With Me", url=f"https://t.me/{bot_name}"),
                InlineKeyboardButton("➕ Add To Group", url=f"https://t.me/{bot_name}?startgroup=true"),
            ],
            [
                InlineKeyboardButton("📖 Help", callback_data="help"),
                InlineKeyboardButton("🎙 VC Guide", callback_data="vc_guide"),
            ],
            [
                InlineKeyboardButton("📢 Channel", url="https://t.me/AnimxClan_Channel"),
                InlineKeyboardButton("⚙ Settings", callback_data="settings"),
            ],
            [
                InlineKeyboardButton("📩 Contact / Promotion", callback_data="contact_promo"),
            ],
        ]
    )
