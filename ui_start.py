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
        "◎ THIS IS <b>『ANIMX MUSIC』</b>\n\n"
        "➤ A premium designed music player bot for Telegram groups & channels.\n\n"
        "🎧 HD Voice Chat Streaming\n"
        "🚀 Fast • Smart • Always Active\n"
        "💬 Chat Naturally Like a Friend\n\n"
        f"{greeting} 💖"
    )


def premium_start_buttons(
    bot_username: str,
    channel_username: str,
    support_url: str = "https://t.me",
    source_url: str = "https://github.com",
) -> InlineKeyboardMarkup:
    _ = channel_username
    bot_name = bot_username.lstrip("@")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "◎ ADD ME TO YOUR CHAT ◎",
                    url=f"https://t.me/{bot_name}?startgroup=true",
                )
            ],
            [InlineKeyboardButton("HELP AND COMMANDS", callback_data="help")],
            [
                InlineKeyboardButton("SUPPORT", url=support_url),
                InlineKeyboardButton("SOURCE", url=source_url),
            ],
            [InlineKeyboardButton("• BOT | YT-API INFO •", callback_data="info")],
        ]
    )
