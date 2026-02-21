from __future__ import annotations

from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def get_dynamic_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    if 12 <= hour < 18:
        return "Good Afternoon"
    if 18 <= hour < 23:
        return "Good Evening"
    return "Late Night Vibes"


def premium_start_caption(user_name: str = "Music Lover") -> str:
    greeting = get_dynamic_greeting()
    safe_name = (user_name or "Music Lover").strip()
    return (
        "<b>BABY • Premium Music AI</b>\n\n"
        f"{greeting}, <b>{safe_name}</b>\n\n"
        "I am your cinematic music and chat companion.\n"
        "Built for vibes, designed for performance.\n\n"
        "Fast • Smart • Always Active\n"
        "HD Voice Chat Streaming\n"
        "Chat naturally like a friend\n"
        "Advanced group controls\n\n"
        "Ready to feel the vibe?"
    )


def premium_start_buttons(bot_username: str, channel_username: str) -> InlineKeyboardMarkup:
    bot_name = bot_username.lstrip("@")
    channel_name = channel_username.lstrip("@")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Chat With Me", url=f"https://t.me/{bot_name}"),
                InlineKeyboardButton("Add To Group", url=f"https://t.me/{bot_name}?startgroup=true"),
            ],
            [
                InlineKeyboardButton("Help", callback_data="help"),
                InlineKeyboardButton("VC Guide", callback_data="vc_guide"),
            ],
            [
                InlineKeyboardButton("Channel", url=f"https://t.me/{channel_name}"),
                InlineKeyboardButton("Settings", callback_data="show_settings_info"),
            ],
        ]
    )

