from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def cinematic_caption(title: str, duration: str, requester: str) -> str:
    safe_title = (title or "Unknown Track").strip()
    safe_duration = (duration or "Live").strip()
    safe_requester = (requester or "User").strip()
    return (
        "<b>ANIMX CINEMA PLAYER</b>\n\n"
        "<b>Now Playing</b>\n"
        f"<b>{safe_title}</b>\n"
        f"Duration: {safe_duration}\n"
        f"Requested by: <b>{safe_requester}</b>\n\n"
        "Feel the vibe..."
    )


def music_controls(support_url: str, dev_url: str, is_paused: bool = False) -> InlineKeyboardMarkup:
    pause_label = "▶" if is_paused else "⏸"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("⏮", callback_data="vcctl_prev"),
                InlineKeyboardButton(pause_label, callback_data="vcctl_pause_resume"),
                InlineKeyboardButton("⏭", callback_data="vcctl_next"),
            ],
            [
                InlineKeyboardButton("🔁", callback_data="vcctl_loop"),
                InlineKeyboardButton("🔀", callback_data="vcctl_shuffle"),
                InlineKeyboardButton("⏹", callback_data="vcctl_stop"),
            ],
            [
                InlineKeyboardButton("Support", url=support_url),
                InlineKeyboardButton("Dev", url=dev_url),
            ],
            [InlineKeyboardButton("Close", callback_data="vcctl_close")],
        ]
    )

