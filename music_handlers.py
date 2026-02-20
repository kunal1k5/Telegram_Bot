from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Set

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatType


class MusicHandlers:
    def __init__(self, owner_username: str, channel_username: str) -> None:
        self.owner_username = owner_username
        self.channel_username = channel_username
        self.loop_enabled_chats: Set[int] = set()
        self.chat_theme: dict[int, str] = {}
        self.theme_db_file = Path("music_theme_prefs.json")
        self._themes = ("rose", "glass", "minimal")
        self._theme_labels = {
            "rose": "Rose",
            "glass": "Glass",
            "minimal": "Minimal",
        }
        self._load_theme_prefs()

    def _load_theme_prefs(self) -> None:
        try:
            if not self.theme_db_file.exists():
                return
            data = json.loads(self.theme_db_file.read_text(encoding="utf-8"))
            raw_map = data.get("chat_theme", {})
            loaded: dict[int, str] = {}
            for key, theme in raw_map.items():
                try:
                    chat_id = int(key)
                except Exception:
                    continue
                if theme in self._themes:
                    loaded[chat_id] = theme
            self.chat_theme = loaded
        except Exception:
            self.chat_theme = {}

    def _save_theme_prefs(self) -> None:
        try:
            payload = {
                "chat_theme": {str(k): v for k, v in self.chat_theme.items()},
                "count": len(self.chat_theme),
            }
            self.theme_db_file.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def format_vc_duration(seconds: Optional[int]) -> str:
        if not seconds or seconds <= 0:
            return "Live"
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def _format_clock(seconds: int) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"

    def _progress_line(self, duration_seconds: Optional[int], elapsed_seconds: int = 0, width: int = 14) -> str:
        if not duration_seconds or duration_seconds <= 0:
            return "LIVE  ○" + ("─" * width)

        elapsed = max(0, min(int(elapsed_seconds), int(duration_seconds)))
        ratio = elapsed / max(1, int(duration_seconds))
        marker_at = min(width, max(0, int(round(ratio * width))))
        bar = ("─" * marker_at) + "○" + ("─" * (width - marker_at))
        return f"{self._format_clock(elapsed)} {bar} {self._format_clock(int(duration_seconds))}"

    @staticmethod
    def _trim_title(title: str, limit: int = 42) -> str:
        clean = (title or "Unknown Track").strip()
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "..."

    @staticmethod
    def _safe_requester(name: str) -> str:
        return (name or "User").strip()[:32]

    @staticmethod
    def _guess_artist_from_title(title: str) -> str:
        clean = (title or "").strip()
        if " - " in clean:
            return clean.split(" - ", 1)[0].strip()[:36] or "Unknown Artist"
        if "|" in clean:
            return clean.split("|", 1)[0].strip()[:36] or "Unknown Artist"
        return "Unknown Artist"

    def _get_theme(self, chat_id: Optional[int]) -> str:
        if not chat_id:
            return "rose"
        theme = self.chat_theme.get(chat_id, "rose")
        return theme if theme in self._themes else "rose"

    def _cycle_theme(self, chat_id: int) -> str:
        current = self._get_theme(chat_id)
        idx = self._themes.index(current)
        nxt = self._themes[(idx + 1) % len(self._themes)]
        self.chat_theme[chat_id] = nxt
        self._save_theme_prefs()
        return nxt

    def vc_now_playing_card(
        self,
        track: Any,
        requested_by: str,
        download_mode: bool = False,
        chat_id: Optional[int] = None,
    ) -> str:
        mode_badge = " [DOWNLOAD MODE]" if download_mode else ""
        duration_seconds = getattr(track, "duration", None)
        title = self._trim_title(getattr(track, "title", "Unknown Track"), 52)
        artist = self._guess_artist_from_title(getattr(track, "title", "Unknown Track"))
        requester = self._safe_requester(requested_by)
        theme = self._get_theme(chat_id)
        theme_name = self._theme_labels.get(theme, "Rose")
        progress = self._progress_line(duration_seconds, width=16).replace(" ○", " ◉")

        if theme == "glass":
            return (
                f"✦ GLASS PLAYER{mode_badge}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎵 {title}\n"
                f"🫧 Artist: {artist}\n"
                f"⏱ Duration: {self.format_vc_duration(duration_seconds)}\n"
                f"🌹 Requested: {requester}\n"
                f"🎨 Theme: {theme_name}\n"
                f"{progress}"
            )
        if theme == "minimal":
            return (
                f"NOW PLAYING{mode_badge}\n"
                f"🎵 {title}\n"
                f"🫧 Artist: {artist}\n"
                f"⏱ Duration: {self.format_vc_duration(duration_seconds)}\n"
                f"🌹 Requested: {requester}\n"
                f"🎨 Theme: {theme_name}\n"
                f"{progress}"
            )
        return (
            f"» Started Streaming 🎧{mode_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"❄️ TITLE      : {title}\n"
            f"🎙 ARTIST     : {artist}\n"
            f"🕓 DURATION   : {self.format_vc_duration(duration_seconds)}\n"
            f"🥀 REQUESTED  : {requester}\n"
            f"🎨 THEME      : {theme_name}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{progress}"
        )

    def vc_queue_card(self, track: Any, position: int, download_mode: bool = False) -> str:
        mode_badge = " [DOWNLOAD MODE]" if download_mode else ""
        title = self._trim_title(getattr(track, "title", "Unknown Track"), 52)
        duration = self.format_vc_duration(getattr(track, "duration", None))
        requester = self._safe_requester(getattr(track, "requested_by", "User"))
        return (
            f"📥 ADDED TO QUEUE{mode_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎵 TITLE      : {title}\n"
            f"🕓 DURATION   : {duration}\n"
            f"🔢 POSITION   : #{position}\n"
            f"🥀 REQUESTED  : {requester}"
        )

    def vc_player_keyboard(self, chat_id: int, is_paused: bool = False) -> InlineKeyboardMarkup:
        play_pause_label = "▶️" if is_paused else "⏸️"
        loop_label = "🔁 ON" if chat_id in self.loop_enabled_chats else "🔁 OFF"

        rows = [
            [
                InlineKeyboardButton("⏮️", callback_data="vcctl_prev"),
                InlineKeyboardButton(play_pause_label, callback_data="vcctl_pause_resume"),
                InlineKeyboardButton("⏭️", callback_data="vcctl_next"),
                InlineKeyboardButton(loop_label, callback_data="vcctl_loop"),
                InlineKeyboardButton("⏹️", callback_data="vcctl_stop"),
            ],
        ]

        support_user = (self.channel_username or "").lstrip("@")
        dev_user = (self.owner_username or "").lstrip("@")
        support_btn = (
            InlineKeyboardButton("🌹 Support", url=f"https://t.me/{support_user}")
            if support_user
            else InlineKeyboardButton("🌹 Support", callback_data="vcctl_refresh")
        )
        dev_btn = (
            InlineKeyboardButton("✨ Dev", url=f"https://t.me/{dev_user}")
            if dev_user
            else InlineKeyboardButton("✨ Dev", callback_data="vcctl_theme")
        )
        rows.append(
            [
                support_btn,
                dev_btn,
                InlineKeyboardButton("❌ Close", callback_data="vcctl_close"),
            ]
        )
        return InlineKeyboardMarkup(rows)

    def vc_queue_preview(self, queue: List[Any], limit: int = 7) -> str:
        if not queue:
            return "Queue is empty."

        lines = ["🎼 UPCOMING TRACKS"]
        for i, item in enumerate(queue[:limit], 1):
            title = self._trim_title(getattr(item, "title", "Unknown"), 34)
            duration = self.format_vc_duration(getattr(item, "duration", None))
            requester = self._safe_requester(getattr(item, "requested_by", "User"))
            lines.append(f"{i}. {title}")
            lines.append(f"   ⏱ {duration} • by {requester}")

        remaining = len(queue) - limit
        if remaining > 0:
            lines.append(f"...and {remaining} more")

        return "\n".join(lines)

    async def _edit_callback_message(self, query: CallbackQuery, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
        if query.message and query.message.photo:
            await query.edit_message_caption(caption=text, reply_markup=reply_markup)
        else:
            await query.edit_message_text(text=text, reply_markup=reply_markup)

    async def send_vc_player_card(
        self,
        update: Update,
        status_message: Message,
        track: Any,
        requested_by: str,
        download_mode: bool,
        get_vc_manager: Callable[[], Awaitable[Any]],
    ) -> None:
        caption = self.vc_now_playing_card(track, requested_by, download_mode=download_mode)
        vc = await get_vc_manager()
        chat_id = update.effective_chat.id
        caption = self.vc_now_playing_card(track, requested_by, download_mode=download_mode, chat_id=chat_id)
        keyboard = self.vc_player_keyboard(chat_id, vc.is_paused(chat_id))
        thumb = getattr(track, "thumbnail", None)

        try:
            if thumb:
                await status_message.delete()
                await update.effective_message.reply_photo(
                    photo=thumb,
                    caption=caption,
                    reply_markup=keyboard,
                )
                return
        except Exception:
            pass

        await status_message.edit_text(caption, reply_markup=keyboard)

    async def update_vc_player_callback_message(
        self,
        query: CallbackQuery,
        track: Any,
        paused: bool = False,
    ) -> None:
        chat_id = query.message.chat_id if query.message else 0
        caption = self.vc_now_playing_card(
            track,
            track.requested_by,
            download_mode=getattr(track, "is_local", False),
            chat_id=chat_id,
        )
        keyboard = self.vc_player_keyboard(chat_id, paused)
        try:
            await self._edit_callback_message(query, caption, reply_markup=keyboard)
        except Exception:
            pass

    async def vstop_command(
        self,
        update: Update,
        register_user_from_update: Callable[[Update], Awaitable[None]],
        get_vc_manager: Callable[[], Awaitable[Any]],
    ) -> None:
        await register_user_from_update(update)
        if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
            await update.effective_message.reply_text("/vstop works in groups only.")
            return

        try:
            vc = await get_vc_manager()
            chat_id = update.effective_chat.id
            await vc.stop_chat(chat_id)
            self.loop_enabled_chats.discard(chat_id)
            await update.effective_message.reply_text("⏹ Voice chat playback stopped and queue cleared.")
        except Exception as e:
            await update.effective_message.reply_text(f"VC stop failed: {e}")

    async def vskip_command(
        self,
        update: Update,
        register_user_from_update: Callable[[Update], Awaitable[None]],
        get_vc_manager: Callable[[], Awaitable[Any]],
    ) -> None:
        await register_user_from_update(update)
        if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
            await update.effective_message.reply_text("/vskip works in groups only.")
            return

        try:
            vc = await get_vc_manager()
            chat_id = update.effective_chat.id
            now_track = vc.get_now_playing(chat_id)
            if chat_id in self.loop_enabled_chats and now_track:
                vc.queues.setdefault(chat_id, []).append(now_track)

            next_track = await vc.skip(chat_id)
            if not next_track:
                await update.effective_message.reply_text("Queue empty. Stopped current playback.")
                return

            await update.effective_message.reply_text(
                "⏭️ Track Skipped\n\n"
                + self.vc_now_playing_card(next_track, next_track.requested_by, chat_id=chat_id),
            )
        except Exception as e:
            await update.effective_message.reply_text(f"VC skip failed: {e}")

    async def vqueue_command(
        self,
        update: Update,
        register_user_from_update: Callable[[Update], Awaitable[None]],
        get_vc_manager: Callable[[], Awaitable[Any]],
    ) -> None:
        await register_user_from_update(update)
        if update.effective_chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
            await update.effective_message.reply_text("/vqueue works in groups only.")
            return

        try:
            vc = await get_vc_manager()
            chat_id = update.effective_chat.id
            now_track = vc.get_now_playing(chat_id)
            queue = vc.get_queue(chat_id)

            lines = ["🎼 RESSO STYLE QUEUE"]
            if now_track:
                lines.append(
                    "\n▶️ NOW PLAYING"
                    f"\n🎵 {self._trim_title(now_track.title, 52)}"
                    f"\n👤 {self._safe_requester(now_track.requested_by)}"
                    f"\n🕓 {self.format_vc_duration(now_track.duration)}"
                )
            else:
                lines.append("\n▶️ NOW PLAYING\nNothing")

            if not queue:
                lines.append("\n📭 Queue: Empty")
            else:
                lines.append("\n📜 Queue List:")
                for i, item in enumerate(queue[:10], 1):
                    lines.append(
                        f"{i}. {self._trim_title(item.title, 42)}"
                        f" • {self.format_vc_duration(item.duration)}"
                        f" • by {self._safe_requester(item.requested_by)}"
                    )

            await update.effective_message.reply_text("\n".join(lines))
        except Exception as e:
            await update.effective_message.reply_text(f"VC queue failed: {e}")

    async def handle_vc_callback(
        self,
        query: CallbackQuery,
        get_vc_manager: Callable[[], Awaitable[Any]],
    ) -> bool:
        if not query.data.startswith("vcctl_"):
            return False

        chat_id = query.message.chat_id if query.message else None
        if not chat_id:
            await query.answer("Chat not found.", show_alert=True)
            return True

        try:
            vc = await get_vc_manager()
            action = query.data.split("_", 1)[1]

            if action == "pause_resume":
                if vc.is_paused(chat_id):
                    await vc.resume_chat(chat_id)
                    now_track = vc.get_now_playing(chat_id)
                    if now_track:
                        await self.update_vc_player_callback_message(query, now_track, paused=False)
                    await query.answer("▶ Resumed")
                else:
                    await vc.pause_chat(chat_id)
                    now_track = vc.get_now_playing(chat_id)
                    if now_track:
                        await self.update_vc_player_callback_message(query, now_track, paused=True)
                    await query.answer("⏸ Paused")
                return True

            if action == "prev":
                prev_track = await vc.play_previous(chat_id)
                if not prev_track:
                    await query.answer("No previous track.", show_alert=True)
                    return True
                await self.update_vc_player_callback_message(query, prev_track, paused=False)
                await query.answer("⏮ Previous")
                return True

            if action in ("next", "skip"):
                now_track = vc.get_now_playing(chat_id)
                if chat_id in self.loop_enabled_chats and now_track:
                    vc.queues.setdefault(chat_id, []).append(now_track)
                next_track = await vc.skip(chat_id)
                if not next_track:
                    try:
                        await self._edit_callback_message(query, "⏹ Playback stopped. Queue is empty.")
                    except Exception:
                        pass
                    await query.answer("Queue ended")
                    return True
                await self.update_vc_player_callback_message(query, next_track, paused=False)
                await query.answer("⏭ Next")
                return True

            if action == "stop":
                await vc.stop_chat(chat_id)
                self.loop_enabled_chats.discard(chat_id)
                try:
                    await self._edit_callback_message(query, "⏹ Playback stopped and queue cleared.")
                except Exception:
                    pass
                await query.answer("Stopped")
                return True

            if action == "queue":
                queue = vc.get_queue(chat_id)
                await query.answer(self.vc_queue_preview(queue), show_alert=True)
                return True

            if action == "refresh":
                now_track = vc.get_now_playing(chat_id)
                if not now_track:
                    await query.answer("No active track.", show_alert=True)
                    return True
                await self.update_vc_player_callback_message(query, now_track, paused=vc.is_paused(chat_id))
                await query.answer("🔄 Refreshed")
                return True

            if action == "shuffle":
                queue_ref = vc.queues.get(chat_id, [])
                if len(queue_ref) < 2:
                    await query.answer("Need at least 2 tracks in queue.", show_alert=True)
                    return True
                random.shuffle(queue_ref)
                await query.answer("🔀 Queue shuffled")
                now_track = vc.get_now_playing(chat_id)
                if now_track:
                    await self.update_vc_player_callback_message(query, now_track, paused=vc.is_paused(chat_id))
                return True

            if action == "loop":
                if chat_id in self.loop_enabled_chats:
                    self.loop_enabled_chats.discard(chat_id)
                    status = "🔁 Loop OFF"
                else:
                    self.loop_enabled_chats.add(chat_id)
                    status = "🔁 Loop ON"
                now_track = vc.get_now_playing(chat_id)
                if now_track:
                    await self.update_vc_player_callback_message(query, now_track, paused=vc.is_paused(chat_id))
                await query.answer(status)
                return True

            if action == "theme":
                new_theme = self._cycle_theme(chat_id)
                now_track = vc.get_now_playing(chat_id)
                if now_track:
                    await self.update_vc_player_callback_message(query, now_track, paused=vc.is_paused(chat_id))
                await query.answer(f"🎨 Theme: {self._theme_labels.get(new_theme, 'Rose')}")
                return True

            if action == "close":
                try:
                    await vc.stop_chat(chat_id)
                    self.loop_enabled_chats.discard(chat_id)
                    if query.message:
                        await query.message.delete()
                    await query.answer("VC closed")
                except Exception:
                    await query.answer("Could not close panel.", show_alert=True)
                return True
        except Exception as e:
            await query.answer(f"VC control failed: {e}", show_alert=True)
            return True

        return False
