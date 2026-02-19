import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VCTrack:
    title: str
    webpage_url: str
    stream_url: str
    requested_by: str


class VCManager:
    """Voice chat music manager using Pyrogram + PyTgCalls."""

    def __init__(self, api_id: int, api_hash: str, assistant_session: str) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.assistant_session = assistant_session
        self._lock = asyncio.Lock()
        self._ready = False

        self._assistant: Any = None
        self._calls: Any = None
        self._audio_piped_cls: Any = None
        self._yt_dlp: Any = None

        self.queues: dict[int, list[VCTrack]] = {}
        self.now_playing: dict[int, VCTrack] = {}
        self.active_calls: set[int] = set()

    async def start(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return

            if not self.api_id or not self.api_hash or not self.assistant_session:
                raise RuntimeError(
                    "VC config missing. Set API_ID, API_HASH, and ASSISTANT_SESSION."
                )

            try:
                from pyrogram import Client
                from pytgcalls import PyTgCalls
                from pytgcalls.types.input_stream import AudioPiped
                import yt_dlp
            except Exception as e:
                raise RuntimeError(
                    "VC dependencies missing. Install pyrogram, tgcrypto, py-tgcalls, yt-dlp. "
                    f"Import error: {type(e).__name__}: {e}"
                ) from e

            self._assistant = Client(
                name="animx_vc_assistant",
                api_id=self.api_id,
                api_hash=self.api_hash,
                session_string=self.assistant_session,
                no_updates=True,
            )
            await self._assistant.start()

            self._calls = PyTgCalls(self._assistant)
            await self._calls.start()

            self._audio_piped_cls = AudioPiped
            self._yt_dlp = yt_dlp
            self._ready = True

    async def stop(self) -> None:
        if not self._ready:
            return
        async with self._lock:
            if not self._ready:
                return
            try:
                await self._calls.stop()
            except Exception:
                pass
            try:
                await self._assistant.stop()
            except Exception:
                pass
            self._ready = False

    async def get_assistant_identity(self) -> tuple[Optional[int], Optional[str]]:
        await self.start()
        try:
            me = await self._assistant.get_me()
            return getattr(me, "id", None), getattr(me, "username", None)
        except Exception:
            return None, None

    async def is_assistant_in_chat(self, chat_id: int) -> bool:
        await self.start()
        me_id, _ = await self.get_assistant_identity()
        if not me_id:
            return False
        try:
            await self._assistant.get_chat_member(chat_id, me_id)
            return True
        except Exception:
            return False

    async def join_chat_via_invite(self, invite_link: str) -> None:
        await self.start()
        await self._assistant.join_chat(invite_link)

    def _is_url(self, text: str) -> bool:
        return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))

    def _resolve_track_sync(self, query: str, requested_by: str) -> VCTrack:
        ydl_opts = {
            "quiet": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "default_search": "ytsearch1",
            "extract_flat": False,
            "skip_download": True,
        }

        search_target = query if self._is_url(query) else f"ytsearch1:{query}"
        with self._yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_target, download=False)
            if not info:
                raise RuntimeError("No track found")
            if "entries" in info:
                entries = info.get("entries") or []
                if not entries:
                    raise RuntimeError("No search result entries")
                info = entries[0]

            webpage_url = info.get("webpage_url") or info.get("url")
            title = info.get("title") or "Unknown Title"
            if not webpage_url:
                raise RuntimeError("Could not resolve webpage url")

            detailed = ydl.extract_info(webpage_url, download=False)
            if not detailed:
                raise RuntimeError("Could not resolve stream info")
            stream_url = detailed.get("url")
            if not stream_url:
                raise RuntimeError("Could not resolve audio stream url")

            return VCTrack(
                title=title[:120],
                webpage_url=webpage_url,
                stream_url=stream_url,
                requested_by=requested_by,
            )

    async def resolve_track(self, query: str, requested_by: str) -> VCTrack:
        return await asyncio.to_thread(self._resolve_track_sync, query, requested_by)

    async def _play_track(self, chat_id: int, track: VCTrack) -> None:
        await self.start()
        stream = self._audio_piped_cls(track.stream_url)

        if chat_id not in self.active_calls:
            try:
                await self._calls.join_group_call(chat_id, stream)
                self.active_calls.add(chat_id)
                self.now_playing[chat_id] = track
                return
            except Exception:
                # If already in call or API variant mismatch, continue fallback.
                pass

        # fallback stream change methods for compatibility
        try:
            await self._calls.change_stream(chat_id, stream)
        except Exception:
            try:
                await self._calls.play(chat_id, stream)
            except Exception as e:
                raise RuntimeError(f"Could not start stream: {e}") from e

        self.active_calls.add(chat_id)
        self.now_playing[chat_id] = track

    async def enqueue_or_play(self, chat_id: int, query: str, requested_by: str) -> tuple[str, VCTrack]:
        track = await self.resolve_track(query, requested_by)
        if chat_id in self.now_playing:
            self.queues.setdefault(chat_id, []).append(track)
            return "queued", track

        await self._play_track(chat_id, track)
        return "playing", track

    async def skip(self, chat_id: int) -> Optional[VCTrack]:
        queue = self.queues.get(chat_id, [])
        if not queue:
            self.now_playing.pop(chat_id, None)
            return None

        next_track = queue.pop(0)
        await self._play_track(chat_id, next_track)
        return next_track

    async def stop_chat(self, chat_id: int) -> None:
        if not self._ready:
            return
        try:
            await self._calls.leave_group_call(chat_id)
        except Exception:
            pass
        self.queues.pop(chat_id, None)
        self.now_playing.pop(chat_id, None)
        self.active_calls.discard(chat_id)

    def get_queue(self, chat_id: int) -> list[VCTrack]:
        return list(self.queues.get(chat_id, []))

    def get_now_playing(self, chat_id: int) -> Optional[VCTrack]:
        return self.now_playing.get(chat_id)
