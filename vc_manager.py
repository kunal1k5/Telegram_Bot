
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

logger = logging.getLogger("ANIMX_VC")


@dataclass
class VCTrack:
    title: str
    webpage_url: str
    stream_url: str
    requested_by: str
    is_local: bool = False
    duration: Optional[int] = None
    thumbnail: Optional[str] = None


@dataclass
class VCChatState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: list[VCTrack] = field(default_factory=list)
    history: list[VCTrack] = field(default_factory=list)
    now_playing: Optional[VCTrack] = None
    active_call: bool = False
    paused: bool = False
    track_started_at: float = 0.0
    paused_total_seconds: float = 0.0
    pause_started_at: Optional[float] = None
    play_serial: int = 0
    last_auto_advanced_serial: int = 0
    auto_task: Optional[asyncio.Task] = None
    assistant_verified: bool = False


class VCManager:
    """Production-focused VC manager using Pyrogram + PyTgCalls."""

    def __init__(self, api_id: int, api_hash: str, assistant_session: str) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.assistant_session = assistant_session
        self._boot_lock = asyncio.Lock()
        self._ready = False

        self._assistant: Any = None
        self._calls: Any = None
        self._audio_piped_cls: Any = None
        self._supports_join_api = False
        self._supports_change_api = False
        self._supports_leave_api = False
        self._supports_play_api = False
        self._yt_dlp: Any = None
        self._cookie_file_path: Optional[str] = None

        self._states: dict[int, VCChatState] = {}
        self.queues: dict[int, list[VCTrack]] = {}
        self.now_playing: dict[int, VCTrack] = {}
        self.history: dict[int, list[VCTrack]] = {}
        self.active_calls: set[int] = set()
        self._paused_chats: set[int] = set()

        self._default_track_seconds = int(os.getenv("VC_DEFAULT_TRACK_SECONDS", "240"))
        self._stream_end_handler_registered = False

    def _state(self, chat_id: int) -> VCChatState:
        state = self._states.get(chat_id)
        if state is None:
            state = VCChatState()
            self._states[chat_id] = state
            self.queues[chat_id] = state.queue
            self.history[chat_id] = state.history
        return state

    def _refresh_public_state(self, chat_id: int, state: VCChatState) -> None:
        if state.now_playing:
            self.now_playing[chat_id] = state.now_playing
        else:
            self.now_playing.pop(chat_id, None)
        if state.active_call:
            self.active_calls.add(chat_id)
        else:
            self.active_calls.discard(chat_id)
        if state.paused:
            self._paused_chats.add(chat_id)
        else:
            self._paused_chats.discard(chat_id)

    async def start(self) -> None:
        if self._ready:
            return
        async with self._boot_lock:
            if self._ready:
                return
            if not self.api_id or not self.api_hash or not self.assistant_session:
                raise RuntimeError("VC config missing. Set API_ID, API_HASH, and ASSISTANT_SESSION.")

            try:
                from pyrogram import Client
                from pytgcalls import PyTgCalls
                import yt_dlp
            except Exception as e:
                pkg_bits: list[str] = []
                for pkg in ("py-tgcalls", "pytgcalls"):
                    try:
                        pkg_bits.append(f"{pkg}={version(pkg)}")
                    except PackageNotFoundError:
                        pkg_bits.append(f"{pkg}=not-installed")
                raise RuntimeError(
                    "VC dependencies missing. Install pyrogram, tgcrypto, py-tgcalls, yt-dlp. "
                    f"Import error: {type(e).__name__}: {e}. Detected packages: {', '.join(pkg_bits)}"
                ) from e

            self._assistant = Client(
                name="animx_vc_assistant",
                api_id=self.api_id,
                api_hash=self.api_hash,
                session_string=self.assistant_session,
                no_updates=True,
            )
            try:
                await self._assistant.start()
            except Exception as e:
                err = str(e).lower()
                if "base64" in err or "session" in err:
                    raise RuntimeError(
                        "Invalid ASSISTANT_SESSION. Use a Pyrogram session string "
                        "(Client.export_session_string), not Telethon StringSession."
                    ) from e
                raise RuntimeError(f"Assistant login failed: {e}") from e

            self._calls = PyTgCalls(self._assistant)
            await self._calls.start()

            self._supports_join_api = hasattr(self._calls, "join_group_call")
            self._supports_change_api = hasattr(self._calls, "change_stream")
            self._supports_leave_api = hasattr(self._calls, "leave_group_call")
            self._supports_play_api = hasattr(self._calls, "play")

            audio_import_errors: list[str] = []
            for module_name, class_name in [
                ("pytgcalls.types.input_stream", "AudioPiped"),
                ("pytgcalls.types.input_stream.audio_piped", "AudioPiped"),
                ("pytgcalls.types", "AudioPiped"),
            ]:
                try:
                    module = importlib.import_module(module_name)
                    self._audio_piped_cls = getattr(module, class_name)
                    break
                except Exception as ie:
                    audio_import_errors.append(f"{module_name}.{class_name}: {ie}")

            if self._audio_piped_cls is None and not self._supports_play_api:
                raise RuntimeError(
                    "Could not initialize VC stream type. AudioPiped import failed and play API unavailable. "
                    + " | ".join(audio_import_errors)
                )

            self._yt_dlp = yt_dlp
            self._register_stream_end_handler()
            self._ready = True

    async def stop(self) -> None:
        if not self._ready:
            return
        async with self._boot_lock:
            if not self._ready:
                return
            for state in list(self._states.values()):
                task = state.auto_task
                if task and not task.done():
                    task.cancel()
                state.auto_task = None
            self._states.clear()
            self.queues.clear()
            self.history.clear()
            self.now_playing.clear()
            self.active_calls.clear()
            self._paused_chats.clear()

            try:
                await self._calls.stop()
            except Exception:
                pass
            try:
                await self._assistant.stop()
            except Exception:
                pass
            if self._cookie_file_path and os.path.exists(self._cookie_file_path):
                try:
                    os.remove(self._cookie_file_path)
                except Exception:
                    pass
                self._cookie_file_path = None
            self._stream_end_handler_registered = False
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
        except Exception as e:
            # Strict handling: only treat "not participant" as recoverable.
            try:
                from pyrogram.errors import PeerIdInvalid, UserNotParticipant

                if isinstance(e, UserNotParticipant):
                    return False
                if isinstance(e, PeerIdInvalid):
                    return False
            except Exception:
                pass
            raise

    async def verify_assistant_once(self, chat_id: int) -> bool:
        state = self._state(chat_id)
        if state.assistant_verified:
            return True
        is_in_chat = await self.is_assistant_in_chat(chat_id)
        if is_in_chat:
            state.assistant_verified = True
        return is_in_chat

    async def join_chat_via_invite(self, invite_link: str) -> None:
        await self.start()
        await self._assistant.join_chat(invite_link)
        for state in self._states.values():
            state.assistant_verified = False

    @staticmethod
    def _extract_chat_id_from_update(update: Any) -> Optional[int]:
        for attr in ("chat_id", "group_call_id"):
            value = getattr(update, attr, None)
            if isinstance(value, int):
                return value
        chat = getattr(update, "chat", None)
        if chat is not None:
            chat_id = getattr(chat, "id", None)
            if isinstance(chat_id, int):
                return chat_id
        call = getattr(update, "call", None)
        if call is not None:
            chat_id = getattr(call, "chat_id", None)
            if isinstance(chat_id, int):
                return chat_id
        return None

    async def _on_stream_end(self, chat_id: int) -> None:
        await self._advance_by_auto(chat_id, source="event")

    def _register_stream_end_handler(self) -> None:
        if self._stream_end_handler_registered or not self._calls:
            return

        async def _handler(_: Any, update: Any) -> None:
            chat_id = self._extract_chat_id_from_update(update)
            if chat_id is None:
                return
            await self._on_stream_end(chat_id)

        for hook_name in ("on_stream_end", "on_stream_ended"):
            hook = getattr(self._calls, hook_name, None)
            if not callable(hook):
                continue
            try:
                hook()(_handler)
                self._stream_end_handler_registered = True
                return
            except Exception:
                pass
            try:
                hook(_handler)
                self._stream_end_handler_registered = True
                return
            except Exception:
                pass

    def _cancel_auto_task(self, state: VCChatState) -> None:
        if state.auto_task and not state.auto_task.done():
            state.auto_task.cancel()
        state.auto_task = None

    def _schedule_auto_advance(self, chat_id: int, state: VCChatState, track: VCTrack) -> None:
        self._cancel_auto_task(state)
        duration = track.duration if track.duration and track.duration > 0 else self._default_track_seconds
        serial = state.play_serial

        async def _worker() -> None:
            try:
                await asyncio.sleep(int(duration) + 3)
                await self._advance_by_auto(chat_id, source="timer", serial_hint=serial)
            except asyncio.CancelledError:
                return
            except Exception:
                return

        state.auto_task = asyncio.create_task(_worker())

    def _resolve_cookie_file(self) -> Optional[str]:
        cookie_file = (os.getenv("YTDLP_COOKIE_FILE", "") or "").strip()
        if cookie_file and os.path.exists(cookie_file):
            return cookie_file
        cookie_text = (os.getenv("YTDLP_COOKIES", "") or "").strip()
        if not cookie_text:
            return None
        if self._cookie_file_path and os.path.exists(self._cookie_file_path):
            return self._cookie_file_path
        normalized = cookie_text.replace("\\n", "\n").strip()
        if not normalized:
            return None
        fd, path = tempfile.mkstemp(prefix="animx_yt_cookies_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(normalized)
        self._cookie_file_path = path
        return path

    def _cleanup_track_file(self, track: Optional[VCTrack]) -> None:
        if not track or not track.is_local:
            return
        try:
            if track.stream_url and os.path.exists(track.stream_url):
                os.remove(track.stream_url)
                parent = os.path.dirname(track.stream_url)
                if parent and os.path.basename(parent).startswith("animx_vc_dl_"):
                    try:
                        os.rmdir(parent)
                    except Exception:
                        pass
        except Exception:
            pass

    def _is_url(self, text: str) -> bool:
        return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))

    def _is_youtube_url(self, text: str) -> bool:
        s = text.lower()
        return "youtube.com/" in s or "youtu.be/" in s

    def _is_antibot_error(self, error_text: str) -> bool:
        t = error_text.lower()
        return "sign in to confirm" in t and "not a bot" in t

    def _pick_stream_url(self, info: dict[str, Any]) -> Optional[str]:
        direct = info.get("url")
        if isinstance(direct, str) and direct.startswith(("http://", "https://")):
            return direct

        requested_formats = info.get("requested_formats") or []
        for fmt in requested_formats:
            url = (fmt or {}).get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url

        formats = info.get("formats") or []
        audio_candidates: list[tuple[float, str]] = []
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            url = fmt.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            vcodec = (fmt.get("vcodec") or "").lower()
            acodec = (fmt.get("acodec") or "").lower()
            abr = float(fmt.get("abr") or 0)
            tbr = float(fmt.get("tbr") or 0)
            score = abr or tbr or 0
            if vcodec == "none" and acodec not in ("none", ""):
                audio_candidates.append((score, url))

        if not audio_candidates:
            return None
        audio_candidates.sort(key=lambda x: x[0], reverse=True)
        return audio_candidates[0][1]
    def _extract_search_candidates(
        self, query: str, base_opts: dict[str, Any], limit: int = 5
    ) -> list[tuple[str, str]]:
        search_opts = dict(base_opts)
        search_opts["extract_flat"] = True
        with self._yt_dlp.YoutubeDL(search_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        if not info:
            return []
        out: list[tuple[str, str]] = []
        for ent in info.get("entries") or []:
            if not isinstance(ent, dict):
                continue
            url = ent.get("webpage_url") or ent.get("url")
            if not url:
                continue
            title = ent.get("title") or "Unknown Title"
            out.append((title, url))
        return out

    def _fetch_youtube_oembed_title(self, url: str) -> Optional[str]:
        try:
            endpoint = "https://www.youtube.com/oembed"
            q = urllib.parse.urlencode({"url": url, "format": "json"})
            req = urllib.request.Request(f"{endpoint}?{q}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            title = (data.get("title") or "").strip()
            return title or None
        except Exception:
            return None

    def _resolve_track_via_download_sync(
        self,
        candidate_url: str,
        candidate_title: str,
        requested_by: str,
        base_opts: dict[str, Any],
    ) -> Optional[VCTrack]:
        dl_dir = tempfile.mkdtemp(prefix="animx_vc_dl_")
        profiles = [
            {"format": "bestaudio[acodec!=none]/bestaudio/best"},
            {"format": "bestaudio/best"},
            {"format": "ba/b"},
            {"format": "best"},
        ]
        for profile in profiles:
            dl_opts = dict(base_opts)
            dl_opts.update(
                {
                    "outtmpl": os.path.join(dl_dir, "%(id)s.%(ext)s"),
                    "restrictfilenames": True,
                    "skip_download": False,
                    "extract_flat": False,
                }
            )
            dl_opts.update(profile)
            try:
                with self._yt_dlp.YoutubeDL(dl_opts) as ydl:
                    info = ydl.extract_info(candidate_url, download=True)
                if not info:
                    continue
                downloaded = ydl.prepare_filename(info)
                if not downloaded or not os.path.exists(downloaded):
                    for name in os.listdir(dl_dir):
                        full = os.path.join(dl_dir, name)
                        if os.path.isfile(full):
                            downloaded = full
                            break
                if not downloaded or not os.path.exists(downloaded):
                    continue
                duration = info.get("duration")
                if not isinstance(duration, int):
                    try:
                        duration = int(duration) if duration else None
                    except Exception:
                        duration = None
                return VCTrack(
                    title=(info.get("title") or candidate_title or "Downloaded Track")[:120],
                    webpage_url=info.get("webpage_url") or candidate_url,
                    stream_url=downloaded,
                    requested_by=requested_by,
                    is_local=True,
                    duration=duration,
                    thumbnail=info.get("thumbnail"),
                )
            except Exception:
                continue
        try:
            if os.path.isdir(dl_dir) and not os.listdir(dl_dir):
                os.rmdir(dl_dir)
        except Exception:
            pass
        return None

    def _resolve_track_sync(self, query: str, requested_by: str) -> VCTrack:
        base_opts = {
            "quiet": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "skip_download": True,
            "extract_flat": False,
        }
        cookie_file = self._resolve_cookie_file()
        if cookie_file:
            base_opts["cookiefile"] = cookie_file

        query_is_url = self._is_url(query)
        target = query if query_is_url else f"ytsearch1:{query}"
        candidates: list[tuple[str, str]] = []
        last_error: Optional[Exception] = None

        try:
            with self._yt_dlp.YoutubeDL({**base_opts, "default_search": "ytsearch1", "format": "bestaudio/best"}) as ydl:
                info = ydl.extract_info(target, download=False)
            if info and "entries" in info:
                entries = info.get("entries") or []
                info = entries[0] if entries else None
            if info:
                title = info.get("title") or "Unknown Title"
                webpage_url = info.get("webpage_url") or info.get("original_url") or info.get("url")
                stream_url = self._pick_stream_url(info)
                duration = info.get("duration")
                if not isinstance(duration, int):
                    try:
                        duration = int(duration) if duration else None
                    except Exception:
                        duration = None
                if isinstance(stream_url, str) and stream_url.startswith(("http://", "https://")):
                    return VCTrack(
                        title=title[:120],
                        webpage_url=webpage_url or query,
                        stream_url=stream_url,
                        requested_by=requested_by,
                        is_local=False,
                        duration=duration,
                        thumbnail=info.get("thumbnail"),
                    )
                if webpage_url:
                    candidates.append((title, webpage_url))
        except Exception as e:
            last_error = e
            if self._is_antibot_error(str(e)):
                raise RuntimeError(
                    "YouTube blocked anonymous extraction for this track. "
                    "Set YTDLP_COOKIES or YTDLP_COOKIE_FILE in Railway vars and redeploy."
                ) from e

        if query_is_url and self._is_youtube_url(query):
            title = self._fetch_youtube_oembed_title(query) or "Unknown Title"
            candidates.append((title, query))
            search_query = title if title != "Unknown Title" else query
            try:
                candidates.extend(self._extract_search_candidates(search_query, base_opts, limit=8))
            except Exception:
                pass
        elif not query_is_url:
            try:
                candidates.extend(self._extract_search_candidates(query, base_opts, limit=8))
            except Exception:
                pass
        seen: set[str] = set()
        for candidate_title, candidate_url in candidates:
            if not candidate_url or candidate_url in seen:
                continue
            seen.add(candidate_url)
            try:
                with self._yt_dlp.YoutubeDL({**base_opts, "format": "bestaudio/best"}) as ydl:
                    info = ydl.extract_info(candidate_url, download=False)
                if info and "entries" in info:
                    entries = info.get("entries") or []
                    info = entries[0] if entries else None
                if not info:
                    continue
                stream_url = self._pick_stream_url(info)
                if not stream_url:
                    continue
                duration = info.get("duration")
                if not isinstance(duration, int):
                    try:
                        duration = int(duration) if duration else None
                    except Exception:
                        duration = None
                return VCTrack(
                    title=(info.get("title") or candidate_title or "Unknown Title")[:120],
                    webpage_url=info.get("webpage_url") or candidate_url,
                    stream_url=stream_url,
                    requested_by=requested_by,
                    is_local=False,
                    duration=duration,
                    thumbnail=info.get("thumbnail"),
                )
            except Exception:
                continue

        for candidate_title, candidate_url in candidates:
            dl_track = self._resolve_track_via_download_sync(candidate_url, candidate_title, requested_by, base_opts)
            if dl_track:
                return dl_track

        if last_error:
            raise RuntimeError(f"Could not resolve stream: {last_error}") from last_error
        raise RuntimeError("Could not resolve stream")

    async def resolve_track(self, query: str, requested_by: str) -> VCTrack:
        return await asyncio.to_thread(self._resolve_track_sync, query, requested_by)

    def _make_input_stream(self, url: str) -> Any:
        if self._audio_piped_cls is not None:
            return self._audio_piped_cls(url)
        return url

    async def _start_or_replace_stream(self, chat_id: int, state: VCChatState, track: VCTrack) -> None:
        async def _attempt_once() -> None:
            stream = self._make_input_stream(track.stream_url)
            if state.active_call:
                if self._supports_change_api:
                    await self._calls.change_stream(chat_id, stream)
                elif self._supports_play_api:
                    await self._calls.play(chat_id, stream)
                elif self._supports_join_api:
                    await self._calls.join_group_call(chat_id, stream)
                else:
                    raise RuntimeError("VC backend does not support stream replacement.")
                return

            if self._supports_join_api:
                await self._calls.join_group_call(chat_id, stream)
                return
            if self._supports_play_api:
                await self._calls.play(chat_id, stream)
                return
            raise RuntimeError("Could not start stream: unsupported VC backend API.")

        try:
            await _attempt_once()
            return
        except Exception as e:
            # One-time peer refresh retry for transient PeerIdInvalid.
            if "peer id invalid" not in str(e).lower():
                raise
            try:
                await self._assistant.get_chat(chat_id)
            except Exception:
                pass
            await _attempt_once()

    def _mark_track_started(self, chat_id: int, state: VCChatState, track: VCTrack) -> None:
        state.play_serial += 1
        state.last_auto_advanced_serial = 0
        state.now_playing = track
        state.active_call = True
        state.paused = False
        state.track_started_at = time.monotonic()
        state.paused_total_seconds = 0.0
        state.pause_started_at = None
        self._refresh_public_state(chat_id, state)
        self._schedule_auto_advance(chat_id, state, track)

    async def _play_track_locked(self, chat_id: int, state: VCChatState, track: VCTrack) -> None:
        await self._start_or_replace_stream(chat_id, state, track)
        self._mark_track_started(chat_id, state, track)

    async def enqueue_or_play(self, chat_id: int, query: str, requested_by: str) -> tuple[str, VCTrack]:
        await self.start()
        if not await self.verify_assistant_once(chat_id):
            raise RuntimeError("Assistant account is not in this chat. Add/unban assistant once and retry.")

        track = await self.resolve_track(query, requested_by)
        state = self._state(chat_id)
        async with state.lock:
            if state.now_playing:
                state.queue.append(track)
                self._refresh_public_state(chat_id, state)
                return "queued", track
            await self._play_track_locked(chat_id, state, track)
            return "playing", track

    async def enqueue_or_play_local(
        self,
        chat_id: int,
        file_path: str,
        title: str,
        requested_by: str,
        duration: Optional[int] = None,
    ) -> tuple[str, VCTrack]:
        await self.start()
        if not await self.verify_assistant_once(chat_id):
            raise RuntimeError("Assistant account is not in this chat. Add/unban assistant once and retry.")

        track = VCTrack(
            title=title[:120] if title else "Downloaded Track",
            webpage_url="local_file",
            stream_url=file_path,
            requested_by=requested_by,
            is_local=True,
            duration=duration,
            thumbnail=None,
        )
        state = self._state(chat_id)
        async with state.lock:
            if state.now_playing:
                state.queue.append(track)
                self._refresh_public_state(chat_id, state)
                return "queued", track
            await self._play_track_locked(chat_id, state, track)
            return "playing", track
    async def _advance_from_queue_locked(self, chat_id: int, state: VCChatState) -> Optional[VCTrack]:
        current = state.now_playing
        while state.queue:
            nxt = state.queue.pop(0)
            try:
                await self._play_track_locked(chat_id, state, nxt)
                if current:
                    state.history.append(current)
                    if len(state.history) > 12:
                        stale = state.history.pop(0)
                        self._cleanup_track_file(stale)
                self._refresh_public_state(chat_id, state)
                return nxt
            except Exception:
                self._cleanup_track_file(nxt)
                continue
        await self._stop_locked(chat_id, state)
        return None

    async def _advance_by_auto(self, chat_id: int, source: str, serial_hint: Optional[int] = None) -> Optional[VCTrack]:
        state = self._states.get(chat_id)
        if not state:
            return None
        async with state.lock:
            if not state.now_playing:
                return None
            if not state.queue:
                return None
            if serial_hint is not None and state.play_serial != serial_hint:
                return None
            if state.last_auto_advanced_serial == state.play_serial:
                return None
            state.last_auto_advanced_serial = state.play_serial
            logger.info(
                "VC auto-advance source=%s chat_id=%s queue_len=%s serial=%s",
                source,
                chat_id,
                len(state.queue),
                state.play_serial,
            )
            return await self._advance_from_queue_locked(chat_id, state)

    async def skip(self, chat_id: int) -> Optional[VCTrack]:
        state = self._state(chat_id)
        async with state.lock:
            if not state.queue:
                await self._stop_locked(chat_id, state)
                return None
            return await self._advance_from_queue_locked(chat_id, state)

    async def play_previous(self, chat_id: int) -> Optional[VCTrack]:
        state = self._state(chat_id)
        async with state.lock:
            if not state.history:
                return None
            prev = state.history.pop()
            if state.now_playing:
                state.queue.insert(0, state.now_playing)
            try:
                await self._play_track_locked(chat_id, state, prev)
                self._refresh_public_state(chat_id, state)
                return prev
            except Exception as e:
                state.history.append(prev)
                raise RuntimeError(f"Could not play previous track: {e}") from e

    async def _stop_locked(self, chat_id: int, state: VCChatState) -> None:
        self._cancel_auto_task(state)

        old_track = state.now_playing
        queued = list(state.queue)
        hist = list(state.history)

        if self._ready and state.active_call and self._supports_leave_api:
            try:
                await self._calls.leave_group_call(chat_id)
            except Exception:
                pass

        state.queue.clear()
        state.history.clear()
        state.now_playing = None
        state.active_call = False
        state.paused = False
        state.pause_started_at = None
        state.paused_total_seconds = 0.0
        state.track_started_at = 0.0

        self._refresh_public_state(chat_id, state)
        self._cleanup_track_file(old_track)
        for item in queued:
            self._cleanup_track_file(item)
        for item in hist:
            self._cleanup_track_file(item)

    async def stop_chat(self, chat_id: int) -> None:
        state = self._state(chat_id)
        async with state.lock:
            await self._stop_locked(chat_id, state)

    async def pause_chat(self, chat_id: int) -> None:
        await self.start()
        state = self._state(chat_id)
        async with state.lock:
            if not state.now_playing:
                raise RuntimeError("No active VC playback.")
            self._cancel_auto_task(state)
            if hasattr(self._calls, "pause_stream"):
                await self._calls.pause_stream(chat_id)
            elif hasattr(self._calls, "pause"):
                await self._calls.pause(chat_id)
            else:
                raise RuntimeError("Pause not supported by current VC backend.")
            if not state.paused:
                state.pause_started_at = time.monotonic()
            state.paused = True
            self._refresh_public_state(chat_id, state)

    async def resume_chat(self, chat_id: int) -> None:
        await self.start()
        state = self._state(chat_id)
        async with state.lock:
            if not state.now_playing:
                raise RuntimeError("No active VC playback.")
            if hasattr(self._calls, "resume_stream"):
                await self._calls.resume_stream(chat_id)
            elif hasattr(self._calls, "resume"):
                await self._calls.resume(chat_id)
            else:
                raise RuntimeError("Resume not supported by current VC backend.")
            if state.pause_started_at is not None:
                state.paused_total_seconds += max(0.0, time.monotonic() - state.pause_started_at)
            state.pause_started_at = None
            state.paused = False
            self._refresh_public_state(chat_id, state)
            if state.now_playing:
                self._schedule_auto_advance(chat_id, state, state.now_playing)

    def is_paused(self, chat_id: int) -> bool:
        state = self._states.get(chat_id)
        return bool(state and state.paused)

    def get_queue(self, chat_id: int) -> list[VCTrack]:
        state = self._states.get(chat_id)
        if not state:
            return []
        return list(state.queue)

    def get_now_playing(self, chat_id: int) -> Optional[VCTrack]:
        state = self._states.get(chat_id)
        return state.now_playing if state else None

    def get_elapsed_seconds(self, chat_id: int) -> int:
        state = self._states.get(chat_id)
        if not state or not state.now_playing or state.track_started_at <= 0:
            return 0
        now = time.monotonic()
        paused_total = state.paused_total_seconds
        if state.paused and state.pause_started_at is not None:
            now = state.pause_started_at
        elapsed = int(max(0.0, (now - state.track_started_at) - paused_total))
        return elapsed
