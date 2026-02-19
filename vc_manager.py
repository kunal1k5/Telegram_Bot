import asyncio
import importlib
import os
import re
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional


@dataclass
class VCTrack:
    title: str
    webpage_url: str
    stream_url: str
    requested_by: str
    is_local: bool = False


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
        self._supports_play_api = False
        self._supports_join_api = False
        self._yt_dlp: Any = None
        self._cookie_file_path: Optional[str] = None

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
                import yt_dlp
            except Exception as e:
                pkg_bits: list[str] = []
                for pkg_name in ("py-tgcalls", "pytgcalls"):
                    try:
                        pkg_bits.append(f"{pkg_name}={version(pkg_name)}")
                    except PackageNotFoundError:
                        pkg_bits.append(f"{pkg_name}=not-installed")
                raise RuntimeError(
                    "VC dependencies missing. Install pyrogram, tgcrypto, py-tgcalls, yt-dlp. "
                    f"Import error: {type(e).__name__}: {e}. "
                    f"Detected packages: {', '.join(pkg_bits)}"
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
                        "(from Client.export_session_string), not Telethon StringSession. "
                        "Also ensure no quotes/spaces/newlines were added in env value."
                    ) from e
                raise RuntimeError(f"Assistant login failed: {e}") from e

            self._calls = PyTgCalls(self._assistant)
            await self._calls.start()

            self._supports_play_api = hasattr(self._calls, "play")
            self._supports_join_api = hasattr(self._calls, "join_group_call")

            # Old API compatibility: join_group_call(chat_id, AudioPiped(url))
            if self._supports_join_api and not self._supports_play_api:
                audio_piped_cls = None
                audio_import_errors: list[str] = []
                for module_name, class_name in [
                    ("pytgcalls.types.input_stream", "AudioPiped"),
                    ("pytgcalls.types.input_stream.audio_piped", "AudioPiped"),
                    ("pytgcalls.types", "AudioPiped"),
                ]:
                    try:
                        module = importlib.import_module(module_name)
                        audio_piped_cls = getattr(module, class_name)
                        break
                    except Exception as ie:
                        audio_import_errors.append(f"{module_name}.{class_name}: {ie}")

                if audio_piped_cls is None:
                    raise RuntimeError(
                        "PyTgCalls old API detected but AudioPiped import failed. "
                        + " | ".join(audio_import_errors)
                    )
                self._audio_piped_cls = audio_piped_cls

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
            if self._cookie_file_path and os.path.exists(self._cookie_file_path):
                try:
                    os.remove(self._cookie_file_path)
                except Exception:
                    pass
                self._cookie_file_path = None
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

    def _cleanup_track_file(self, track: Optional[VCTrack]) -> None:
        """Delete local cached files after track is done."""
        if not track or not track.is_local:
            return
        try:
            if track.stream_url and os.path.exists(track.stream_url):
                os.remove(track.stream_url)
        except Exception:
            pass

    def _is_url(self, text: str) -> bool:
        return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))

    def _is_youtube_url(self, text: str) -> bool:
        s = text.lower()
        return "youtube.com/" in s or "youtu.be/" in s

    def _is_antibot_error(self, error_text: str) -> bool:
        return (
            "sign in to confirm" in error_text.lower()
            and "not a bot" in error_text.lower()
        )

    def _resolve_cookie_file(self) -> Optional[str]:
        """Resolve yt-dlp cookies from env for Railway/server usage."""
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

    def _pick_stream_url(self, info: dict[str, Any]) -> Optional[str]:
        """Pick a playable stream URL from yt-dlp info with fallbacks."""
        direct = info.get("url")
        if isinstance(direct, str) and direct.startswith(("http://", "https://")):
            return direct

        requested_formats = info.get("requested_formats") or []
        for fmt in requested_formats:
            url = (fmt or {}).get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url

        formats = info.get("formats") or []
        audio_candidates = []
        other_candidates = []
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            url = fmt.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            vcodec = (fmt.get("vcodec") or "").lower()
            acodec = (fmt.get("acodec") or "").lower()
            abr = fmt.get("abr") or 0
            tbr = fmt.get("tbr") or 0
            score = (abr or tbr or 0)
            if vcodec == "none" and acodec not in ("none", ""):
                audio_candidates.append((score, url))
            else:
                other_candidates.append((score, url))

        if audio_candidates:
            audio_candidates.sort(key=lambda x: x[0], reverse=True)
            return audio_candidates[0][1]
        if other_candidates:
            other_candidates.sort(key=lambda x: x[0], reverse=True)
            return other_candidates[0][1]
        return None

    def _extract_search_candidates(
        self, query: str, base_opts: dict[str, Any], limit: int = 5
    ) -> list[tuple[str, str]]:
        """Return (title, url) candidates using flat search mode."""
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

    def _resolve_track_sync(self, query: str, requested_by: str) -> VCTrack:
        base_opts = {
            "quiet": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "default_search": "ytsearch1",
            "extract_flat": False,
            "skip_download": True,
        }
        cookie_file = self._resolve_cookie_file()
        if cookie_file:
            base_opts["cookiefile"] = cookie_file

        format_profiles = [
            {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            },
            {
                "format": "bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            },
            {
                "format": "bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["ios", "mweb", "tv"]}},
            },
            {"format": "best"},
            {},
        ]

        query_is_url = self._is_url(query)
        search_target = query if query_is_url else f"ytsearch1:{query}"
        last_error: Optional[Exception] = None

        if query_is_url:
            webpage_url = query
            title = "Unknown Title"
            if self._is_youtube_url(query):
                # Try to fetch metadata in flat mode; if it fails, we still continue with raw URL.
                try:
                    search_opts = dict(base_opts)
                    search_opts["extract_flat"] = True
                    with self._yt_dlp.YoutubeDL(search_opts) as ydl:
                        info = ydl.extract_info(query, download=False)
                    if isinstance(info, dict):
                        title = info.get("title") or title
                        webpage_url = info.get("webpage_url") or webpage_url
                except Exception:
                    pass
        else:
            search_opts = dict(base_opts)
            search_opts["extract_flat"] = True
            try:
                with self._yt_dlp.YoutubeDL(search_opts) as ydl:
                    info = ydl.extract_info(search_target, download=False)
            except Exception as e:
                err = str(e)
                if self._is_antibot_error(err):
                    raise RuntimeError(
                        "YouTube blocked anonymous extraction for this track. "
                        "Set YTDLP_COOKIES (Netscape cookies text) or YTDLP_COOKIE_FILE in Railway vars, then redeploy."
                    ) from e
                raise RuntimeError(f"Could not resolve track: {e}") from e

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

        candidates: list[tuple[str, str]] = [(title, webpage_url)]
        if query_is_url and self._is_youtube_url(webpage_url):
            try:
                alt_query = title if title and title != "Unknown Title" else query
                for alt_title, alt_url in self._extract_search_candidates(alt_query, base_opts, limit=5):
                    if alt_url != webpage_url:
                        candidates.append((alt_title, alt_url))
            except Exception:
                pass

        for candidate_title, candidate_url in candidates:
            for profile in format_profiles:
                detail_opts = dict(base_opts)
                detail_opts.update(profile)
                try:
                    with self._yt_dlp.YoutubeDL(detail_opts) as ydl:
                        detailed = ydl.extract_info(candidate_url, download=False)
                    if not detailed:
                        raise RuntimeError("Could not resolve stream info")
                    stream_url = self._pick_stream_url(detailed)
                    if not stream_url:
                        raise RuntimeError("Could not resolve audio stream url (no playable format)")

                    final_title = detailed.get("title") or candidate_title or "Unknown Title"
                    final_url = detailed.get("webpage_url") or candidate_url
                    return VCTrack(
                        title=final_title[:120],
                        webpage_url=final_url,
                        stream_url=stream_url,
                        requested_by=requested_by,
                        is_local=False,
                    )
                except Exception as e:
                    err = str(e)
                    if self._is_antibot_error(err):
                        raise RuntimeError(
                            "YouTube blocked anonymous extraction for this track. "
                            "Set YTDLP_COOKIES (Netscape cookies text) or YTDLP_COOKIE_FILE in Railway vars, then redeploy."
                        ) from e
                    last_error = e
                    continue

        if last_error:
            raise RuntimeError(f"Could not resolve stream: {last_error}") from last_error
        raise RuntimeError("Could not resolve stream")

    async def resolve_track(self, query: str, requested_by: str) -> VCTrack:
        return await asyncio.to_thread(self._resolve_track_sync, query, requested_by)

    async def _play_track(self, chat_id: int, track: VCTrack) -> None:
        await self.start()

        # New py-tgcalls API (2.x): play(chat_id, stream_url) handles start + replace.
        if self._supports_play_api:
            try:
                await self._calls.play(chat_id, track.stream_url)
            except Exception as e:
                raise RuntimeError(f"Could not start stream: {e}") from e
            self.active_calls.add(chat_id)
            self.now_playing[chat_id] = track
            return

        # Old API fallback.
        if not self._supports_join_api or self._audio_piped_cls is None:
            raise RuntimeError("Unsupported PyTgCalls API detected in this environment.")

        stream = self._audio_piped_cls(track.stream_url)
        if chat_id not in self.active_calls:
            try:
                await self._calls.join_group_call(chat_id, stream)
                self.active_calls.add(chat_id)
                self.now_playing[chat_id] = track
                return
            except Exception:
                pass

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

    async def enqueue_or_play_local(
        self, chat_id: int, file_path: str, title: str, requested_by: str
    ) -> tuple[str, VCTrack]:
        track = VCTrack(
            title=title[:120] if title else "Downloaded Track",
            webpage_url="local_file",
            stream_url=file_path,
            requested_by=requested_by,
            is_local=True,
        )
        if chat_id in self.now_playing:
            self.queues.setdefault(chat_id, []).append(track)
            return "queued", track
        await self._play_track(chat_id, track)
        return "playing", track

    async def skip(self, chat_id: int) -> Optional[VCTrack]:
        old_track = self.now_playing.get(chat_id)
        queue = self.queues.get(chat_id, [])
        if not queue:
            self.now_playing.pop(chat_id, None)
            self._cleanup_track_file(old_track)
            return None

        next_track = queue.pop(0)
        await self._play_track(chat_id, next_track)
        self._cleanup_track_file(old_track)
        return next_track

    async def stop_chat(self, chat_id: int) -> None:
        old_track = self.now_playing.get(chat_id)
        queued_tracks = self.queues.get(chat_id, [])
        if not self._ready:
            self._cleanup_track_file(old_track)
            for item in queued_tracks:
                self._cleanup_track_file(item)
            return
        try:
            await self._calls.leave_group_call(chat_id)
        except Exception:
            pass
        self.queues.pop(chat_id, None)
        self.now_playing.pop(chat_id, None)
        self.active_calls.discard(chat_id)
        self._cleanup_track_file(old_track)
        for item in queued_tracks:
            self._cleanup_track_file(item)

    def get_queue(self, chat_id: int) -> list[VCTrack]:
        return list(self.queues.get(chat_id, []))

    def get_now_playing(self, chat_id: int) -> Optional[VCTrack]:
        return self.now_playing.get(chat_id)
