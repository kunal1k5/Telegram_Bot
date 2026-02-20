import asyncio
import importlib
import json
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
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
        self.history: dict[int, list[VCTrack]] = {}
        self.active_calls: set[int] = set()
        self._auto_tasks: dict[int, asyncio.Task] = {}
        self._play_tokens: dict[int, int] = {}
        self._paused_chats: set[int] = set()
        self._default_track_seconds = int(os.getenv("VC_DEFAULT_TRACK_SECONDS", "240"))
        self._stream_end_handler_registered = False

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
            self._register_stream_end_handler()
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
            for task in list(self._auto_tasks.values()):
                if task and not task.done():
                    task.cancel()
            self._auto_tasks.clear()
            self._play_tokens.clear()
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
                parent = os.path.dirname(track.stream_url)
                if parent and os.path.basename(parent).startswith("animx_vc_dl_"):
                    try:
                        os.rmdir(parent)
                    except Exception:
                        pass
        except Exception:
            pass

    def _cancel_auto_task(self, chat_id: int) -> None:
        task = self._auto_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    def _next_token(self, chat_id: int) -> int:
        token = self._play_tokens.get(chat_id, 0) + 1
        self._play_tokens[chat_id] = token
        return token

    @staticmethod
    def _extract_chat_id_from_update(update: Any) -> Optional[int]:
        """Best-effort extraction for different PyTgCalls update shapes."""
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
        """Advance queue immediately when backend sends stream-end event."""
        if not self._ready:
            return
        if chat_id not in self.now_playing:
            return
        if not self.queues.get(chat_id):
            return
        logger.info(
            "VC auto-advance source=event chat_id=%s queue_len=%s",
            chat_id,
            len(self.queues.get(chat_id, [])),
        )
        try:
            await self.skip(chat_id)
        except Exception:
            logger.warning("VC auto-advance source=event failed chat_id=%s", chat_id)
            return

    def _register_stream_end_handler(self) -> None:
        """Register stream-end callback if current PyTgCalls build exposes one."""
        if self._stream_end_handler_registered or not self._calls:
            return

        async def _handler(_: Any, update: Any) -> None:
            chat_id = self._extract_chat_id_from_update(update)
            if chat_id is None:
                return
            await self._on_stream_end(chat_id)

        hooks = ("on_stream_end", "on_stream_ended")
        for hook_name in hooks:
            hook = getattr(self._calls, hook_name, None)
            if not callable(hook):
                continue
            try:
                # Decorator style: @calls.on_stream_end()
                hook()(_handler)
                self._stream_end_handler_registered = True
                return
            except Exception:
                pass
            try:
                # Direct callback style: calls.on_stream_end(handler)
                hook(_handler)
                self._stream_end_handler_registered = True
                return
            except Exception:
                pass

    async def _auto_advance_worker(self, chat_id: int, token: int, delay: int) -> None:
        try:
            await asyncio.sleep(delay)
            if self._play_tokens.get(chat_id) != token:
                return
            if not self.queues.get(chat_id):
                return
            logger.info(
                "VC auto-advance source=timer chat_id=%s queue_len=%s",
                chat_id,
                len(self.queues.get(chat_id, [])),
            )
            try:
                await self.skip(chat_id)
            except Exception:
                # Keep queue moving even if one transition fails.
                if self._play_tokens.get(chat_id) == token and self.queues.get(chat_id):
                    logger.warning(
                        "VC auto-advance source=timer retrying chat_id=%s queue_len=%s",
                        chat_id,
                        len(self.queues.get(chat_id, [])),
                    )
                    await asyncio.sleep(1)
                    try:
                        await self.skip(chat_id)
                    except Exception:
                        logger.warning("VC auto-advance source=timer failed chat_id=%s", chat_id)
                        return
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def _schedule_auto_advance(self, chat_id: int, track: VCTrack, token: int) -> None:
        self._cancel_auto_task(chat_id)
        duration = track.duration if track.duration and track.duration > 0 else self._default_track_seconds
        self._auto_tasks[chat_id] = asyncio.create_task(
            self._auto_advance_worker(chat_id, token, int(duration) + 3)
        )

    def _ensure_auto_advance(self, chat_id: int) -> None:
        """Ensure auto-advance task exists when queue is waiting."""
        if chat_id in self._auto_tasks and not self._auto_tasks[chat_id].done():
            return
        now_track = self.now_playing.get(chat_id)
        if not now_track:
            return
        if not self.queues.get(chat_id):
            return
        token = self._play_tokens.get(chat_id) or self._next_token(chat_id)
        self._schedule_auto_advance(chat_id, now_track, token)

    def _is_playback_healthy(self, chat_id: int) -> bool:
        """Heuristic: playback state is healthy only if call is active or explicitly paused."""
        if chat_id in self.active_calls:
            return True
        if chat_id in self._paused_chats:
            return True
        return False

    def _reset_chat_playback_state(self, chat_id: int) -> None:
        """Drop stale playback markers for a chat."""
        self._cancel_auto_task(chat_id)
        self._play_tokens.pop(chat_id, None)
        self._paused_chats.discard(chat_id)
        self.active_calls.discard(chat_id)
        stale_track = self.now_playing.pop(chat_id, None)
        self._cleanup_track_file(stale_track)
        stale_history = self.history.pop(chat_id, [])
        for item in stale_history:
            self._cleanup_track_file(item)

    def _push_history(self, chat_id: int, track: Optional[VCTrack], max_items: int = 12) -> None:
        if not track:
            return
        stack = self.history.setdefault(chat_id, [])
        stack.append(track)
        if len(stack) <= max_items:
            return
        stale = stack.pop(0)
        self._cleanup_track_file(stale)

    def _is_url(self, text: str) -> bool:
        return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))

    def _is_youtube_url(self, text: str) -> bool:
        s = text.lower()
        return "youtube.com/" in s or "youtu.be/" in s

    def _extract_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract canonical YouTube video id from common URL shapes."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            path = parsed.path or ""
            if "youtu.be" in host:
                vid = path.strip("/").split("/")[0]
                return vid or None
            if "youtube.com" in host:
                if path == "/watch":
                    q = urllib.parse.parse_qs(parsed.query)
                    vid = (q.get("v") or [None])[0]
                    return vid
                # /shorts/<id>, /embed/<id>, /live/<id>
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                    return parts[1]
        except Exception:
            return None
        return None

    def _fetch_youtube_oembed_title(self, url: str) -> Optional[str]:
        """Fetch video title via YouTube oEmbed (works even when formats are restricted)."""
        try:
            endpoint = "https://www.youtube.com/oembed"
            q = urllib.parse.urlencode({"url": url, "format": "json"})
            req = urllib.request.Request(
                f"{endpoint}?{q}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            title = (data.get("title") or "").strip()
            return title or None
        except Exception:
            return None

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

    def _resolve_track_via_download_sync(
        self,
        candidate_url: str,
        candidate_title: str,
        requested_by: str,
        base_opts: dict[str, Any],
    ) -> Optional[VCTrack]:
        """Fallback: download audio locally and play as local VC track."""
        dl_dir = tempfile.mkdtemp(prefix="animx_vc_dl_")
        profile_opts: list[dict[str, Any]] = [
            {
                "format": "bestaudio[acodec!=none]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android", "web", "ios", "mweb"]}},
            },
            {
                "format": "bestaudio[acodec!=none]/bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["tv", "tv_embedded", "web_creator"]}},
            },
            {
                "format": "bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["android", "web", "ios", "mweb"]}},
            },
            {
                "format": "bestaudio/best",
                "extractor_args": {"youtube": {"player_client": ["tv", "tv_embedded", "web_creator"]}},
            },
            {"format": "ba/b"},
            {"format": "best"},
            {},  # no format preference - let yt-dlp choose
        ]
        try:
            for profile in profile_opts:
                dl_opts = dict(base_opts)
                dl_opts.update(
                    {
                        "quiet": True,
                        "noplaylist": True,
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

                    final_duration = info.get("duration")
                    if not isinstance(final_duration, int):
                        try:
                            final_duration = int(final_duration) if final_duration else None
                        except Exception:
                            final_duration = None
                    final_thumbnail = info.get("thumbnail")
                    return VCTrack(
                        title=(info.get("title") or candidate_title or "Downloaded Track")[:120],
                        webpage_url=info.get("webpage_url") or candidate_url,
                        stream_url=downloaded,
                        requested_by=requested_by,
                        is_local=True,
                        duration=final_duration,
                        thumbnail=final_thumbnail,
                    )
                except Exception:
                    continue
        except Exception:
            pass
        # Cleanup dir if nothing worked.
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
            "default_search": "ytsearch1",
            "extract_flat": False,
            "skip_download": True,
        }
        cookie_file = self._resolve_cookie_file()
        if cookie_file:
            base_opts["cookiefile"] = cookie_file

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
                if title == "Unknown Title":
                    oembed_title = self._fetch_youtube_oembed_title(query)
                    if oembed_title:
                        title = oembed_title
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

        # Always-download mode for VC reliability:
        # resolve/search first, then download local audio and play that file in VC.
        for candidate_title, candidate_url in candidates:
            dl_track = self._resolve_track_via_download_sync(
                candidate_url, candidate_title, requested_by, base_opts
            )
            if dl_track:
                return dl_track
            last_error = RuntimeError(f"Download fallback failed for candidate: {candidate_url}")

        # Extra recovery path: if original URL is restricted, try similar search results and download fallback.
        if query_is_url and self._is_youtube_url(webpage_url):
            try:
                fallback_query = title if title and title != "Unknown Title" else query
                if fallback_query.startswith(("http://", "https://")):
                    oembed_title = self._fetch_youtube_oembed_title(webpage_url)
                    if oembed_title:
                        fallback_query = oembed_title
                video_id = self._extract_youtube_video_id(webpage_url)
                search_queries: list[str] = []
                seen: set[str] = set()
                for q in (
                    fallback_query,
                    f"{fallback_query} audio",
                    f"{fallback_query} official audio",
                    video_id or "",
                ):
                    q = (q or "").strip()
                    if q and q not in seen:
                        seen.add(q)
                        search_queries.append(q)

                for q in search_queries:
                    for alt_title, alt_url in self._extract_search_candidates(q, base_opts, limit=10):
                        if alt_url == webpage_url:
                            continue
                        dl_track = self._resolve_track_via_download_sync(
                            alt_url, alt_title, requested_by, base_opts
                        )
                        if dl_track:
                            return dl_track
            except Exception:
                pass

        if last_error:
            raise RuntimeError(f"Could not resolve stream: {last_error}") from last_error
        raise RuntimeError("Could not resolve stream")

    async def resolve_track(self, query: str, requested_by: str) -> VCTrack:
        return await asyncio.to_thread(self._resolve_track_sync, query, requested_by)

    async def _play_track(self, chat_id: int, track: VCTrack) -> None:
        await self.start()
        token = self._next_token(chat_id)

        # New py-tgcalls API (2.x): play(chat_id, stream_url) handles start + replace.
        if self._supports_play_api:
            try:
                await self._calls.play(chat_id, track.stream_url)
            except Exception as e:
                raise RuntimeError(f"Could not start stream: {e}") from e
            self.active_calls.add(chat_id)
            self.now_playing[chat_id] = track
            self._schedule_auto_advance(chat_id, track, token)
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
                self._schedule_auto_advance(chat_id, track, token)
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
        self._paused_chats.discard(chat_id)
        self._schedule_auto_advance(chat_id, track, token)

    async def enqueue_or_play(self, chat_id: int, query: str, requested_by: str) -> tuple[str, VCTrack]:
        track = await self.resolve_track(query, requested_by)
        if chat_id in self.now_playing and not self._is_playback_healthy(chat_id):
            logger.warning("VC stale playback state detected chat_id=%s; resetting state", chat_id)
            self._reset_chat_playback_state(chat_id)

        if chat_id in self.now_playing:
            self.queues.setdefault(chat_id, []).append(track)
            self._ensure_auto_advance(chat_id)
            return "queued", track

        await self._play_track(chat_id, track)
        return "playing", track

    async def enqueue_or_play_local(
        self,
        chat_id: int,
        file_path: str,
        title: str,
        requested_by: str,
        duration: Optional[int] = None,
    ) -> tuple[str, VCTrack]:
        track = VCTrack(
            title=title[:120] if title else "Downloaded Track",
            webpage_url="local_file",
            stream_url=file_path,
            requested_by=requested_by,
            is_local=True,
            duration=duration,
            thumbnail=None,
        )
        if chat_id in self.now_playing and not self._is_playback_healthy(chat_id):
            logger.warning("VC stale playback state detected chat_id=%s (local); resetting state", chat_id)
            self._reset_chat_playback_state(chat_id)

        if chat_id in self.now_playing:
            self.queues.setdefault(chat_id, []).append(track)
            self._ensure_auto_advance(chat_id)
            return "queued", track
        await self._play_track(chat_id, track)
        return "playing", track

    async def skip(self, chat_id: int) -> Optional[VCTrack]:
        old_track = self.now_playing.get(chat_id)
        self._cancel_auto_task(chat_id)
        self._paused_chats.discard(chat_id)
        queue = self.queues.get(chat_id, [])
        if not queue:
            await self.stop_chat(chat_id)
            return None
        # Try queued tracks until one starts successfully.
        while queue:
            next_track = queue.pop(0)
            try:
                await self._play_track(chat_id, next_track)
                self._push_history(chat_id, old_track)
                return next_track
            except Exception:
                # Broken entry in queue: discard and continue with next.
                self._cleanup_track_file(next_track)
                continue

        # Queue exhausted due failures.
        await self.stop_chat(chat_id)
        return None

    async def play_previous(self, chat_id: int) -> Optional[VCTrack]:
        stack = self.history.get(chat_id, [])
        if not stack:
            return None

        previous_track = stack.pop()
        current_track = self.now_playing.get(chat_id)
        self._cancel_auto_task(chat_id)
        self._paused_chats.discard(chat_id)
        if current_track:
            self.queues.setdefault(chat_id, []).insert(0, current_track)

        try:
            await self._play_track(chat_id, previous_track)
            return previous_track
        except Exception as e:
            if current_track and self.queues.get(chat_id):
                q = self.queues.get(chat_id) or []
                if q and q[0] is current_track:
                    q.pop(0)
            stack.append(previous_track)
            raise RuntimeError(f"Could not play previous track: {e}") from e

    async def stop_chat(self, chat_id: int) -> None:
        old_track = self.now_playing.get(chat_id)
        queued_tracks = self.queues.get(chat_id, [])
        history_tracks = self.history.get(chat_id, [])
        self._cancel_auto_task(chat_id)
        self._play_tokens.pop(chat_id, None)
        self._paused_chats.discard(chat_id)
        if not self._ready:
            self._cleanup_track_file(old_track)
            for item in queued_tracks:
                self._cleanup_track_file(item)
            for item in history_tracks:
                self._cleanup_track_file(item)
            self.queues.pop(chat_id, None)
            self.history.pop(chat_id, None)
            self.now_playing.pop(chat_id, None)
            self.active_calls.discard(chat_id)
            return
        try:
            await self._calls.leave_group_call(chat_id)
        except Exception:
            pass
        self.queues.pop(chat_id, None)
        self.history.pop(chat_id, None)
        self.now_playing.pop(chat_id, None)
        self.active_calls.discard(chat_id)
        self._cleanup_track_file(old_track)
        for item in queued_tracks:
            self._cleanup_track_file(item)
        for item in history_tracks:
            self._cleanup_track_file(item)

    async def pause_chat(self, chat_id: int) -> None:
        if not self._ready:
            raise RuntimeError("VC is not running")
        self._cancel_auto_task(chat_id)
        if hasattr(self._calls, "pause_stream"):
            await self._calls.pause_stream(chat_id)
        elif hasattr(self._calls, "pause"):
            await self._calls.pause(chat_id)
        else:
            raise RuntimeError("Pause not supported by current VC backend")
        self._paused_chats.add(chat_id)

    async def resume_chat(self, chat_id: int) -> None:
        if not self._ready:
            raise RuntimeError("VC is not running")
        if hasattr(self._calls, "resume_stream"):
            await self._calls.resume_stream(chat_id)
        elif hasattr(self._calls, "resume"):
            await self._calls.resume(chat_id)
        else:
            raise RuntimeError("Resume not supported by current VC backend")
        self._paused_chats.discard(chat_id)
        now_track = self.now_playing.get(chat_id)
        token = self._play_tokens.get(chat_id, 0)
        if now_track and token:
            self._schedule_auto_advance(chat_id, now_track, token)

    def is_paused(self, chat_id: int) -> bool:
        return chat_id in self._paused_chats

    def get_queue(self, chat_id: int) -> list[VCTrack]:
        return list(self.queues.get(chat_id, []))

    def get_now_playing(self, chat_id: int) -> Optional[VCTrack]:
        return self.now_playing.get(chat_id)
