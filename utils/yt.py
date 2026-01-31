import asyncio
import os
import re
import tempfile
from typing import Tuple

import yt_dlp

from config import DOWNLOAD_DIR

YTDL_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": True,
    "quiet": True,
    "no_warnings": True,
    "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
}


def _is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text))


def _download_sync(query: str) -> Tuple[str, str]:
    """Blocking yt-dlp download implementation.

    Returns (file_path, title).
    Raises RuntimeError on failure.
    """

    ydl_opts = YTDL_OPTS.copy()

    if not _is_url(query):
        query = f"ytsearch1:{query}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=True)
        if info is None:
            raise RuntimeError("Could not download audio.")

        # When using ytsearch, results are inside "entries"
        if "entries" in info:
            info = info["entries"][0]

        if info is None:
            raise RuntimeError("No audio results found.")

        file_path = ydl.prepare_filename(info)
        title = info.get("title") or "Unknown title"
        return file_path, title


async def download_audio(query: str) -> Tuple[str, str]:
    """Download audio for the given query or URL using yt-dlp.

    This runs the blocking yt-dlp call in a thread executor so it
    does not block the asyncio event loop.
    """

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _download_sync, query)
