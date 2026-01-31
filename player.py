import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

from pyrogram import Client
from pytgcalls import GroupCallFactory
from pytgcalls.implementation.group_call_file import GroupCallFile

from config import DOWNLOAD_DIR

logger = logging.getLogger(__name__)


@dataclass
class Track:
    chat_id: int
    file_path: str
    title: str
    requested_by: Optional[str] = None


class MusicPlayer:
    """Manages voice chat playback for multiple groups using PyTgCalls 3.x."""

    def __init__(self, app: Client) -> None:
        self.app = app
        self.factory = GroupCallFactory(self.app)

        # Per-chat queues, state, and group call objects
        self.queues: Dict[int, asyncio.Queue[Track]] = {}
        self.current: Dict[int, Optional[Track]] = {}
        self.locks: Dict[int, asyncio.Lock] = {}
        self.calls: Dict[int, GroupCallFile] = {}

    async def start(self) -> None:
        logger.info("Music player ready.")

    async def shutdown(self) -> None:
        """Best-effort cleanup of all active calls and temp files."""

        for chat_id in list(self.queues.keys()):
            try:
                await self.stop(chat_id)
            except Exception:
                continue

    def _get_queue(self, chat_id: int) -> asyncio.Queue[Track]:
        if chat_id not in self.queues:
            self.queues[chat_id] = asyncio.Queue()
        return self.queues[chat_id]

    def _get_call(self, chat_id: int) -> GroupCallFile:
        if chat_id not in self.calls:
            group_call = self.factory.get_file_group_call(play_on_repeat=False)

            @group_call.on_playout_ended
            async def _on_playout_ended(_, __):  # type: ignore[no-redef]
                logger.info("Playout ended in chat %s", chat_id)
                await self._play_next(chat_id)

            self.calls[chat_id] = group_call

        return self.calls[chat_id]

    def _get_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self.locks:
            self.locks[chat_id] = asyncio.Lock()
        return self.locks[chat_id]

    async def add_to_queue(self, track: Track) -> int:
        """Add a track to the queue for a chat.

        Returns the position in the queue (1 = now playing).
        """

        queue = self._get_queue(track.chat_id)
        await queue.put(track)
        size = queue.qsize()
        logger.info(
            "Added track to queue chat=%s title=%s position=%s",
            track.chat_id,
            track.title,
            size,
        )

        if not self.current.get(track.chat_id):
            asyncio.create_task(self._play_next(track.chat_id))
            return 1

        return size + 1

    async def _play_next(self, chat_id: int) -> None:
        lock = self._get_lock(chat_id)
        async with lock:
            queue = self._get_queue(chat_id)

            previous = self.current.get(chat_id)
            if previous and os.path.exists(previous.file_path):
                try:
                    os.remove(previous.file_path)
                    logger.info("Removed old file %s", previous.file_path)
                except OSError as e:
                    logger.warning("Failed to remove file %s: %s", previous.file_path, e)

            if queue.empty():
                self.current[chat_id] = None
                try:
                    group_call = self._get_call(chat_id)
                    await group_call.stop()
                    logger.info("Left voice chat for chat %s (queue empty)", chat_id)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Error leaving voice chat for chat %s: %s", chat_id, e)
                return

            track = await queue.get()
            self.current[chat_id] = track

            logger.info("Starting playback in chat=%s title=%s", chat_id, track.title)

            group_call = self._get_call(chat_id)
            group_call.input_filename = track.file_path

            try:
                if group_call.is_connected:
                    group_call.restart_playout()
                else:
                    await group_call.start(chat_id)
            except Exception as err:  # noqa: BLE001
                logger.error("Failed to start stream in chat %s: %s", chat_id, err)
                await self._play_next(chat_id)

    async def pause(self, chat_id: int) -> bool:
        try:
            group_call = self._get_call(chat_id)
            group_call.pause_playout()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Pause failed in chat %s: %s", chat_id, e)
            return False

    async def resume(self, chat_id: int) -> bool:
        try:
            group_call = self._get_call(chat_id)
            group_call.resume_playout()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Resume failed in chat %s: %s", chat_id, e)
            return False

    async def skip(self, chat_id: int) -> bool:
        queue = self._get_queue(chat_id)
        if queue.empty() and not self.current.get(chat_id):
            return False

        try:
            group_call = self._get_call(chat_id)
            await group_call.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("leave_group_call during skip failed for chat %s: %s", chat_id, e)

        asyncio.create_task(self._play_next(chat_id))
        return True

    async def stop(self, chat_id: int) -> None:
        queue = self._get_queue(chat_id)

        while not queue.empty():
            track = await queue.get()
            if os.path.exists(track.file_path):
                try:
                    os.remove(track.file_path)
                except OSError:
                    pass

        self.queues[chat_id] = asyncio.Queue()

        current = self.current.get(chat_id)
        if current and os.path.exists(current.file_path):
            try:
                os.remove(current.file_path)
            except OSError:
                pass

        self.current[chat_id] = None

        try:
            group_call = self._get_call(chat_id)
            await group_call.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("Error leaving group call for chat %s: %s", chat_id, e)

        for root, _, files in os.walk(DOWNLOAD_DIR):
            for name in files:
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) == 0:
                        os.remove(path)
                except OSError:
                    continue
