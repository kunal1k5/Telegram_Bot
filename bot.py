import asyncio
import logging
import os
from typing import Final

import telegram
from telegram import Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from pyrogram import Client

from config import API_HASH, API_ID, BOT_TOKEN
from player import MusicPlayer, Track
from utils.yt import download_audio

# On Windows, make sure to use the selector event loop policy for asyncio
if os.name == "nt":  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ANIMX_CLAN")

BOT_NAME: Final[str] = "ANIMX CLAN"
BOT_USERNAME: Final[str] = "@AnimxClanBot"

START_TEXT: Final[str] = (
    "🎶 ANIMX CLAN Music Bot 🎶\n"
    "Add me to a group, start a voice chat, and use:\n"
    "/play song_name_or_link"
)


# ========================= COMMAND HANDLERS ========================= #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(
        "/start received chat_id=%s type=%s user=%s",
        update.effective_chat.id if update.effective_chat else None,
        update.effective_chat.type if update.effective_chat else None,
        update.effective_user.id if update.effective_user else None,
    )

    await update.effective_message.reply_text(START_TEXT)


async def _ensure_group(update: Update) -> bool:
    chat = update.effective_chat
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await update.effective_message.reply_text(
            "This command can only be used in groups.",
        )
        return False
    return True


async def play(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    message = update.effective_message
    chat = update.effective_chat
    assert chat is not None

    if not context.args:
        await message.reply_text(
            "Usage: /play <song name or YouTube URL>",
            parse_mode=ParseMode.HTML,
        )
        return

    query = " ".join(context.args).strip()

    # Simple anti-spam: limit length of query
    if len(query) > 128:
        await message.reply_text("Query too long.")
        return

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    waiting = await message.reply_text("🔎 Searching and downloading audio...")

    try:
        file_path, title = await download_audio(query)
    except Exception as e:  # noqa: BLE001
        logger.error("Download failed: %s", e)
        await waiting.edit_text("Failed to download audio. Try a different query.")
        return

    requested_by = update.effective_user.mention_html() if update.effective_user else "Unknown"
    track = Track(chat_id=chat.id, file_path=file_path, title=title, requested_by=requested_by)

    position = await player.add_to_queue(track)

    if position == 1 and not player.current.get(chat.id):
        text = f"▶️ Now playing: <b>{title}</b>\nRequested by: {requested_by}"
    else:
        text = (
            f"✅ Added to queue: <b>{title}</b> (position {position})\n"
            f"Requested by: {requested_by}"
        )

    try:
        await waiting.edit_text(text, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest:
        await message.reply_text(text, parse_mode=ParseMode.HTML)


async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.pause(chat.id):
        await update.effective_message.reply_text("⏸ Paused.")
    else:
        await update.effective_message.reply_text("Nothing is playing or pause failed.")


async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.resume(chat.id):
        await update.effective_message.reply_text("▶️ Resumed.")
    else:
        await update.effective_message.reply_text("Nothing is paused or resume failed.")


async def skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    if await player.skip(chat.id):
        await update.effective_message.reply_text("⏭ Skipped.")
    else:
        await update.effective_message.reply_text("Nothing to skip.")


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_group(update):
        return

    chat = update.effective_chat
    assert chat is not None

    app: Application = context.application
    player: MusicPlayer = app.bot_data["player"]

    await player.stop(chat.id)
    await update.effective_message.reply_text("🛑 Stopped and left voice chat.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


# ========================= APP LIFECYCLE ========================= #


async def on_startup(app: Application) -> None:
    logger.info("Deleting webhook and starting clients...")

    # Ensure no webhook conflicts exist
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logger.exception("Failed to delete webhook")

    # Create a Pyrogram client using the same bot token
    pyro_client = Client(
        "animx_clan_bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN,
        workdir="./",
        in_memory=True,
    )

    await pyro_client.start()

    player = MusicPlayer(pyro_client)
    await player.start()

    # Store shared objects in app.bot_data
    app.bot_data["pyro_client"] = pyro_client
    app.bot_data["player"] = player

    logger.info("Startup completed.")


async def on_shutdown(app: Application) -> None:
    logger.info("Shutting down PyTgCalls and Pyrogram client...")

    player: MusicPlayer = app.bot_data.get("player")
    if player:
        try:
            await player.shutdown()
        except Exception:
            logger.exception("Error while shutting down player")

    pyro_client: Client = app.bot_data.get("pyro_client")
    if pyro_client:
        try:
            await pyro_client.stop()
        except Exception:
            logger.exception("Error while stopping Pyrogram client")

    logger.info("Shutdown complete.")


def main() -> None:
    application: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register lifecycle hooks
    application.post_init = on_startup
    application.post_shutdown = on_shutdown

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("play", play))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("skip", skip))
    application.add_handler(CommandHandler("stop", stop_cmd))

    # Error handler
    application.add_error_handler(error_handler)

    logger.info("%s (%s) is starting...", BOT_NAME, BOT_USERNAME)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
