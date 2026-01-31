import os

# ================== ANIMX CLAN MUSIC BOT CONFIG ==================
# Fill in these values before running the bot.

# Get these from https://my.telegram.org/apps
API_ID: int = int(os.getenv("API_ID", "123456"))
API_HASH: str = os.getenv("API_HASH", "your_api_hash_here")

# Get this from @BotFather
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "your_bot_token_here")

# Optional: Owner / admin user ID (for future extensions)
OWNER_ID: int | None = int(os.getenv("OWNER_ID", "0")) or None

# Directory where downloaded audio files will be stored temporarily
DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "downloads")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
