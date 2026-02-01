# ANIMX CLAN Telegram Bot - Baby ❤️

A feature-rich Telegram bot with AI chat, music downloads, admin tools, and anti-spam protection.

- **Bot Name:** Baby ❤️ (ANIMX CLAN)
- **Bot Username:** @AnimxClanBot
- **Language:** Python 3.10+
- **Framework:** python-telegram-bot 21.4
- **AI:** Google Gemini + OpenRouter (fallback)

---

## ✨ Features

### 🎵 Music Download
- `/song <name>` - Download songs with Telegram music player
- `/download <name>` - Same as /song
- `/yt <link>` - Download from YouTube URL
- `play <song name>` - Quick play without command
- MP3 conversion with metadata (title, artist, duration)
- Automatic file cleanup

### 💬 AI Chat
- Natural Hinglish conversations
- Works in private chats and groups
- Smart group triggers (mention, reply, "baby" keyword)
- Language switching (English/Hindi/Hinglish)
- Personality: Friendly, cute, human-like "Baby"

### 📢 Broadcasting
- `/broadcast <message>` - Send to all users (admin only)
- `/stop` - Opt out of broadcasts
- Automatic opt-out tracking
- Group + user broadcast support

### 🎯 Dynamic Commands (37+)
**Greetings:** `/gm`, `/ga`, `/ge`, `/gn`, `/bye`, `/welcome`, `/thanks`, `/sorry`, `/mood`
**Chat:** `/chat`, `/ask`, `/about`, `/privacy`
**Emotions:** `/sad`, `/happy`, `/angry`, `/motivate`, `/howareyou`, `/missyou`, `/hug`
**Productivity:** `/tip`, `/confidence`, `/focus`, `/sleep`, `/lifeline`
**Fun:** `/joke`, `/roast`, `/truth`, `/dare`, `/fact`

### 🛡️ Admin Moderation (9 commands)
- `/del` - Delete messages
- `/ban` / `/unban` - Ban management
- `/mute` / `/unmute` - Mute management
- `/promote` / `/demote` - Admin management
- `/pin` / `/unpin` - Pin messages
- `/admin` - Admin help menu (admin-only)

### 🚫 Anti-Spam System
- Detects repeated messages (3x auto-delete)
- Prevents flooding (5 msgs/10s)
- Blocks spam links
- Stops emoji floods (10+ emojis)
- Never moderates admins
- Friendly warnings: "Thoda slow 🙂 spam mat karo"

### 👥 Group Features
- `/all <message>` - Tag active users (admin only)
- `@all <message>` - Quick tag mention
- Auto-group registration
- Cooldown system (5 min)

---

## 📋 Prerequisites

- Python **3.10+**
- **Bot Token** from [@BotFather](https://t.me/BotFather)
- **Google Gemini API Key** from [AI Studio](https://aistudio.google.com/app/apikey)
- **FFmpeg** (for audio conversion)

### Install FFmpeg

#### Windows
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH
4. Verify: `ffmpeg -version`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y ffmpeg
```

---

## 🚀 Installation

### Local Setup

## 🚀 Installation

### Local Setup

```bash
cd ANIMX_MUSIC_BOT
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Method 1: Environment Variables (Recommended for Railway)

Set these in Railway dashboard or `.env` file:

```env
BOT_TOKEN=your_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
ADMIN_ID=your_telegram_user_id
```

### Method 2: Direct in Code (Local Testing)

Edit `bot.py` and set:
```python
BOT_TOKEN = "your_bot_token_here"
GEMINI_API_KEY = "your_gemini_key_here"
ADMIN_ID = your_user_id
```

---

## 🎮 Running the Bot

### Local Run
```bash
cd ANIMX_MUSIC_BOT
python bot.py
```

### Railway Deployment

1. Push to GitHub:
```bash
git add .
git commit -m "Deploy bot"
git push origin main
```

2. Railway auto-deploys from GitHub
3. Set environment variables in Railway dashboard
4. Bot starts automatically!

See [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md) for detailed Railway instructions.

---

## 📖 Usage

### Private Chat
- Just message the bot naturally
- Use commands like `/song <name>` for music
- Bot responds with AI personality

### Group Chat
Bot responds when:
- Mentioned: `@AnimxClanBot hello`
- Replied to
- "baby" keyword detected
- Greetings: hi, hello, gm, gn
- Commands used

### Admin Commands (Groups only)
1. Add bot to group
2. Make bot admin
3. Use admin commands: `/del`, `/ban`, `/mute`, etc.
4. View admin help: `/admin`

---

## 🔧 Troubleshooting

### Bot not responding?
- Check BOT_TOKEN is correct
- Verify GEMINI_API_KEY is active
- In groups, mention bot or reply to it

### Song download fails?
- Check FFmpeg is installed: `ffmpeg -version`
- Ensure yt-dlp is updated: `pip install -U yt-dlp`
- Verify internet connection

### Gemini API errors?
- Check API key at https://aistudio.google.com
- Verify API quota not exceeded
- OpenRouter will be used as fallback if configured

---

## 📁 File Structure

```
ANIMX_MUSIC_BOT/
├── bot.py                    # Main bot file (3000+ lines)
├── config.py                 # Configuration (optional)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── RAILWAY_DEPLOY.md         # Railway deployment guide
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
├── users_database.json      # Registered users (auto-created)
├── groups_database.json     # Registered groups (auto-created)
├── opted_out_users.json     # Broadcast opt-outs (auto-created)
├── downloads/               # Temporary song downloads (auto-cleanup)
└── utils/
    └── yt.py               # YouTube download utilities
```

---

## 🎯 Command List

### Basic
- `/start` - Welcome message
- `/help` - Command list
- `/stop` - Opt out of broadcasts

### Music
- `/song <name>` - Download song
- `/yt <url>` - Download from YouTube
- `play <name>` - Quick play

### Chat & Info
- `/chat <message>` - AI chat
- `/ask <question>` - Ask anything
- `/about` - Bot info
- `/privacy` - Privacy policy

### Emotions & Support
- `/sad`, `/happy`, `/angry`
- `/motivate`, `/confidence`
- `/howareyou`, `/missyou`, `/hug`

### Fun
- `/joke`, `/roast`, `/fact`
- `/truth`, `/dare`

### Admin (Groups)
- `/del` - Delete message
- `/ban` / `/unban` - Ban user
- `/mute` / `/unmute` - Mute user
- `/promote` / `/demote` - Admin management
- `/pin` / `/unpin` - Pin management
- `/admin` - Admin help

### Group Utils
- `/all <message>` - Tag active users
- `@all <message>` - Quick tag

---

## 🤝 Credits

- **Developer:** @kunal1k5
- **Bot:** Baby ❤️ (ANIMX CLAN)
- **Framework:** python-telegram-bot
- **AI:** Google Gemini
- **Music:** yt-dlp + FFmpeg

---

## 📜 License

This bot is for personal/educational use. 

Made with ❤️ by ANIMX CLAN

The `DOWNLOAD_DIR` directory (default: `downloads`) is used for temporary audio files and is created automatically.

---

## 4. Running the Bot (Windows & Linux)

From the `ANIMX_MUSIC_BOT` directory with your virtual environment activated:

```bash
python bot.py
```

You should see logs indicating that ANIMX CLAN has started and is polling for updates.

---

## 5. Using the Bot in Groups

1. Start the bot with `python bot.py`.
2. Open Telegram and search for `@AnimxClanBot`.
3. Add the bot to your **group**.
4. Give it the necessary **admin permissions** (at least: read messages, manage voice chats is not strictly required but recommended in some setups).
5. Start a **voice chat** in the group.
6. In the group chat, send:

   ```
   /play never gonna give you up
   ```

   or

   ```
   /play https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

7. The bot will join the voice chat and play the requested audio.

Commands:

- `/play <query or URL>` – Queue or start playback.
- `/pause` – Pause the current track.
- `/resume` – Resume playback.
- `/skip` – Skip current track and play next in queue.
- `/stop` – Stop, clear the queue, and leave the voice chat.

Multiple groups are supported; each group gets its own queue.

---

## 6. Running on a VPS (Linux)

1. Install system dependencies and ffmpeg (Ubuntu example):

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip ffmpeg
   ```

2. Upload or `git clone` the `ANIMX_MUSIC_BOT` folder to your VPS.

3. Install Python dependencies:

   ```bash
   cd ANIMX_MUSIC_BOT
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure `config.py` with your credentials.

5. Run the bot:

   ```bash
   python bot.py
   ```

6. (Optional) Use a process manager like `screen`, `tmux`, or `systemd` to keep the bot running:

   - `tmux new -s animx`
   - `python bot.py`
   - Detach with `Ctrl+B`, then `D`.

---

## 7. Notes & Troubleshooting

- Make sure the group has an **active voice chat** before using `/play`; otherwise the join may fail.
- If the bot does not join or play audio:
  - Confirm `ffmpeg` is correctly installed and in `PATH`.
  - Check that your `API_ID`, `API_HASH`, and `BOT_TOKEN` are correct.
  - Look at the console logs for errors.
- The bot uses `yt-dlp` to download audio. Very long or unusual URLs may fail; try a simpler search term instead.

---

## 8. Project Structure

```text
ANIMX_MUSIC_BOT/
├── bot.py              # Main runner (python-telegram-bot commands)
├── config.py           # Bot token, API ID, API hash
├── player.py           # Voice chat & streaming logic (pyrogram + pytgcalls)
├── utils/
│   └── yt.py           # YouTube audio downloader via yt-dlp
├── requirements.txt    # Python dependencies
└── README.md           # Setup & usage instructions
```

Last Railway rebuild trigger.

You can now configure `config.py`, run `python bot.py`, add `@AnimxClanBot` to your groups, start a voice chat, and control music with `/play`, `/pause`, `/resume`, `/skip`, and `/stop`.
