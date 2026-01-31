# ANIMX CLAN Telegram Music Bot

ANIMX CLAN is a production-ready Telegram music bot that plays high-quality audio in group voice chats.

- **Bot Name:** ANIMX CLAN
- **Bot Username:** @AnimxClanBot
- **Language:** Python 3.10+
- **Libraries:** python-telegram-bot, pyrogram, pytgcalls, yt-dlp, ffmpeg

---

## Features

- `/start` – Show welcome message
- `/play <song name or YouTube URL>` – Play music in the group voice chat
- `/pause` – Pause current track
- `/resume` – Resume playback
- `/skip` – Skip current track
- `/stop` – Stop music, clear queue, and leave voice chat
- Multi-group support
- Automatic join/leave of group voice chats
- High-quality audio streaming
- Graceful error handling and basic anti-spam checks
- Automatic cleanup of downloaded files

---

## 1. Prerequisites

- Python **3.10+**
- A Telegram **Bot Token** from [@BotFather](https://t.me/BotFather)
- A Telegram **API ID** and **API Hash** from https://my.telegram.org/apps
- **ffmpeg** installed and available in your system `PATH`

### Install ffmpeg

#### Windows

1. Download a static build from: https://www.gyan.dev/ffmpeg/builds/
2. Extract the archive (e.g. to `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your **System PATH**:
   - Search for "Environment Variables" → **Edit the system environment variables**.
   - Click **Environment Variables...**.
   - Select `Path` under **System variables** → **Edit** → **New** → add `C:\ffmpeg\bin`.
   - Click **OK** on all dialogs.
4. Open a new terminal and run:
   ```bash
   ffmpeg -version
   ```
   You should see version output.

#### Ubuntu / Debian (Linux)

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
```

#### CentOS / RHEL

```bash
sudo yum install -y epel-release
sudo yum install -y ffmpeg
ffmpeg -version
```

---

## 2. Installation

Clone or copy the `ANIMX_MUSIC_BOT` folder to your server or machine.

```bash
cd ANIMX_MUSIC_BOT
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> On Windows PowerShell, activate with:
>
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

---

## 3. Configuration

Open `config.py` and fill in your credentials:

```python
API_ID = 123456  # from my.telegram.org
API_HASH = "your_api_hash_here"
BOT_TOKEN = "123456:ABC-DEF_your_bot_token_here"
```

You can also set these via environment variables (`API_ID`, `API_HASH`, `BOT_TOKEN`) if preferred.

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
