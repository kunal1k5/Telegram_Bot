# Baby Userbot - Setup Guide 💕

## What is this?
This is a **Telegram USERBOT** that replies to ALL messages using your personal account.

## Requirements
1. **Telegram Account** (your phone number)
2. **API Credentials** from https://my.telegram.org
3. **Gemini API Key** (you already have one)

---

## Step 1: Get API Credentials

1. Go to https://my.telegram.org
2. Login with your phone number
3. Click "API development tools"
4. Create an app (any name)
5. Copy:
   - **API_ID** (number like 12345678)
   - **API_HASH** (string like "abc123def456")

---

## Step 2: Install Dependencies

```bash
pip install -r userbot_requirements.txt
```

---

## Step 3: Set Environment Variables

### On Windows PowerShell:
```powershell
$env:API_ID="YOUR_API_ID"
$env:API_HASH="YOUR_API_HASH"
$env:PHONE_NUMBER="+91XXXXXXXXXX"  # Your phone with country code
$env:OPENROUTER_API_KEY="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"
$env:OPENROUTER_MODEL="openai/gpt-4o-mini"
$env:GEMINI_API_KEY=""  # Optional fallback
```

### On Linux/Mac:
```bash
export API_ID="YOUR_API_ID"
export API_HASH="YOUR_API_HASH"
export PHONE_NUMBER="+91XXXXXXXXXX"
export OPENROUTER_API_KEY="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"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
export GEMINI_API_KEY=""
```

---

## Step 4: Run the Userbot

```bash
python userbot.py
```

### First Time Login:
1. It will ask for **phone number** (enter with country code: +91XXXXXXXXXX)
2. You'll get a **code on Telegram** - enter it
3. If you have **2FA**, enter password
4. Done! ✅

---

## Step 5: Test It

1. Go to any group where your account is member
2. Send any message
3. **Baby will reply automatically!** ❤️

---

## Features ✨

- ✅ Replies to **ALL** group messages
- ✅ Replies to private messages
- ✅ Fast replies (< 2 seconds)
- ✅ **Hinglish** by default
- ✅ Language switching:
  - "English me bolo" → English
  - "Hindi me bolo" → Hinglish
- ✅ Cute personality with emojis 🥰❤️✨

---

## Important Notes ⚠️

1. **This is a USERBOT** - it uses YOUR account, not a bot token
2. **No privacy mode issues** - userbots can see all messages
3. **Be careful** - Don't spam or violate Telegram ToS
4. **Session file** - `baby_userbot.session` will be created (keep it safe!)
5. **API limits** - Telegram has rate limits, don't spam too fast

---

## Troubleshooting 🔧

### "API_ID and API_HASH required"
- Make sure you set environment variables correctly
- Get them from https://my.telegram.org

### "Invalid phone number"
- Use international format: +91XXXXXXXXXX
- Include country code with +

### "Gemini API error"
- Check if API key is correct
- Try a new Gemini API key from https://makersuite.google.com/app/apikey

### "FloodWait" error
- Telegram rate limit hit
- Wait for the specified time
- Don't reply to too many messages too fast

---

## Stop the Userbot

Press `Ctrl + C` in terminal

---

## Example Usage

**User in group:** "Hello kaise ho?"  
**Baby:** "Haan yaar! Mai theek hoon ❤️ Tum batao? 🥰"

**User:** "English me bolo"  
**Baby:** "Sure! I'll speak in English now! ❤️"

**User:** "How are you?"  
**Baby:** "I'm great! How about you? ✨"

---

## Security 🔒

- Never share your `baby_userbot.session` file
- Never share your API_HASH
- Keep your credentials private
- Don't give session file to anyone

---

Enjoy your cute Baby userbot! 💕✨🥰
