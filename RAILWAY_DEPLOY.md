# 🚂 Railway Deployment Guide - ANIMX CLAN Bot

## Prerequisites
1. Railway account (https://railway.app)
2. Telegram Bot Token from @BotFather
3. **OpenRouter API Key** (https://openrouter.ai/keys) - Recommended
   - OR Google Gemini API Key (https://aistudio.google.com/app/apikey)
4. GitHub repository with the bot code

---

## Step-by-Step Deployment

### 1. Prepare Environment Variables

**Required:**
- `BOT_TOKEN` - Your Telegram bot token from @BotFather

**AI Service (Choose one or both):**
- `OPENROUTER_API_KEY` - OpenRouter API key (Primary, recommended)
- `OPENROUTER_MODEL` - Model to use (default: `openai/gpt-4o-mini`)
- `GEMINI_API_KEY` - Google Gemini API key (Fallback, optional)

**Optional:**
- `ADMIN_ID` - Your Telegram user ID for admin features

### 2. Deploy on Railway

**Option A: Deploy from GitHub**

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository (`kunal1k5/Telegram_Bot`)
5. Railway will auto-detect Python project

**Option B: Deploy with Railway CLI**

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### 3. Set Environment Variables

In Railway dashboard:

1. Click your project
2. Go to **Variables** tab
3. Add these variables:
   - `BOT_TOKEN`: `123456:ABC-DEF...` (your bot token)
   - `OPENROUTER_API_KEY`: `sk-or-v1-...` (your OpenRouter key)
   - `OPENROUTER_MODEL`: `openai/gpt-4o-mini` (optional, this is default)
   - `GEMINI_API_KEY`: `AIza...` (optional fallback)
   - `ADMIN_ID`: `7971841264` (your Telegram user ID)
4. Click "Deploy" to restart with new variables

**Note:** Bot will use OpenRouter first, then fallback to Gemini if needed.

### 4. Configure Service Type

Railway should auto-detect as a **Worker** (not Web service).

If it doesn't:
1. Go to **Settings**
2. Under "Service Type" → Select **Worker**
3. Save changes

### 5. Verify Deployment

Check the **Logs** tab in Railway:

You should see:
```
✅ Bot started: @AnimxClanBot
💬 Ready to chat!
```

---

## Troubleshootingis valid
- Ensure at least one AI API key is set (OPENROUTER_API_KEY or GEMINI_API_KEY)

### AI API errors?
- **OpenRouter:** Verify API key is active at https://openrouter.ai/keys
- **Gemini:** Verify API key is active at https://aistudio.google.com
- Check API quota/limits
- Bot will automatically fallback between services_KEY` are valid

### Gemini API errors?
- Verify API key is active at https://aistudio.google.com
- Check API quota/limits
- Ensure you have Gemini API enabled

### Bot doesn't respond?
- Make sure it's running (check Railway logs)
- Test with `/start` command first
- In groups, mention the bot: `@AnimxClanBot hello`

---

## Local Testing (Optional)

Before deploying to Railway, test locally:

```bash
# Set environment variables
export BOT_TOKEN="your_bot_token"
export GEMINI_API_KEY="your_gemini_key"

# Or on Windows PowerShell:
$env:BOT_TOKEN="your_bot_token"
$env:GEMINI_API_KEY="your_gemini_key"

# Install dependencies
pip install -r requirements.txt

# Run bot
python bot.py
```

---

## Cost

- **Railway**: Free tier available (500 hours/month)
- **Gemini API**: Free tier (60 requests/minute)
- **Total**: $0 for small-scale usage

---

## Support

Issues? Contact: @kunal1k5
Channel: @AnimxClanChannel

Happy chatting! 🎉
