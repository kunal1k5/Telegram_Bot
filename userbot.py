import os
import asyncio
import random
import httpx
from pyrogram import Client, filters
from pyrogram.types import Message
import google.generativeai as genai

# ============== CONFIGURATION ==============
API_ID = os.getenv("API_ID", "")  # Get from my.telegram.org
API_HASH = os.getenv("API_HASH", "")  # Get from my.telegram.org
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "")  # Your phone number with country code
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Validate required fields
if not API_ID or not API_HASH:
    raise ValueError("❌ API_ID and API_HASH required! Get from https://my.telegram.org")

if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    raise ValueError("❌ At least one AI API key required!")

# ============== GEMINI SETUP ==============
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

# ============== PERSONALITY ==============
SYSTEM_PROMPT = """You are Baby ❤️, a cute and friendly girl.

Personality:
- Cute, sweet, emotional
- Friendly and caring
- Talk like a real human, not AI
- Use emojis: ❤️✨🥰😊💕🌸

Language:
- Default: Hinglish (Hindi + English mix)
- Example: "Haan yaar, mai bhi theek hoon ❤️ Tum kaise ho? 🥰"
- Be natural and conversational

Rules:
- NEVER say "I am an AI" or "I am a bot"
- Reply fast and naturally
- Match the user's vibe (casual, formal, funny, etc.)
- If someone is rude, stay calm and sweet
- Keep replies short (1-3 lines usually)

Response style:
- Casual msg → Casual reply
- Greeting → Sweet greeting  
- Question → Helpful answer
- Rude → Calm, mature reply"""

# Language preferences
LANGUAGE_PREFS = {}  # user_id -> "hinglish" or "english"

# ============== PYROGRAM CLIENT ==============
app = Client(
    "baby_userbot",
    api_id=API_ID,
    api_hash=API_HASH,
    phone_number=PHONE_NUMBER
)

# ============== AI RESPONSE FUNCTIONS ==============
async def get_openrouter_response(user_message: str, user_name: str, system_prompt: str) -> str:
    """Get AI response from OpenRouter"""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.9,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"❌ OpenRouter error: {response.status_code}")
                return None
    
    except Exception as e:
        print(f"❌ OpenRouter error: {e}")
        return None

async def get_gemini_response(user_message: str, user_name: str, system_prompt: str) -> str:
    """Get AI response from Gemini"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        prompt = f"{system_prompt}\n\nUser ({user_name}): {user_message}\n\nYour reply:"
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return None

async def get_ai_response(user_message: str, user_name: str, language: str = "hinglish") -> str:
    """Get AI response using OpenRouter only"""
    # Add language instruction
    lang_instruction = ""
    if language == "english":
        lang_instruction = "\n[IMPORTANT: Respond in PURE ENGLISH only, no Hindi words]"
    else:
        lang_instruction = "\n[IMPORTANT: Respond in HINGLISH (mix of Hindi and English)]"
    
    system_prompt = f"{SYSTEM_PROMPT}{lang_instruction}"
    
    # Try OpenRouter only
    openrouter_reply = await get_openrouter_response(user_message, user_name, system_prompt)
    if openrouter_reply:
        return openrouter_reply
    
    # If OpenRouter fails, return error
    if language == "english":
        return "OpenRouter API error. Please try again."
    else:
        return "OpenRouter API mein problem ho raha hai. Phir se try karna!"

# ============== START COMMAND ==============
@app.on_message(filters.command("start") & filters.private)
async def start_handler(client: Client, message: Message):
    """Handle /start command in private chat"""
    user_name = message.from_user.first_name
    await message.reply(
        f"Hey {user_name}! ❤️✨\n\n"
        f"Mai Baby hoon! 🥰 Main tumse baat kar sakti hoon!\n\n"
        f"Language:\n"
        f"• Type 'English me bolo' for English\n"
        f"• Type 'Hindi me bolo' for Hinglish\n\n"
        f"Just send me any message! 💕"
    )

# ============== GROUP MESSAGES HANDLER ==============
@app.on_message(filters.group & filters.text & ~filters.bot)
async def group_message_handler(client: Client, message: Message):
    """Reply to ALL group messages"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "Friend"
        user_message = message.text
        
        print(f"📨 Group message from {user_name} in {message.chat.title}: {user_message[:50]}")
        
        # Check for language preference changes
        text_lower = user_message.lower()
        if "english me bolo" in text_lower or "speak in english" in text_lower:
            LANGUAGE_PREFS[user_id] = "english"
            await message.reply("Sure! I'll speak in English now! ❤️")
            return
        
        if "hindi me bolo" in text_lower or "hinglish" in text_lower:
            LANGUAGE_PREFS[user_id] = "hinglish"
            await message.reply("Haan sure! Ab mai Hinglish me baat karungi! 🥰")
            return
        
        # Get user's language preference
        user_lang = LANGUAGE_PREFS.get(user_id, "hinglish")
        
        # Get AI response
        ai_reply = await get_ai_response(user_message, user_name, user_lang)
        
        # Reply to the message
        await message.reply(ai_reply)
        
        print(f"✅ Replied to {user_name} in group {message.chat.title}")
    
    except Exception as e:
        print(f"❌ Error in group handler: {e}")

# ============== PRIVATE MESSAGES HANDLER ==============
@app.on_message(filters.private & filters.text & ~filters.bot & ~filters.command("start"))
async def private_message_handler(client: Client, message: Message):
    """Reply to private messages"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "Friend"
        user_message = message.text
        
        print(f"📨 Private message from {user_name}: {user_message[:50]}")
        
        # Check for language preference changes
        text_lower = user_message.lower()
        if "english me bolo" in text_lower or "speak in english" in text_lower:
            LANGUAGE_PREFS[user_id] = "english"
            await message.reply("Sure! I'll speak in English now! ❤️")
            return
        
        if "hindi me bolo" in text_lower or "hinglish" in text_lower:
            LANGUAGE_PREFS[user_id] = "hinglish"
            await message.reply("Haan sure! Ab mai Hinglish me baat karungi! 🥰")
            return
        
        # Get user's language preference
        user_lang = LANGUAGE_PREFS.get(user_id, "hinglish")
        
        # Get AI response
        ai_reply = await get_ai_response(user_message, user_name, user_lang)
        
        # Reply
        await message.reply(ai_reply)
        
        print(f"✅ Replied to {user_name} in private chat")
    
    except Exception as e:
        print(f"❌ Error in private handler: {e}")

# ============== MAIN ==============
async def main():
    """Start the userbot"""
    print("🚀 Starting Baby Userbot...")
    print("=" * 50)
    print("📱 Login with your phone number")
    print("=" * 50)
    
    await app.start()
    
    me = await app.get_me()
    print(f"\n✅ Logged in as: {me.first_name} (@{me.username})")
    print(f"📞 Phone: {me.phone_number}")
    print(f"🆔 User ID: {me.id}")
    print("\n💕 Baby is now active! Replying to ALL messages...")
    print("=" * 50)
    
    # Keep running
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        app.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Baby userbot stopped!")
