from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1280
HEIGHT = 720


def get_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "🌅 Good Morning"
    if 12 <= hour < 18:
        return "🌞 Good Afternoon"
    if 18 <= hour < 23:
        return "🌙 Good Evening"
    return "🌌 Late Night Vibes"


def _load_profile_image(profile_photo_url: str) -> Optional[Image.Image]:
    if not profile_photo_url:
        return None
    try:
        response = requests.get(profile_photo_url, timeout=8)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def create_dynamic_start_card(user_name: str, profile_photo_url: Optional[str] = None) -> BytesIO:
    pfp_src = _load_profile_image(profile_photo_url or "")

    # Background: blurred profile photo if available, otherwise deep fallback color.
    if pfp_src is not None:
        bg = pfp_src.copy()
    else:
        bg = Image.new("RGB", (WIDTH, HEIGHT), (30, 20, 60))

    bg = bg.resize((WIDTH, HEIGHT)).filter(ImageFilter.GaussianBlur(25))

    # Dark overlay for readability.
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 140))
    canvas = Image.alpha_composite(bg.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(canvas)

    greeting = get_greeting()

    # Bigger fonts with fallback.
    try:
        title_font = ImageFont.truetype("arial.ttf", 90)
        sub_font = ImageFont.truetype("arial.ttf", 55)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Center text.
    title_text = f"HEY BABY {user_name}"
    sub_text = "ANIMX MUSIC"

    draw.text(
        (WIDTH / 2, HEIGHT / 2 - 120),
        title_text,
        fill=(255, 255, 255),
        font=title_font,
        anchor="mm",
    )

    draw.text(
        (WIDTH / 2, HEIGHT / 2 - 20),
        sub_text,
        fill=(255, 180, 255),
        font=sub_font,
        anchor="mm",
    )

    draw.text(
        (WIDTH / 2, HEIGHT / 2 + 80),
        greeting,
        fill=(255, 200, 255),
        font=sub_font,
        anchor="mm",
    )

    # Profile circle highlight (bottom-right).
    if pfp_src is not None:
        pfp = pfp_src.resize((220, 220))
        mask = Image.new("L", (220, 220), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, 220, 220), fill=255)
        canvas.paste(pfp, (WIDTH - 300, HEIGHT - 300), mask)

    output = BytesIO()
    canvas.convert("RGB").save(output, format="JPEG", quality=95)
    output.seek(0)
    output.name = "start_panel.jpg"
    return output
