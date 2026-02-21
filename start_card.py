from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1280
HEIGHT = 720


def get_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    if 12 <= hour < 18:
        return "Good Afternoon"
    if 18 <= hour < 23:
        return "Good Evening"
    return "Late Night Vibes"


def _load_base_image(base_image_path: str) -> Image.Image:
    path = Path(base_image_path)
    if path.exists():
        return Image.open(path).convert("RGB").resize((WIDTH, HEIGHT))

    # Fallback gradient background if banner file is missing.
    img = Image.new("RGB", (WIDTH, HEIGHT), (16, 20, 42))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        r = 16 + int((48 * y) / max(1, HEIGHT - 1))
        g = 20 + int((28 * y) / max(1, HEIGHT - 1))
        b = 42 + int((84 * y) / max(1, HEIGHT - 1))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def _load_profile_image(profile_photo_url: str) -> Optional[Image.Image]:
    if not profile_photo_url:
        return None
    try:
        response = requests.get(profile_photo_url, timeout=6)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def create_start_card(base_image_path: str, user_name: str, profile_photo_url: Optional[str] = None) -> BytesIO:
    bg = _load_base_image(base_image_path).filter(ImageFilter.GaussianBlur(0.2))
    greeting = get_greeting()

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    strip = Image.new("RGBA", (WIDTH, 230), (0, 0, 0, 155))
    overlay.paste(strip, (0, HEIGHT - 270))
    canvas = Image.alpha_composite(bg.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(canvas)

    title_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

    draw.text((80, HEIGHT - 235), "BABY • Premium Music AI", fill=(255, 255, 255), font=title_font)
    draw.text((80, HEIGHT - 195), f"{greeting}, {user_name}", fill=(255, 210, 245), font=text_font)
    draw.text((80, HEIGHT - 155), "Cinematic music + chat companion", fill=(210, 220, 240), font=text_font)
    draw.text((80, HEIGHT - 120), "Fast • Smart • Always Active", fill=(210, 220, 240), font=text_font)

    pfp = _load_profile_image(profile_photo_url or "")
    if pfp is not None:
        pfp = pfp.resize((130, 130))
        mask = Image.new("L", (130, 130), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.ellipse((0, 0, 129, 129), fill=255)
        canvas.paste(pfp, (WIDTH - 190, HEIGHT - 235), mask)

    output = BytesIO()
    canvas.convert("RGB").save(output, format="JPEG", quality=95)
    output.seek(0)
    output.name = "start_panel.jpg"
    return output

