from __future__ import annotations

from io import BytesIO
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1280
HEIGHT = 720


def _load_image_from_url(url: str, timeout: int = 10) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _fallback_base_image() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (24, 24, 30))


def create_music_card(thumbnail_url: str, title: str, duration: str, requester: str) -> BytesIO:
    thumb = _load_image_from_url(thumbnail_url) or _fallback_base_image()

    bg = thumb.resize((WIDTH, HEIGHT))
    bg = bg.filter(ImageFilter.GaussianBlur(30))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 180))
    canvas = Image.alpha_composite(bg.convert("RGBA"), overlay)

    cover = thumb.resize((450, 450))
    canvas.paste(cover, (120, 135))

    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    # Slightly larger pseudo-title by drawing twice for bold-ish readability with default font.
    x, y = 650, 280
    draw.text((650, 230), "NOW PLAYING", font=small_font, fill=(255, 255, 255))
    draw.text((x, y), (title or "Unknown Track")[:45], font=title_font, fill=(255, 255, 255))
    draw.text((x + 1, y), (title or "Unknown Track")[:45], font=title_font, fill=(255, 255, 255))
    draw.text((650, 330), f"Duration: {duration}", font=small_font, fill=(210, 210, 210))
    draw.text((650, 370), f"Requested by: {requester}", font=small_font, fill=(210, 210, 210))

    output = BytesIO()
    canvas.convert("RGB").save(output, format="JPEG", quality=95)
    output.seek(0)
    output.name = "now_playing.jpg"
    return output

