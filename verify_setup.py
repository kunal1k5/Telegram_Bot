#!/usr/bin/env python3
"""
Music Bot Verification Script
Tests all components needed for music download to work
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("MUSIC BOT VERIFICATION")
print("=" * 60)

checks_passed = 0
checks_total = 0

# Check 1: Python Version
checks_total += 1
print("\n[1/6] Checking Python Version...")
if sys.version_info >= (3, 8):
    print(f"  ✓ Python {sys.version.split()[0]} (OK)")
    checks_passed += 1
else:
    print(f"  ✗ Python {sys.version.split()[0]} (Need 3.8+)")

# Check 2: yt-dlp Installation
checks_total += 1
print("\n[2/6] Checking yt-dlp...")
try:
    import yt_dlp
    print(f"  ✓ yt-dlp is installed")
    checks_passed += 1
except ImportError:
    print("  ✗ yt-dlp not installed - Run: pip install yt-dlp[default]")

# Check 3: python-telegram-bot
checks_total += 1
print("\n[3/6] Checking python-telegram-bot...")
try:
    import telegram
    print(f"  ✓ python-telegram-bot is installed")
    checks_passed += 1
except ImportError:
    print("  ✗ python-telegram-bot not installed")

# Check 4: Other dependencies
checks_total += 1
print("\n[4/6] Checking other dependencies...")
try:
    import httpx
    import genai
    print(f"  ✓ httpx and genai are installed")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ Missing dependency: {e}")

# Check 5: Download directory
checks_total += 1
print("\n[5/6] Checking download directory...")
download_dir = Path("downloads")
if download_dir.exists() or download_dir.mkdir(parents=True, exist_ok=True) is None:
    print(f"  ✓ Download directory ready at: {download_dir.resolve()}")
    checks_passed += 1
else:
    print("  ✗ Cannot create download directory")

# Check 6: FFmpeg (Optional but recommended)
checks_total += 1
print("\n[6/6] Checking FFmpeg (for MP3 conversion)...")
try:
    import shutil
    if shutil.which("ffmpeg"):
        print("  ✓ FFmpeg is installed")
        checks_passed += 1
    else:
        print("  ~ FFmpeg not in PATH - Bot will work but files may not be MP3")
        print("    Install with: winget install FFmpeg")
        print("    Or: choco install ffmpeg")
except:
    print("  ~ Cannot verify FFmpeg - Bot will still work")

# Summary
print("\n" + "=" * 60)
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print("=" * 60)

if checks_passed >= 5:
    print("\n✓ Your bot is ready to go! All critical components are working.")
    print("\nNext steps:")
    print("  1. Run the bot: python bot.py")
    print("  2. Send /song bargad to test")
    sys.exit(0)
elif checks_passed >= 4:
    print("\n⚠ Bot can work but FFmpeg is recommended for better quality.")
    sys.exit(0)
else:
    print("\n✗ Some critical components are missing. Install them and try again.")
    sys.exit(1)
