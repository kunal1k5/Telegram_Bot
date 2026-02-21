from pathlib import Path
import runpy
import sys


# Compatibility launcher: always run the modular project bot.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "project"
TARGET = PROJECT_DIR / "bot.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Expected bot entrypoint not found: {TARGET}")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

runpy.run_path(str(TARGET), run_name="__main__")
