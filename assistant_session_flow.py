import argparse
import asyncio
import json
import os
from pathlib import Path

from telethon import TelegramClient
from telethon.sessions import StringSession


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / ".assistant_login_state.json"
TEMP_SESSION = BASE_DIR / "assistant_temp"


def _read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _write_state(data: dict) -> None:
    STATE_FILE.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")


async def send_code(api_id: int, api_hash: str, phone: str) -> None:
    client = TelegramClient(str(TEMP_SESSION), api_id, api_hash)
    await client.connect()
    try:
        sent = await client.send_code_request(phone)
        _write_state(
            {
                "api_id": api_id,
                "api_hash": api_hash,
                "phone": phone,
                "phone_code_hash": sent.phone_code_hash,
            }
        )
        print("OTP sent successfully. Now run: python assistant_session_flow.py verify --code <OTP>")
    finally:
        await client.disconnect()


async def verify_code(code: str, password: str | None) -> None:
    state = _read_state()
    if not state:
        raise SystemExit("No pending login state found. Run send first.")

    client = TelegramClient(str(TEMP_SESSION), state["api_id"], state["api_hash"])
    await client.connect()
    try:
        try:
            await client.sign_in(
                phone=state["phone"],
                code=code,
                phone_code_hash=state["phone_code_hash"],
            )
        except Exception:
            if not password:
                raise SystemExit("2FA enabled. Run verify again with --password <your_2fa_password>")
            await client.sign_in(password=password)

        session_str = StringSession.save(client.session)
        print("\nASSISTANT_SESSION:")
        print(session_str)
        try:
            STATE_FILE.unlink(missing_ok=True)
            for path in BASE_DIR.glob("assistant_temp.session*"):
                path.unlink(missing_ok=True)
        except Exception:
            pass
    finally:
        await client.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Telethon assistant session in two steps.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    send = sub.add_parser("send", help="Send OTP to phone")
    send.add_argument("--api-id", type=int, default=int(os.getenv("API_ID", "0")))
    send.add_argument("--api-hash", type=str, default=os.getenv("API_HASH", ""))
    send.add_argument("--phone", type=str, required=True)

    verify = sub.add_parser("verify", help="Verify OTP and print ASSISTANT_SESSION")
    verify.add_argument("--code", type=str, required=True)
    verify.add_argument("--password", type=str, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.cmd == "send":
        if not args.api_id or not args.api_hash:
            raise SystemExit("API credentials missing. Provide --api-id and --api-hash (or set API_ID/API_HASH).")
        await send_code(args.api_id, args.api_hash, args.phone)
        return
    await verify_code(args.code, args.password)


if __name__ == "__main__":
    asyncio.run(main())
