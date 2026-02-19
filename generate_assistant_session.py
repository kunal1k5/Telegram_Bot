import asyncio
import os
from getpass import getpass

from telethon import TelegramClient
from telethon.sessions import StringSession


async def main() -> None:
    api_id_raw = os.getenv("API_ID") or input("Enter API_ID: ").strip()
    api_hash = os.getenv("API_HASH") or input("Enter API_HASH: ").strip()

    if not api_id_raw.isdigit():
        raise SystemExit("API_ID must be numeric.")
    if not api_hash:
        raise SystemExit("API_HASH is required.")

    api_id = int(api_id_raw)
    session = StringSession()

    async with TelegramClient(session, api_id, api_hash) as client:
        if not await client.is_user_authorized():
            phone = input("Enter assistant phone number with country code (e.g. +9198xxxxxx): ").strip()
            sent = await client.send_code_request(phone)
            code = input("Enter OTP code from Telegram: ").strip()
            try:
                await client.sign_in(phone=phone, code=code, phone_code_hash=sent.phone_code_hash)
            except Exception:
                password = getpass("2FA password (if enabled): ")
                await client.sign_in(password=password)

        print("\nASSISTANT_SESSION (save in Railway env):")
        print(client.session.save())


if __name__ == "__main__":
    asyncio.run(main())
