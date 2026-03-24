import asyncio
import telebot.async_telebot as telebot
from db import MatchCache
from web import create_app
from config import api_token
from scraper import OddsScraper
from scraper import OddsUpdater
from bot import register_handlers
import uvicorn

async def main():
    cache = MatchCache()
    await cache.start()
    print(f"Redis is up: {await cache.ping()}")

    scraper = OddsScraper()
    await scraper.start()

    updater = OddsUpdater(scraper, cache, update_interval=900)
    asyncio.create_task(updater.start())
    print("Updater started")

    app = create_app(cache)
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    print("Server started")

    telegram_bot = telebot.AsyncTeleBot(api_token)
    register_handlers(telegram_bot, cache)
    print("Bot started")

    await telegram_bot.infinity_polling()


if __name__ == "__main__":
    asyncio.run(main())