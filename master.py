import asyncio
import telebot.async_telebot as telebot
from db import MatchCache
from config import api_token
from scraper import OddsScraper
from scraper import OddsUpdater
from bot import register_handlers

async def main():
    telegram_bot = telebot.AsyncTeleBot(api_token)
    cache = MatchCache()
    await cache.start()
    print(f"Redis is up: {await cache.ping()}")

    scraper = OddsScraper()
    await scraper.start()

    updater = OddsUpdater(scraper, cache, update_interval=900)
    asyncio.create_task(updater.start())

    register_handlers(telegram_bot, cache)
    print("Bot started")
    await telegram_bot.infinity_polling()


if __name__ == "__main__":
    asyncio.run(main())