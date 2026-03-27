import asyncio
import telebot.async_telebot as telebot
from bot import register_handlers
import uvicorn
from fastapi import FastAPI
from db import MatchCache
from config import api_token
from scraper import OddsScraper, OddsUpdater
from api import register_api
from fastapi.middleware.cors import CORSMiddleware

async def run_api(app):
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    cache = MatchCache()
    await cache.start()
    print(f"Redis is up: {await cache.ping()}")

    scraper = OddsScraper()
    await scraper.start()

    updater = OddsUpdater(scraper, cache, update_interval=900)
    asyncio.create_task(updater.start())
    print("Updater started")

    telegram_bot = telebot.AsyncTeleBot(api_token)
    register_handlers(telegram_bot, cache)
    print("Bot started")

    app = FastAPI()
    origins = [
        "http://localhost:3000",
        "http://5.42.105.7:3000",
        "http://72.56.234.189:3000",
        "http://localhost:5173"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_api(app, cache)


    await asyncio.gather(
        telegram_bot.infinity_polling(),
        run_api(app),
    )

if __name__ == "__main__":
    asyncio.run(main())