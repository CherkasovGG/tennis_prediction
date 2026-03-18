import threading

import telebot
from db import MatchCache
from config import api_token
from scraper import OddsScraper
from scraper import OddsUpdater

if __name__ == "__main__":
    telegram_bot = telebot.TeleBot(api_token)
    cache = MatchCache()
    scraper = OddsScraper()
    updater = OddsUpdater(scraper, cache, update_interval=900)
    threading.Thread(target=updater.start, daemon=True).start()
    print("Bot started")
    telegram_bot.infinity_polling()