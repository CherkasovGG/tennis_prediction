from collections import defaultdict

import time
import threading
import telebot
from telebot import types
from config.config import api_token

from model_to_predict import predict_match
from get_info import find_match_odds


bot = telebot.TeleBot(api_token)

user_state = {}
MATCH_ODDS = defaultdict(dict)
UPDATE_INTERVAL = 900
FULL_REFRESH_INTERVAL = 43200
MATCH_ODDS_LOCK = threading.Lock()

def odds_updater():
    global MATCH_ODDS

    last_full_refresh = time.time()

    while True:
        try:
            print("[SCRAPER] Обновление коэффициентов...")

            new_data = find_match_odds()

            with MATCH_ODDS_LOCK:

                if time.time() - last_full_refresh > FULL_REFRESH_INTERVAL:
                    print("[SCRAPER] Полный ресет коэффициентов")
                    MATCH_ODDS.clear()
                    MATCH_ODDS.update(new_data)
                    last_full_refresh = time.time()

                else:
                    for match, odds in new_data.items():
                        MATCH_ODDS[match] = odds

                print(f"[SCRAPER] Матчей в базе: {len(MATCH_ODDS)}")

        except Exception as e:
            print("[SCRAPER ERROR]", e)

        time.sleep(UPDATE_INTERVAL)

def make_matches_keyboard():
    markup = types.InlineKeyboardMarkup()

    with MATCH_ODDS_LOCK:
        items = list(MATCH_ODDS.items())

    for (p1, p2), odds in items:
        odds_a, odds_b = odds

        text = f"{p1} vs {p2} ({odds_a:.2f} / {odds_b:.2f})"

        btn = types.InlineKeyboardButton(
            text=text,
            callback_data=f"match:{p1}|{p2}"
        )

        markup.add(btn)

    return markup

def get_pair_odds(p1, p2):
    with MATCH_ODDS_LOCK:
        if (p1, p2) in MATCH_ODDS:
            return MATCH_ODDS[(p1, p2)]

        if (p2, p1) in MATCH_ODDS:
            odds = MATCH_ODDS[(p2, p1)]
            return odds[1], odds[0]

    return None, None

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Отправь мне двух игроков через пробел или запятую,\n"
        "например: Nadal Djokovic или Nadal, Djokovic\n\n"
        "Или нажми /menu чтобы выбрать из списка."
    )


@bot.message_handler(commands=["menu"])
def choose_players_menu(message):
    user_id = message.from_user.id
    user_state.pop(user_id, None)
    bot.send_message(
        message.chat.id,
        "Выбери первого игрока:",
        reply_markup=make_matches_keyboard(),
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("match:"))
def on_match_click(call):
    data = call.data.split(":", 1)[1]
    p1, p2 = data.split("|")

    bot.answer_callback_query(call.id, "Считаем прогноз...")

    msg = bot.send_message(
        call.message.chat.id,
        f"{p1} vs {p2}\nЗапуск модели..."
    )

    try:
        odds_a, odds_b = get_pair_odds(p1, p2)

        if odds_a is None:
            odds_a = 2.0
        if odds_b is None:
            odds_b = 1.8

        prob_a = predict_match(p1, p2, odds_a, odds_b)

        text = (
            f"{p1} vs {p2}\n\n"
            f"P({p1}) = {prob_a:.3f}\n"
            f"Коэфы: {odds_a:.2f} / {odds_b:.2f}"
        )

    except Exception as e:
        text = f"Ошибка: {repr(e)}"

    bot.edit_message_text(
        chat_id=msg.chat.id,
        message_id=msg.message_id,
        text=text
    )


@bot.message_handler(content_types=["text"])
def get_players(message):
    text = message.text.strip()
    cleaned = text.replace(",", " ")
    parts = cleaned.split()
    if len(parts) != 2:
        bot.reply_to(
            message,
            "Нужно ввести ровно две фамилии на английском.\n"
            "Пример: Nadal Djokovic\n"
            "Или нажми /menu чтобы выбрать из списка."
        )
        return

    p1, p2 = parts[0], parts[1]
    bot.reply_to(message, f"Получена пара: {p1} vs {p2}")

    odds_a = 2.0
    odds_b = 1.8

    try:
        print("[LOG] Запуск predict_match...")
        prob_a = predict_match(p1, p2, odds_a, odds_b)
        print(
            f"[LOG] predict_match вернул: {prob_a:.4f}"
        )
        bot.reply_to(
            message,
            f"{p1} vs {p2}\n"
            f"Модель: P({p1} выигрывает) = {prob_a:.3f}\n"
            f"Коэффы: A={odds_a:.2f}, B={odds_b:.2f}"
        )
    except Exception as e:
        bot.reply_to(message, f"Ошибка при расчёте: {repr(e)}")
        bot.send_message(
            message.chat.id,
            f"[LOG] Exception в predict_match: {repr(e)}"
        )


if __name__ == "__main__":

    MATCH_ODDS = find_match_odds()

    updater_thread = threading.Thread(
        target=odds_updater,
        daemon=True
    )
    updater_thread.start()

    bot.infinity_polling()