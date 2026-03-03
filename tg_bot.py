from collections import defaultdict

import telebot
from telebot import types
from config.config import api_token

from model_to_predict import predict_match
from get_info import find_match_odds

from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


bot = telebot.TeleBot(api_token)

PLAYERS = [
    "Djokovic",
    "Nadal",
    "Federer",
    "Alcaraz",
    "Medvedev",
    "Sinner",
]

user_state = {}
MATCH_ODDS = defaultdict(dict)

def make_matches_keyboard():
    markup = types.InlineKeyboardMarkup()

    for (p1, p2), odds in MATCH_ODDS.items():
        odds_a, odds_b = odds

        text = f"{p1} vs {p2} ({odds_a:.2f} / {odds_b:.2f})"

        btn = types.InlineKeyboardButton(
            text=text,
            callback_data=f"match:{p1}|{p2}"
        )

        markup.add(btn)

    return markup

def get_pair_odds(p1, p2):
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
    bot.infinity_polling()