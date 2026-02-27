import telebot
from telebot import types
from config.config import api_token
from sklearn.preprocessing import StandardScaler
from model_to_predict import predict_match
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

def make_players_keyboard(exclude=None):
    markup = types.InlineKeyboardMarkup()
    exclude = exclude or []
    row = []
    for name in PLAYERS:
        if name in exclude:
            continue
        btn = types.InlineKeyboardButton(text=name, callback_data=f"player:{name}")
        row.append(btn)
        if len(row) == 2:
            markup.row(*row)
            row = []
    if row:
        markup.row(*row)
    return markup


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
        reply_markup=make_players_keyboard(),
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("player:"))
def on_player_click(call):
    user_id = call.from_user.id
    name = call.data.split(":", 1)[1]
    state = user_state.get(user_id, {})

    if "p1" not in state:
        state["p1"] = name
        user_state[user_id] = state

        bot.answer_callback_query(call.id)
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"Первый игрок: {name}\nТеперь выбери второго:",
            reply_markup=make_players_keyboard(exclude=[name]),
        )
    else:
        state["p2"] = name
        user_state[user_id] = state
        p1, p2 = state["p1"], state["p2"]

        bot.answer_callback_query(call.id, "Запрос принят, считаем прогноз...")

        msg = bot.send_message(
            call.message.chat.id,
            f"Получена пара: {p1} vs {p2}\nНачинаем расчёт..."
        )

        try:
            odds_a, odds_b = 2.0, 1.8
            print("[LOG] Запуск predict_match...")

            prob_a = predict_match(p1, p2, odds_a, odds_b)

            print(f"[LOG] predict_match вернул: {prob_a:.4f}")

            text = (
                f"Пара: {p1} vs {p2}\n"
                f"Модель: P({p1} выигрывает) = {prob_a:.3f}\n"
                f"Кэфы: A={odds_a:.2f}, B={odds_b:.2f}"
            )

            print(text)
        except Exception as e:
            text = f"Ошибка при расчёте: {repr(e)}"
            bot.send_message(
                call.message.chat.id,
                f"[LOG] Exception в predict_match: {repr(e)}"
            )
        print('BOT TEXT')
        bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=msg.message_id,
            text=text,
        )

        user_state.pop(user_id, None)


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


if __name__ == "main":
    print("Bot started...")
    bot.infinity_polling()