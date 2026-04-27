from model import predict_match
from .keyboard import make_matches_keyboard


def format_decision_text(p1, p2, prob_a, odds_a, odds_b, decision):
    action = decision.get("action", "no_bet")
    stake = decision.get("stake", 0.0)
    edge = decision.get("edge", 0.0)
    kelly = decision.get("kelly", 0.0)
    reason = decision.get("reason", "")

    if action == "no_bet":
        bet_line = "Рекомендация: не ставить на этот матч. \n"
    elif action == "bet_a":
        bet_line = (
            f"Рекомендация: ставить на {p1}.\n"
            f"Рекомендуемый размер ставки: {stake:.2f} (в процентах из текущего банкролла).\n"
        )
    elif action == "bet_b":
        bet_line = (
            f"Рекомендация: ставить на {p2}.\n"
            f"Рекомендуемый размер ставки: {stake:.2f} (в процентах из текущего банкролла).\n"
        )
    else:
        bet_line = "Рекомендация: не ставить (неизвестное действие)."

    text = (
        f"{p1} vs {p2}\n\n"
        f"Модель: Вероятность того, что {p1} выигрывает = {prob_a:.3f}\n"
        f"Коэффы: A={odds_a:.2f}, B={odds_b:.2f}\n\n"
        f"{bet_line}"
        f"edge = {edge:.3f}, Kelly = {kelly:.3f}\n"
        f"{reason}"
    )
    return text


def register_handlers(telegram_bot, cache):
    @telegram_bot.message_handler(commands=["start", "help"])
    async def send_welcome(message):
        await telegram_bot.reply_to(
            message,
            "Отправь мне двух игроков через пробел или запятую,\n"
            "например: Nadal Djokovic или Nadal, Djokovic\n\n"
            "Или нажми /menu чтобы выбрать из списка.",
        )

    @telegram_bot.message_handler(commands=["menu"])
    async def choose_players_menu(message):
        await telegram_bot.send_message(
            message.chat.id,
            "Выбери матч:",
            reply_markup=await make_matches_keyboard(cache),
        )

    @telegram_bot.callback_query_handler(func=lambda call: call.data.startswith("match:"))
    async def on_match_click(call):
        _, payload = call.data.split(":", 1)
        p1, p2 = payload.split("|")

        await telegram_bot.answer_callback_query(call.id, "Считаем прогноз…")
        msg = await telegram_bot.send_message(
            call.message.chat.id, f"{p1} vs {p2}\nЗапуск модели…"
        )

        try:
            odds_a, odds_b = await cache.get_odds_for_match(p1, p2)
            odds_a = odds_a if odds_a is not None else 2.0
            odds_b = odds_b if odds_b is not None else 1.8

            prob_a, decision = predict_match(p1, p2, odds_a, odds_b, bankroll=100.0)

            text = format_decision_text(p1, p2, prob_a, odds_a, odds_b, decision)
        except Exception as exc:
            text = f"Ошибка: {repr(exc)}"

        await telegram_bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=msg.message_id,
            text=text,
        )

    @telegram_bot.message_handler(content_types=["text"])
    async def get_players(message):
        parts = message.text.strip().replace(",", " ").split()

        if len(parts) != 2:
            await telegram_bot.reply_to(
                message,
                "Нужно ввести ровно две фамилии на английском.\n"
                "Пример: Nadal Djokovic\n"
                "Или нажми /menu чтобы выбрать из списка.",
            )
            return

        p1, p2 = parts
        await telegram_bot.reply_to(message, f"Получена пара: {p1} vs {p2}")

        odds_a, odds_b = await cache.get_odds_for_match(p1, p2)
        odds_a = odds_a if odds_a is not None else 2.0
        odds_b = odds_b if odds_b is not None else 1.8

        try:
            prob_a, decision = predict_match(p1, p2, odds_a, odds_b, bankroll=100.0)

            text = format_decision_text(p1, p2, prob_a, odds_a, odds_b, decision)
            await telegram_bot.reply_to(message, text)
        except Exception as exc:
            await telegram_bot.reply_to(message, f"Ошибка: {repr(exc)}")
