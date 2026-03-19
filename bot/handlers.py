from model import predict_match
from .keyboard import make_matches_keyboard


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
        msg = await telegram_bot.send_message(call.message.chat.id, f"{p1} vs {p2}\nЗапуск модели…")

        try:
            odds_a, odds_b = await cache.get_odds_for_match(p1, p2)
            odds_a = odds_a if odds_a is not None else 2.0
            odds_b = odds_b if odds_b is not None else 1.8

            prob_a = predict_match(p1, p2, odds_a, odds_b)

            text = (
                f"{p1} vs {p2}\n\n"
                f"P({p1}) = {prob_a:.3f}\n"
                f"Коэффы: {odds_a:.2f} / {odds_b:.2f}"
            )
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

        odds_a, odds_b = cache.get_odds_for_match(p1, p2)
        odds_a = odds_a if odds_a is not None else 2.0
        odds_b = odds_b if odds_b is not None else 1.8

        try:
            prob_a = predict_match(p1, p2, odds_a, odds_b)

            await telegram_bot.reply_to(
                message,
                f"{p1} vs {p2}\n"
                f"Модель: P({p1} выигрывает) = {prob_a:.3f}\n"
                f"Коэффы: A={odds_a:.2f}, B={odds_b:.2f}",
            )
        except Exception as exc:
            await telegram_bot.reply_to(message, f"Ошибка: {repr(exc)}")