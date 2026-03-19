from telebot import types


def make_matches_keyboard(cache) -> types.InlineKeyboardMarkup:
    markup = types.InlineKeyboardMarkup()

    for (p1, p2), (odds_a, odds_b) in cache.get_all_matches().items():
        btn = types.InlineKeyboardButton(
            text=f"{p1} vs {p2} ({odds_a:.2f} / {odds_b:.2f})",
            callback_data=f"match:{p1}|{p2}",
        )
        markup.add(btn)

    return markup