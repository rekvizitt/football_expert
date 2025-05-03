import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import datetime
from src.api import FootballExpertApi
from src.logger import logger

leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
seasons = ["2425"]
api = FootballExpertApi(leagues, seasons)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я могу предсказать результаты футбольных матчей."
        "\n\nДоступные команды:"
        "\n<b>/start</b> - Начать работу"
        "\n<b>/leagues</b> - Получить список лиг"
        "\n<b>/matches</b> - Получить список предстоящих матчей"
        "\n<b>/predict</b> команда_дома команда_гостей - Предсказать результат матча"
        "\n<b>/metrics</b> - Отобразить метрики моделей"
    )

async def matches_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет список предстоящих матчей."""
    
    # Проверяем наличие кэша предстоящих матчей
    matches = api.upcoming_matches
    if matches is None or matches.empty:
        await update.message.reply_text("На этой неделе нет предстоящих матчей.")
        return

    # Группируем матчи по лигам
    leagues = matches['league'].unique()
    keyboard = []

    for league in leagues:
        league_matches = matches[matches['league'] == league]
        keyboard.append([InlineKeyboardButton(f"⭐ {league}", callback_data="noop")])  # Разделитель для лиги

        for _, match in league_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            date_time = f"{match['date'].date()} {match['time']}"
            button_text = f"{home_team} vs {away_team} ({date_time})"
            callback_data = f"predict_{home_team}_{away_team}_{match['date'].date()}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите матч для предсказания:", reply_markup=reply_markup)


async def leagues_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет меню выбора лиги."""
    leagues = api.leagues
    page_size = len(leagues)
    pages = [leagues[i:i + page_size] for i in range(0, len(leagues), page_size)]

    current_page = 0
    keyboard = []
    for league in pages[current_page]:
        keyboard.append([InlineKeyboardButton(league, callback_data=f"league_{league}")])

    if len(pages) > 1:
        if current_page > 0:
            keyboard.append([InlineKeyboardButton("Предыдущая страница", callback_data="prev_page")])
        if current_page < len(pages) - 1:
            keyboard.append([InlineKeyboardButton("Следующая страница", callback_data="next_page")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите лигу:", reply_markup=reply_markup)


async def generate_match_prediction_response(home_team_name, away_team_name, date=None):
    """Формирует текст ответа для предсказания результата матча с HTML-форматированием."""
    if date == None:
        date = datetime.date.today()
    
    # Fetch match data
    match_data, encoded_match_data = api.get_match_data(home_team_name, away_team_name, date)
    if match_data is None:
        return f"<b>❌ Ошибка:</b> Не удалось сгенерировать данные для матча {home_team_name} vs {away_team_name}."

    # Predict match results
    results = api.predictor.predict_and_determine_winner(encoded_match_data)
    combined_prediction_proba = results["combined_probabilities"]
    winner = results["winner"]

    # Calculate Poisson probabilities for exact score
    home_team_avg_goals = match_data['home_xg_last_5'].iloc[0]  
    away_team_avg_goals = match_data['away_xg_last_5'].iloc[0]
    poisson_probabilities = api.predictor.poisson_distribution(home_team_avg_goals, away_team_avg_goals)

    # Format the response
    # Эмодзи для разных секций и результатов
    team_emoji = "⚽️"
    vs_emoji = "🆚"
    prediction_emoji = "🔮"
    stats_emoji = "📊"
    winner_emoji = "🏆"
    score_emoji = "🎯"
    
    # Определяем эмодзи для результатов
    result_emoji = {
        "home_win": "🏠",
        "draw": "🤝",
        "away_win": "✈️"
    }
    
    # Заголовок сообщения
    response = (
        f"{prediction_emoji} <b>ПРЕДСКАЗАНИЕ МАТЧА</b> {prediction_emoji}\n"
        f"{team_emoji} <b>{home_team_name}</b> {vs_emoji} <b>{away_team_name}</b>\n\n"
    )
    
    # Прогнозы моделей
    response += f"<b>🤖 ПРОГНОЗЫ МОДЕЛЕЙ:</b>\n"
    for model_name, prediction in results["predictions"].items():
        interpretation = api.predictor.interpret_prediction(prediction, home_team_name, away_team_name)
        response += f"• <i>{model_name}</i>: {interpretation}\n"
    
    # Средние вероятности
    response += f"\n<b>{stats_emoji} ВЕРОЯТНОСТИ ИСХОДОВ:</b>\n"
    response += f"{result_emoji['home_win']} Победа <b>{home_team_name}</b>: <code>{combined_prediction_proba['home_win']:.1%}</code>\n"
    response += f"{result_emoji['draw']} Ничья: <code>{combined_prediction_proba['draw']:.1%}</code>\n"
    response += f"{result_emoji['away_win']} Победа <b>{away_team_name}</b>: <code>{combined_prediction_proba['away_win']:.1%}</code>\n"
    
    # Победитель матча
    response += f"\n<b>{winner_emoji} НАИБОЛЕЕ ВЕРОЯТНЫЙ ИСХОД:</b>\n"
    response += f"<b>{winner}</b>\n"
    
    # Вероятности точного счета
    response += f"\n<b>{score_emoji} ВЕРОЯТНЫЙ СЧЕТ (распределение Пуассона):</b>\n"
    
    # Отображаем топ-5 наиболее вероятных счетов
    sorted_probabilities = sorted(poisson_probabilities.items(), key=lambda x: x[1], reverse=True)
    for i, (score, probability) in enumerate(sorted_probabilities[:5]):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
        response += f"{medal} <b>{score[0]}:{score[1]}</b> — <code>{probability*100:.1f}%</code>\n"
    
    # Добавляем информационную строку в конце
    response += "\n<i>Прогноз основан на статистических моделях и имеет информационный характер</i>"
    
    return response

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Предсказывает результат матча по названиям команд."""
    # Проверяем, что пользователь ввел ровно два аргумента
    args = context.args
    if len(args) != 2:
        await update.message.reply_text(
            "❌ Ошибка: Неверное количество аргументов.\n"
            "Использование: /predict <команда_дома> <команда_гостей>"
        )
        return

    home_team_name, away_team_name = args[0], args[1]

    # Проверяем, что обе команды указаны и не пустые
    if not home_team_name.strip() or not away_team_name.strip():
        await update.message.reply_text("❌ Ошибка: Одна или обе команды не указаны.")
        return

    # Проверяем, существуют ли такие команды в базе данных
    home_team_exists = api.find_team_name(home_team_name)
    away_team_exists = api.find_team_name(away_team_name)

    if not home_team_exists:
        await update.message.reply_text(f"❌ Ошибка: Команда дома '{home_team_name}' не найдена.")
        return

    if not away_team_exists:
        await update.message.reply_text(f"❌ Ошибка: Команда гостей '{away_team_name}' не найдена.")
        return

    # Генерируем ответ
    response = await generate_match_prediction_response(home_team_exists, away_team_exists)

    # Если возникла ошибка при генерации предсказания
    if "Ошибка" in response:
        await update.message.reply_html(response)
        return

    # Отправляем результат предсказания
    await update.message.reply_html(f"{response}", disable_web_page_preview=True)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик кнопок для выбора матча или лиги."""
    query = update.callback_query
    await query.answer()

    # Разбираем данные из callback_data
    data_parts = query.data.split("_")

    if data_parts[0] == "predict":
        if len(data_parts) != 4:
            await query.edit_message_text(text="Ошибка: Некорректные данные для предсказания.")
            return

        _, home_team, away_team, date = data_parts

        # Генерируем ответ
        response = await generate_match_prediction_response(home_team, away_team, date)

        if "Не удалось сгенерировать данные" in response:
            await query.edit_message_text(text=response)
            return

        await query.edit_message_text(text=response, parse_mode="HTML")

    elif data_parts[0] == "league":  # Если выбрана лига
        await league_handler(update, context)  # Передаем обработку в league_handler

    else:
        await query.edit_message_text(text="Неизвестный запрос.")

async def league_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик выбора лиги."""
    query = update.callback_query
    await query.answer()

    league = query.data[7:]  # Удаляем "league_" из callback_data

    logger.info(f"Selected league: {league}")

    # Получаем матчи для выбранной лиги
    matches = api.upcoming_matches[api.upcoming_matches['league'] == league]
    if matches is None or matches.empty:
        await query.edit_message_text(text=f"На этой неделе нет предстоящих матчей в лиге {league}.")
        return

    # Создаем клавиатуру с кнопками для каждого матча
    keyboard = []
    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        date_time = f"{match['date'].date()} {match['time']}"
        button_text = f"{home_team} vs {away_team} ({date_time})"
        callback_data = f"predict_{home_team}_{away_team}_{match['date'].date()}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Отправляем сообщение с клавиатурой
    await query.edit_message_text(
        text=f"📋 Расписание матчей для лиги <b>{league}</b>:\n\nВыберите матч для предсказания:",
        parse_mode="HTML",
        reply_markup=reply_markup
    )

async def metrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отображает метрики моделей с HTML-форматированием и их объяснением."""
    metrics_files = {
        "Gradient Boosting": "gradient_boosting_metrics.json",
        "Logistic Regression": "logistic_regression_metrics.json",
        "Random Forest": "random_forest_metrics.json",
        "XGBoost": "xgboost_metrics.json",
    }

   # Формируем ответ с HTML-форматированием
    response = (
        "<b>📊 МЕТРИКИ МОДЕЛЕЙ</b>\n"
        "══════════════════════\n\n"
    )

    for model, file in metrics_files.items():
        try:
            with open(f"{api.metrics_dir}/{file}", "r") as f:
                metrics = json.load(f)
                response += (
                    f"<b>🔍 {model}</b>\n"
                    f"  ✅ Точность: <code>{metrics['accuracy']:.4f}</code>\n"
                    f"  ⭐ F1-мера: <code>{metrics['f1']:.4f}</code>\n"
                    f"  📈 ROC-AUC: <code>{metrics['roc_auc']:.4f}</code>\n"
                    f"  ────────────────\n\n"
                )
        except FileNotFoundError:
            response += f"❌ <b>{model}</b>: Метрики не найдены\n\n"

    # Добавляем общее объяснение метрик в конце
    response += (
        "<b>💡 ПОЯСНЕНИЕ МЕТРИК</b>\n"
        "══════════════════════\n"
        "✅ <b>Точность</b>: доля правильных предсказаний модели (0,4-0,5: это минимальный уровень точности, который можно считать удовлетворительным.)\n\n"
        "⭐ <b>F1-мера</b>: баланс между точностью и полнотой (0,4-0,5: это минимальный уровень F1-меры, который можно считать удовлетворительным.)\n\n"
        "📈 <b>ROC-AUC</b>: способность модели различать классы (чем ближе к 1, тем лучше)\n"
    )
    # Отправляем сообщение с HTML-форматированием
    await update.message.reply_html(response)

async def help_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Отображает справку."""
    await update.message.reply_text(
        "Этот бот предсказывает результаты футбольных матчей.\n\n"
        "/start - Начать работу\n"
        "/leagues - Получить список лиг\n"
        "/matches - Получить список предстоящих матчей\n"
        "/predict <команда_дома> <команда_гостей> - Предсказать результат матча\n"
        "/metrics - Отобразить метрики моделей"
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Проверяем, существует ли объект message
    if update and hasattr(update, "message") and update.message:
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте снова позже.")
    elif update and hasattr(update, "callback_query") and update.callback_query:
        await update.callback_query.answer("Произошла ошибка. Пожалуйста, попробуйте снова позже.")
    else:
        logger.warning("Не удалось отправить сообщение об ошибке пользователю.")

def main() -> None:
    """Запуск бота."""
    # Инициализация бота
    token = "TOKEN"
    application = Application.builder().token(token).build()
    
    # Добавление обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("leagues", leagues_command))
    application.add_handler(CommandHandler("matches", matches_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("metrics", metrics_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(CallbackQueryHandler(league_handler, pattern="^league_"))

    # Добавление обработчика ошибок
    application.add_error_handler(error_handler)

    # Запуск бота
    logger.info("Starting the bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()