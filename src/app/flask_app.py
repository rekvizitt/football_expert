import json
import os
from flask import Flask, request, render_template, redirect, url_for
from src.api import FootballExpertApi
from src.logger import logger
from src.database.db_manager import DataBaseManager

leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
seasons = ["2425"]
api = FootballExpertApi(leagues, seasons)

app = Flask(__name__)
database = DataBaseManager()

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Страница лиг
@app.route('/leagues', methods=['GET'])
def leagues_page():
    leagues = api.leagues
    return render_template('leagues.html', leagues=leagues)

# Страница матчей выбранной лиги
@app.route('/leagues/<league>', methods=['GET'])
def league_matches_page(league):
    matches = api.upcoming_matches[api.upcoming_matches['league'] == league]
    if matches is None or matches.empty:
        return render_template('error.html', message="Нет предстоящих матчей"), 404
    matches_copy = matches.copy()
    matches_copy['date'] = matches_copy['date'].dt.strftime('%Y-%m-%d')
    matches_data = matches_copy.to_dict(orient='records')
    return render_template('league_matches.html', league=league, matches=matches_data)

# Страница со всеми матчами
@app.route('/matches', methods=['GET'])
def matches_page():
    matches = api.upcoming_matches
    if matches is None or matches.empty:
        return render_template('error.html', message="Нет предстоящих матчей"), 404
    matches_copy = matches.copy()
    matches_copy['date'] = matches_copy['date'].dt.strftime('%Y-%m-%d')
    matches_data = matches_copy.to_dict(orient='records')
    return render_template('matches.html', matches=matches_data)

# Страница для предсказания результата любого матча
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        return redirect(url_for('predict_match_page', home_team=home_team, away_team=away_team))
    leagues = api.leagues
    teams = api.create_team_league_dict()
    return render_template('predict.html', leagues=leagues, teams=teams)

# Предсказание матча
@app.route('/predict_match/<home_team>/<away_team>', methods=['GET'])
def predict_match_page(home_team, away_team):
    
    if not home_team:
        return render_template('error.html', message=f"Команда дома '{home_team}' не найдена"), 400
    if not away_team:
        return render_template('error.html', message=f"Команда гостей '{away_team}' не найдена"), 400
    try:
        # Генерируем данные матча
        match_date = api.get_match_date_or_today(home_team, away_team)
        match_data, encoded_match_data = api.get_match_data(home_team, away_team, match_date)
        if match_data is None:
            return render_template('error.html', message=f"Не удалось сгенерировать данные для матча {home_team} vs {away_team}"), 400

        # Predict match results
        results = api.predictor.predict_and_determine_winner(encoded_match_data)
        combined_prediction_proba = results["combined_probabilities"]
        winner = results["winner"]

        # Calculate Poisson probabilities for exact score
        home_team_avg_goals = match_data['home_xg_last_5'].iloc[0]  
        away_team_avg_goals = match_data['away_xg_last_5'].iloc[0]
        poisson_probabilities = api.predictor.poisson_distribution(home_team_avg_goals, away_team_avg_goals)

        top_scores = sorted(poisson_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

        # Формируем ответ
        response = {
            "home_team": home_team,
            "away_team": away_team,
            "predictions": {
                "home_win": float(combined_prediction_proba['home_win']),
                "draw": float(combined_prediction_proba['draw']),
                "away_win": float(combined_prediction_proba['away_win']),
            },
            "winner": winner,  # Добавляем определенного победителя
            "score_probabilities": dict(top_scores),       
        }
        # Сохраняем предсказание в базу данных
        db_result = database.add_match_prediction(home_team, away_team, response)
        if not db_result['success']:
            logger.warning(f"Failed to save prediction to DB: {db_result.get('error')}")
        return render_template('predict_match.html', prediction=response)
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        return render_template('error.html', message="Ошибка при предсказании матча"), 500

# Страница с метриками моделей
@app.route('/metrics', methods=['GET'])
def metrics_page():
    metrics_files = {
        "Gradient Boosting": "gradient_boosting_metrics.json",
        "Logistic Regression": "logistic_regression_metrics.json",
        "Random Forest": "random_forest_metrics.json",
        "XGBoost": "xgboost_metrics.json",
    }
    metrics_data = {}
    for model, file in metrics_files.items():
        try:
            with open(f"{api.metrics_dir}/{file}", "r") as f:
                metrics = json.load(f)
                metrics_data[model] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_score": metrics.get("f1", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                }
        except FileNotFoundError:
            metrics_data[model] = {"error": "Metrics not found"}
        except Exception as e:
            logger.error(f"Error loading metrics for {model}: {e}")
            metrics_data[model] = {"error": "Error loading metrics"}
    return render_template('metrics.html', metrics=metrics_data)

# Страница справки
@app.route('/help', methods=['GET'])
def help_page():
    return render_template('help.html')

@app.route('/database', methods=['GET', 'POST'])
def database_page():
    if request.method == 'POST':
        try:
            action = request.form.get('action')
            if action == 'fill_default':
                result = database.fill_database(custom_data=False)
                if result['success']:
                    logger.info("Database filled with default data")
                else:
                    return render_template('error.html', message=f"Ошибка при заполнении базы данных: {result.get('error', 'Неизвестная ошибка')}")
            elif action == 'fill_custom':
                return redirect(url_for('fill_database_page'))
            elif action == 'backup':
                result = database.backup_database()
                if result['success']:
                    logger.info("Database backuped")
                else:
                    return render_template('error.html', message=f"Ошибка при создании резервной копии: {result.get('error', 'Неизвестная ошибка')}")
            elif action == 'optimize':
                result = database.optimize_database()
                if result['success']:
                    logger.info("Database optimized")
                else:
                    return render_template('error.html', message=f"Ошибка при оптимизации базы данных: {result.get('error', 'Неизвестная ошибка')}")
            elif action == 'view_table':
                table_name = request.form.get('table_name')
                if not table_name:
                    return render_template('error.html', message="Не указано имя таблицы")
                return redirect(url_for('view_table_page', table=table_name))
            else:
                return render_template('error.html', message="Неизвестное действие")
                
        except Exception as e:
            logger.error(f"Error processing database action: {e}")
            
        return redirect(url_for('database_page'))
        
    # Получаем информацию о базе данных    
    db_size = os.path.getsize(database.db_path) if os.path.exists(database.db_path) else 0
    db_tables = database.get_total_tables_count()
    db_records = database.get_total_records_count()
    db_tables_info = database.get_tables_info()
    
    return render_template('database.html',
                          db_size=db_size,
                          db_path=database.db_path,
                          db_tables=db_tables,
                          db_records=db_records,
                          db_tables_info=db_tables_info)

@app.route('/table_view/<table>')
def view_table_page(table):
    # table_name = request.form.get('table_name')
    page = int(request.form.get('page', 1))
    per_page = 100
    sort_column = request.args.get('sort')
    sort_order = request.args.get('order', 'asc')
    filter_column = request.args.get('filter_col')
    filter_value = request.args.get('filter_val')
    filter_strict = request.args.get('filter_strict') == 'on'
    
    result = database.get_table_data(
        table_name=table,
        page=page,
        per_page=per_page,
        sort_column=sort_column,
        sort_order=sort_order,
        filter_column=filter_column,
        filter_value=filter_value,
        filter_strict=filter_strict
    )
    
    if not result['success']:
         return render_template('error.html', message=f"Ошибка при получении данных: {result.get('error')}")
    
    total_records = result['total_records']
    total_pages = (total_records + per_page - 1) // per_page
    
    return render_template('table_view.html',
                           table_name=table,
                           columns=result['columns'],
                           data=result['data'],
                           total_records=total_records,
                           current_page=page,
                           total_pages=total_pages,
                           sort_column=sort_column,
                           sort_order=sort_order,
                           filter_column=filter_column,
                           filter_value=filter_value,
                           filter_strict=filter_strict)

@app.route('/fill_database', methods=['GET', 'POST'])
def fill_database_page():
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            data_type = request.form.get('data_type')
            data_content = request.form.get('data_content')

            # Здесь должна быть логика обработки пользовательских данных
            logger.debug(f"{data_type}: {data_content}")
            database.fill_database(custom_data=[data_type, data_content])
            
            # Временная заглушка для примера
            result = {
                'success': True,
                'message': 'База данных успешно заполнена пользовательскими данными'
            }
            
            if result['success']:
                return redirect(url_for('database_page'))
            else:
                return render_template('error.html', message=result.get('error', 'Неизвестная ошибка'))
                
        except Exception as e:
            logger.error(f"Error filling database with custom data: {e}")
            return render_template('error.html', message=f"Ошибка при обработке данных: {str(e)}")
    
    return render_template('fill_database.html')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering_page():
    try:
        clustered_teams = database.create_clusters()
        return render_template("clustering.html", clustered_teams=clustered_teams)
    except Exception as e:
        logger.error(f"Ошибка при кластеризации: {e}")
        return render_template("error.html", message=f"Не удалось выполнить кластеризацию: {str(e)}")

if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", True))