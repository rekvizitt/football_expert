import json
import os
from flask import Flask, request, render_template, redirect, url_for
from src.api import FootballExpertApi
from src.logger import logger

leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
seasons = ["2425"]
api = FootballExpertApi(leagues, seasons)

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", True))