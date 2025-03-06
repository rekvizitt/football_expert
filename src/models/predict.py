import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import math
from src.config import ConfigManager
from src.logger import logger
from src.data.prepare_data import DataPrepare

class MatchPredictor:
    def __init__(self):
        config_manager = ConfigManager()
        self.models_dir = config_manager.models_dir
            
        self.models = {}
        self.load_models()

    def load_models(self):
            """Загрузка всех доступных моделей."""
            model_paths = {
                'logistic_regression': self.models_dir / 'logistic_regression.pkl',
                'random_forest': self.models_dir / 'random_forest.pkl',
                'gradient_boosting': self.models_dir / 'gradient_boosting.pkl',
                'xgboost': self.models_dir / 'xgboost.pkl',
            }
            for model_name, model_path in model_paths.items():
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    logger.info(f'Модель {model_name} загружена.')
                except Exception as e:
                    logger.warning(f'Ошибка загрузки модели {model_name}: {e}')

    def predict(self, match_data: dict) -> dict:
        """
        Предсказать результат матча.
        :param match_data: Словарь с данными о матче.
        :return: Результаты предсказаний для всех моделей.
        """
        if not self.models:
            raise ValueError("Нет загруженных моделей.")
        predictions = {}
        prediction_probas = {}
        for model_name, model in self.models.items():
            match_data = match_data[model.feature_names_in_]
            prediction = model.predict(match_data)
            prediction_proba = model.predict_proba(match_data)
            predictions[model_name] = prediction[0]
            prediction_probas[model_name] = prediction_proba
        return {
            "predictions": predictions,
            "prediction_probas": prediction_probas
        }
    
    def interpret_prediction(self, prediction: int, home_team: str, away_team: str) -> str:
        """
        Интерпретация предсказания.
        :param prediction: Числовое значение предсказания.
        :param home_team: Название домашней команды.
        :param away_team: Название гостевой команды.
        :return: Интерпретированный результат.
        """
        if prediction == 0:
            return f"Победа {home_team}"
        elif prediction == 1:
            return f"Ничья между {home_team} и {away_team}"
        elif prediction == 2:
            return f"Победа {away_team}"
        else:
            return "Неизвестный результат"
    
    def determine_winner_by_probabilities(self, probabilities: list) -> str:
        """
        Определение победителя матча на основе вероятностей.
        :param probabilities: Список вероятностей [P_home_win, P_draw, P_away_win].
        :return: Результат матча ("Победа хозяев", "Ничья", "Победа гостей").
        """
        p_home, p_draw, p_away = probabilities

        # Правило 1: Если вероятность победы одной из команд превышает 70%
        if p_home >= 0.7:
            return "Победа хозяев"
        elif p_away >= 0.7:
            return "Победа гостей"

        # Правило 2: Если вероятность ничьи превышает 40%
        if p_draw > 0.4:
            return "Ничья"

        # Правило 3: Если вероятность победы Home Team превышает вероятность ничьи и Away Team минимум на 15%
        if p_home - p_draw >= 0.15 and p_home - p_away >= 0.15:
            return "Победа хозяев"

        # Правило 4: Если вероятность победы Away Team превышает вероятность ничьи и Home Team минимум на 15%
        if p_away - p_draw >= 0.15 and p_away - p_home >= 0.15:
            return "Победа гостей"

        # Правило 5: Если разница между вероятностью победы Home Team и Away Team составляет менее 10%,
        # но вероятность ничьи меньше 30%, выбирается команда с большей вероятностью победы
        if abs(p_home - p_away) < 0.1 and p_draw < 0.3:
            return "Победа хозяев" if p_home > p_away else "Победа гостей"

        # По умолчанию: Ничья
        return "Ничья"
            
    def predict_and_determine_winner(self, match_data: dict) -> dict:
        """
        Предсказать результат матча и определить победителя на основе вероятностей.
        :param match_data: Словарь с данными о матче.
        :return: Результаты предсказаний и победитель.
        """
        results = self.predict(match_data)
        combined_prediction_proba = np.mean(list(results["prediction_probas"].values()), axis=0)[0]

        # Получаем вероятности для каждой категории
        p_home, p_draw, p_away = combined_prediction_proba

        # Определяем победителя
        winner = self.determine_winner_by_probabilities([p_home, p_draw, p_away])

        return {
            "predictions": results["predictions"],
            "combined_probabilities": {
                "home_win": p_home,
                "draw": p_draw,
                "away_win": p_away
            },
            "winner": winner
        }
            
if __name__ == "__main__":
    predictor = MatchPredictor()
    dp = DataPrepare()   
    # get match_data
    home_team = "Manchester United"
    away_team = "Liverpool"
    date = datetime(2025, 3, 1)
    match_data = dp.fetch_match_data(home_team, away_team, date)
    
    results = predictor.predict_and_determine_winner(match_data)
    print("\n=== Результаты предсказаний ===")
    for model_name, prediction in results["predictions"].items():
        print(f"{model_name}: {predictor.interpret_prediction(prediction, home_team, away_team)}")

    print("\n=== Средние вероятности ===")
    print(f"Победа {home_team}: {results['combined_probabilities']['home_win']:.2%}")
    print(f"Ничья: {results['combined_probabilities']['draw']:.2%}")
    print(f"Победа {away_team}: {results['combined_probabilities']['away_win']:.2%}")

    print("\n=== Победитель матча ===")
    print(f"Результат: {results['winner']}")