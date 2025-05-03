import pandas as pd
import numpy as np
import joblib
import scipy.stats as stats
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import ConfigManager
from src.logger import logger
from src.data.prepare_data import DataPrepare
from src.data.utils import find_team_name

class MatchPredictor:
    def __init__(self):
        config_manager = ConfigManager()
        self.models_dir = config_manager.models_dir
        self.models = {}
        self.load_models()

    def load_models(self):
            """Загрузка всех доступных моделей."""
            model_paths = {
                # TODO: random forest best
                # 'logistic_regression': self.models_dir / 'logistic_regression.pkl',
                'random_forest': self.models_dir / 'random_forest.pkl',
                # 'gradient_boosting': self.models_dir / 'gradient_boosting.pkl',
                # 'xgboost': self.models_dir / 'xgboost.pkl',
            }
            for model_name, model_path in model_paths.items():
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    logger.info(f'Модель {model_name} загружена.')
                except Exception as e:
                    logger.warning(f'Ошибка загрузки модели {model_name}: {e}')

    def predict(self, encoded_match_data: dict) -> dict:
        """
        Предсказать результат матча.
        :param encoded_match_data: Словарь с данными о матче.
        :return: Результаты предсказаний для всех моделей.
        """
        if not self.models:
            raise ValueError("Нет загруженных моделей.")
        predictions = {}
        prediction_probas = {}
        for model_name, model in self.models.items():
            match_data = encoded_match_data[model.feature_names_in_]
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
            
    def predict_and_determine_winner(self, encoded_match_data: dict) -> dict:
        """
        Предсказать результат матча и определить победителя на основе вероятностей.
        :param encoded_match_data: Словарь с данными о матче.
        :return: Результаты предсказаний и победитель.
        """
        results = self.predict(encoded_match_data)
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
        
    def poisson_distribution(self, home_team_avg_goals, away_team_avg_goals):
        """
        Распределение Пуассона для расчета вероятностей точного счета.
        :param home_team_avg_goals: Среднее количество голов, забитых домашней командой.
        :param away_team_avg_goals: Среднее количество голов, забитых гостевой командой.
        :return: Словарь с вероятностями точного счета.
        """
        poisson_home = stats.poisson(home_team_avg_goals)
        poisson_away = stats.poisson(away_team_avg_goals)

        score_probabilities = {}
        for home_goals in range(6):  # рассматриваем до 5 голов
            for away_goals in range(6):
                probability = poisson_home.pmf(home_goals) * poisson_away.pmf(away_goals)
                score_probabilities[(home_goals, away_goals)] = probability

        return score_probabilities
    
    # def calculate_score_probabilities(self, home_team_avg_goals, away_team_avg_goals):
    #     """
    #     Расчет вероятностей точного счета.
    #     :param home_team_avg_goals: Среднее количество голов, забитых домашней командой.
    #     :param away_team_avg_goals: Среднее количество голов, забитых гостевой командой.
    #     :return: Словарь с вероятностями точного счета.
    #     """
    #     score_probabilities = {}
    #     for home_goals in range(6):  # рассматриваем до 5 голов
    #         for away_goals in range(6):
    #             probability = np.random.poisson(home_team_avg_goals) / (home_team_avg_goals ** home_goals) * np.exp(-home_team_avg_goals) * \
    #                         np.random.poisson(away_team_avg_goals) / (away_team_avg_goals ** away_goals) * np.exp(-away_team_avg_goals)
    #             score_probabilities[(home_goals, away_goals)] = probability

    #     return score_probabilities
    
    # def skellam_distribution(self, home_team_avg_goals, away_team_avg_goals):
    #     """
    #     Распределение Скетла для расчета вероятностей точного счета.
    #     :param home_team_avg_goals: Среднее количество голов, забитых домашней командой.
    #     :param away_team_avg_goals: Среднее количество голов, забитых гостевой командой.
    #     :return: Словарь с вероятностями точного счета.
    #     """
    #     # Распределение Скетла можно аппроксимировать как разность двух независимых распределений Пуассона
    #     score_probabilities = {}
    #     for home_goals in range(6):  # рассматриваем до 5 голов
    #         for away_goals in range(6):
    #             # Используем отрицательное биномиальное распределение для аппроксимации
    #             home_prob = stats.nbinom.pmf(home_goals, home_team_avg_goals, 0.5)
    #             away_prob = stats.nbinom.pmf(away_goals, away_team_avg_goals, 0.5)
    #             probability = home_prob * away_prob
    #             score_probabilities[(home_goals, away_goals)] = probability

    #     return score_probabilities
    
    # def monte_carlo_simulation(self, home_team_avg_goals, away_team_avg_goals, num_simulations=10000):
    #     """
    #     Метод Монте-Карло для симуляции матча.
    #     :param home_team_avg_goals: Среднее количество голов, забитых домашней командой.
    #     :param away_team_avg_goals: Среднее количество голов, забитых гостевой командой.
    #     :param num_simulations: Количество симуляций.
    #     :return: Словарь с вероятностями победы каждой команды.
    #     """
    #     home_wins = 0
    #     away_wins = 0
    #     draws = 0

    #     for _ in range(num_simulations):
    #         home_goals = np.random.poisson(home_team_avg_goals)
    #         away_goals = np.random.poisson(away_team_avg_goals)

    #         if home_goals > away_goals:
    #             home_wins += 1
    #         elif home_goals < away_goals:
    #             away_wins += 1
    #         else:
    #             draws += 1

    #     home_win_probability = home_wins / num_simulations
    #     away_win_probability = away_wins / num_simulations
    #     draw_probability = draws / num_simulations

    #     return {
    #         "home_win": home_win_probability,
    #         "away_win": away_win_probability,
    #         "draw": draw_probability
    #     }
            
    def print_feature_importances(predictor, encoded_data):
        """Выводит важность признаков для всех загруженных моделей."""
        logger.debug("\nАнализ важности признаков:")
        
        for model_name, model in predictor.models.items():
            try:
                # Проверяем, что модель поддерживает feature_importances_
                if hasattr(model, 'feature_importances_'):
                    # Проверяем соответствие количества признаков
                    if len(model.feature_importances_) == len(encoded_data.columns):
                        importances = model.feature_importances_
                        features = encoded_data.columns
                        importance_df = pd.DataFrame({'feature': features, 'importance': importances})
                        logger.debug(f"\n{model_name} (feature_importances_):")
                        logger.debug(importance_df.sort_values('importance', ascending=False).head(20))
                    else:
                        logger.debug(f"\n{model_name}: Количество признаков в модели ({len(model.feature_importances_)}) "
                            f"не совпадает с данными ({len(encoded_data.columns)})")
                
                # Для логистической регрессии
                elif hasattr(model, 'coef_'):
                    if len(model.coef_[0]) == len(encoded_data.columns):
                        coef = model.coef_[0]
                        features = encoded_data.columns
                        coef_df = pd.DataFrame({'feature': features, 'coef': coef})
                        logger.debug(f"\n{model_name} (coefficients):")
                        logger.debug(coef_df.sort_values('coef', ascending=False).head(20))
                    else:
                        logger.debug(f"\n{model_name}: Количество коэффициентов ({len(model.coef_[0])}) "
                            f"не совпадает с признаками ({len(encoded_data.columns)})")
                        
            except Exception as e:
                logger.error(f"\n{model_name}: Ошибка при анализе важности признаков - {str(e)}")
                
    # def compare_features(self, encoded_data):
    #     """Сравнивает признаки модели с текущими данными."""
    #     if not hasattr(self.models['logistic_regression'], 'feature_names_in_'):
    #         logger.error("Модели не содержат информации о признаках (обучены старой версией sklearn?)")
    #         return
        
    #     model_features = set(self.models['logistic_regression'].feature_names_in_)
    #     data_features = set(encoded_data.columns)
        
    #     logger.info(f"Признаков в модели: {len(model_features)}")
    #     logger.info(f"Признаков в данных: {len(data_features)}")
        
    #     extra_in_data = data_features - model_features
    #     missing_in_data = model_features - data_features
        
    #     if extra_in_data:
    #         logger.warning(f"Лишние признаки в данных: {extra_in_data}")
    #     if missing_in_data:
    #         logger.warning(f"Отсутствующие признаки в данных: {missing_in_data}")
            
if __name__ == "__main__":
    predictor = MatchPredictor()
    dp = DataPrepare()   
    # get match_data
    home_team = find_team_name("Lille")
    away_team = find_team_name("Dortmund")
    date = datetime(2025, 3, 1)
    match_data, encoded_match_data = dp.fetch_match_data(home_team, away_team, date)
    
    results = predictor.predict_and_determine_winner(encoded_match_data)
    print("\n=== Результаты предсказаний ===")
    for model_name, prediction in results["predictions"].items():
        print(f"{model_name}: {predictor.interpret_prediction(prediction, home_team, away_team)}")

    print("\n=== Средние вероятности ===")
    print(f"Победа {home_team}: {results['combined_probabilities']['home_win']:.2%}")
    print(f"Ничья: {results['combined_probabilities']['draw']:.2%}")
    print(f"Победа {away_team}: {results['combined_probabilities']['away_win']:.2%}")

    print("\n=== Победитель матча ===")
    print(f"Результат: {results['winner']}")
    
    home_team_avg_goals = match_data['home_xg_last_5'].iloc[0]
    away_team_avg_goals = match_data['away_xg_last_5'].iloc[0]
    
    # Распределение Пуассона для расчета вероятностей точного счета
    poisson_probabilities = predictor.poisson_distribution(home_team_avg_goals, away_team_avg_goals)
    print("\n=== Вероятности точного счета (распределение Пуассона) ===")
    sorted_probabilities = sorted(poisson_probabilities.items(), key=lambda x: x[1], reverse=True)
    for score, probability in sorted_probabilities:
        if probability > 0.05:
            print(f"{score[0]}:{score[1]} - {probability*100:.2f}%")
        else:
            break
        
    # score_probabilities = predictor.calculate_score_probabilities(home_team_avg_goals, away_team_avg_goals)
    # # Сортировка вероятностей по убыванию
    # sorted_probabilities = sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)

    # # Вывод вероятностей точного счета
    # print("\n=== Вероятности точного счета ===")
    # for score, probability in sorted_probabilities:
    #     if probability > 0.01:
    #         print(f"{score[0]}:{score[1]} - {probability*100:.2f}%")
    #     else:
    #         break

    # skellam_probabilities = predictor.skellam_distribution(home_team_avg_goals, away_team_avg_goals)

    # # Вывод вероятностей
    # print("\n=== Вероятности точного счета (распределение Скетла) ===")
    # sorted_probabilities = sorted(skellam_probabilities.items(), key=lambda x: x[1], reverse=True)
    # for score, probability in sorted_probabilities:
    #     if probability > 0.02:
    #         print(f"{score[0]}:{score[1]} - {probability*100:.2f}%")
    #     else:
    #         break

    # # Метод Монте-Карло для симуляции матча
    # monte_carlo_probabilities = predictor.monte_carlo_simulation(home_team_avg_goals, away_team_avg_goals)
    # print("\n=== Метод Монте-Карло ===")
    # print(f"Победа {home_team}: {monte_carlo_probabilities['home_win']:.2%}")
    # print(f"Ничья: {monte_carlo_probabilities['draw']:.2%}")
    # print(f"Победа {away_team}: {monte_carlo_probabilities['away_win']:.2%}")