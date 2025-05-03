from src.data.wrapper import DataWrapper
from src.data.preprocess import DataPreprocess
from src.data.prepare_data import DataPrepare
from src.data.utils import load_data, find_team_name, create_team_names_list, create_team_league_dict
from src.models.train import ModelTrainer
from src.models.predict import MatchPredictor
from src.config import ConfigManager
from src.logger import logger
from pathlib import Path
import datetime

class FootballExpertApi:
    def __init__(self, leagues, seasons):
        self.config_manager = ConfigManager()
        self.leagues = leagues
        self.seasons = seasons
        self.data_wrapper = DataWrapper(self.leagues, self.seasons)
        self.data_preprocess = DataPreprocess()
        self.processed_data_dir = self.config_manager.processed_dir
        self.metrics_dir = self.config_manager.metrics_dir
        self.upcoming_matches_path = Path(self.processed_data_dir) / "upcoming_matches.json"
        self.upcoming_matches = load_data(self.upcoming_matches_path)
        self.dp = DataPrepare()
        self.trainer = ModelTrainer()
        self.predictor = MatchPredictor()
        self.find_team_name = find_team_name
        self.create_team_names_list = create_team_names_list
        self.create_team_league_dict = create_team_league_dict

    def prepare_train_data(self):
        self.dp.prepare_train_data()

    def train_models(self):
        self.trainer.train_and_save_models()

    def get_match_data(self, home_team, away_team, date):
        return self.dp.fetch_match_data(home_team, away_team, date)

    def get_match_date_or_today(self, home_team, away_team):
        match = self.upcoming_matches[
            (self.upcoming_matches['home_team'] == home_team) &
            (self.upcoming_matches['away_team'] == away_team)
        ]
        
        if not match.empty:
            return match.iloc[0]['date'].date()
        else:
            return datetime.date.today()

    def predict_match(self, home_team, away_team, date):
        _, encoded_match_data = self.get_match_data(home_team, away_team, date)
        results = self.predictor.predict_and_determine_winner(encoded_match_data)
        return results

    def print_prediction_results(self, home_team, away_team, results):
        print("\n=== Результаты предсказаний ===")
        for model_name, prediction in results["predictions"].items():
            print(f"{model_name}: {self.predictor.interpret_prediction(prediction, home_team, away_team)}")

        print("\n=== Средние вероятности ===")
        print(f"Победа {home_team}: {results['combined_probabilities']['home_win']:.2%}")
        print(f"Ничья: {results['combined_probabilities']['draw']:.2%}")
        print(f"Победа {away_team}: {results['combined_probabilities']['away_win']:.2%}")

        print("\n=== Победитель матча ===")
        print(f"Результат: {results['winner']}")

    def calculate_poisson_probabilities(self, match_data):
        home_team_avg_goals = match_data['home_xg_last_5'].iloc[0]
        away_team_avg_goals = match_data['away_xg_last_5'].iloc[0]
        poisson_probabilities = self.predictor.poisson_distribution(home_team_avg_goals, away_team_avg_goals)
        return poisson_probabilities

    def print_poisson_probabilities(self, poisson_probabilities):
        print("\n=== Вероятности точного счета (распределение Пуассона) ===")
        sorted_probabilities = sorted(poisson_probabilities.items(), key=lambda x: x[1], reverse=True)
        for score, probability in sorted_probabilities:
            if probability > 0.05:
                print(f"{score[0]}:{score[1]} - {probability*100:.2f}%")
            else:
                break

if __name__ == '__main__':
    leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
    seasons = ["2425"]
    api = FootballExpertApi(leagues, seasons)
    
    # Example train models:
    api.prepare_train_data()
    api.train_models()

    # Example predict match:
    home_team = find_team_name("Leverkusen")
    away_team = find_team_name("Liverpool")
    date = datetime.datetime(2025, 3, 1)
    match_data, _ = api.get_match_data(home_team, away_team, date)
    logger.debug(f"Fetched match data: {match_data}")

    results = api.predict_match(home_team, away_team, date)
    api.print_prediction_results(home_team, away_team, results)

    poisson_probabilities = api.calculate_poisson_probabilities(match_data)
    api.print_poisson_probabilities(poisson_probabilities)