from pathlib import Path
from datetime import datetime, timedelta
import os
import re
import joblib
from src.config import ConfigManager
from src.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPrepare:
    def __init__(self):
        config_manager = ConfigManager()

        self.train_data_dir = config_manager.train_data_dir
        self.processed_data_dir = config_manager.processed_dir

        self.team_ratings_path = Path(self.processed_data_dir) / "team_ratings.json"
        self.team_stats_path = Path(self.processed_data_dir) / "team_stats.json"
        self.match_results_path = Path(self.processed_data_dir) / "match_results.json"

        self.encoder_path = Path(self.train_data_dir) / "label_encoder.pkl"
        self.scaler_path = Path(self.train_data_dir) / "standard_scaler.pkl"

        self.team_ratings = self.load_data(self.team_ratings_path)
        self.team_stats = self.load_data(self.team_stats_path)
        self.match_results = self.load_data(self.match_results_path)

    def load_data(self, path):
        if not path.exists():
            logger.error(f"File not found: {path}")
            return pd.DataFrame()
        try:
            return pd.read_json(path)
        except Exception as e:
            logger.error(f"Error while loading data from {path}: {e}")
            return pd.DataFrame()

    def save_data(self, data, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_json(path, orient='records', indent=4)
            logger.debug(f"Data saved to {path}")
        except Exception as e:
            logger.error(f"Error while saving data to {path}: {e}")

    def save_preprocessors(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(pd.concat([self.match_results['home_team'], self.match_results['away_team']]))
        joblib.dump(self.label_encoder, self.encoder_path)
        logger.debug(f"LabelEncoder saved to {self.encoder_path}")

        self.standard_scaler = StandardScaler()
        features = self.train_data.drop(columns=['date', 'home_team', 'away_team', 'score', 'result']).select_dtypes(include=[float, int]).columns
        self.standard_scaler.fit(self.train_data[features])
        joblib.dump(self.standard_scaler, self.scaler_path)
        logger.debug(f"StandardScaler saved to {self.scaler_path}")

    def prepare_train_data(self):
        self.match_results['date'] = pd.to_datetime(self.match_results['date'], errors='coerce')

        train_data = []

        for _, match in self.match_results.iterrows():
            date = match['date']
            home_team = match['home_team']
            away_team = match['away_team']

            home_stats = self.calculate_team_stats(home_team, date)
            away_stats = self.calculate_team_stats(away_team, date)

            if not home_stats or not away_stats:
                continue

            match_data = {
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                **{f'home_{key}': value for key, value in home_stats.items()},
                **{f'away_{key}': value for key, value in away_stats.items()},
            }

            if pd.isnull(match['score']):
                logger.warning(f"Score для матча {home_team} vs {away_team} на дату {date} отсутствует. Пропускаем матч.")
                continue

            match_data['score'] = match['score']
            train_data.append(match_data)

        self.train_data = pd.DataFrame(train_data)
        logger.debug(f"Train data prepared with {len(self.train_data)} records")

        self.train_data['result'] = self.train_data.apply(
            lambda row: self.calculate_result(row['score'], row['date'], row['home_team']), axis=1
        )

        self.train_data['result'] = self.train_data['result'].map({'W': 1, 'D': 0, 'L': -1})

        X = self.train_data.drop(columns=['date', 'home_team', 'away_team', 'score', 'result'])
        y = self.train_data['result']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.save_data(X_train, Path(self.train_data_dir) / "X_train.json")
        self.save_data(X_test, Path(self.train_data_dir) / "X_test.json")
        self.save_data(y_train, Path(self.train_data_dir) / "y_train.json")
        self.save_data(y_test, Path(self.train_data_dir) / "y_test.json")

        logger.debug(f"X_train shape: {X_train.shape}")
        logger.debug(f"X_test shape: {X_test.shape}")
        logger.debug(f"y_train shape: {y_train.shape}")
        logger.debug(f"y_test shape: {y_test.shape}")

    def calculate_team_stats(self, team_name, date):
        team_stats = self.team_stats[self.team_stats['team'] == team_name]
        if team_stats.empty:
            logger.warning(f"Статистика для команды {team_name} отсутствует.")
            return {}

        team_rating = self.team_ratings[self.team_ratings['team'] == team_name]
        if team_rating.empty:
            logger.warning(f"Рейтинги для команды {team_name} отсутствуют.")
            return {}

        attack_rating = team_rating['attack'].iloc[0] if not team_rating['attack'].isnull().all() else None
        midfield_rating = team_rating['midfield'].iloc[0] if not team_rating['midfield'].isnull().all() else None
        defence_rating = team_rating['defence'].iloc[0] if not team_rating['defence'].isnull().all() else None
        overall_rating = team_rating['overall'].iloc[0] if not team_rating['overall'].isnull().all() else None

        last_5_matches = self.match_results[
            (self.match_results['home_team'] == team_name) | (self.match_results['away_team'] == team_name)
        ]
        last_5_matches = last_5_matches[last_5_matches['date'] < date].sort_values(by='date', ascending=False).head(5)

        if last_5_matches.empty:
            logger.warning(f"Нет матчей для команды {team_name} за последние 5 матчей до {date}.")
            return {}

        goals_last_5 = last_5_matches.apply(
            lambda row: int(row['score'].split('–')[0]) if row['home_team'] == team_name and pd.notnull(row['score']) else
            int(row['score'].split('–')[1]) if row['away_team'] == team_name and pd.notnull(row['score']) else 0, axis=1).sum()

        conceded_goals_last_5 = last_5_matches.apply(
            lambda row: int(row['score'].split('–')[1]) if row['home_team'] == team_name and pd.notnull(row['score']) else
            int(row['score'].split('–')[0]) if row['away_team'] == team_name and pd.notnull(row['score']) else 0, axis=1).sum()

        xg_last_5 = last_5_matches.apply(
            lambda row: row['home_xg'] if row['home_team'] == team_name else row['away_xg'], axis=1).mean()

        results_last_5 = last_5_matches.apply(lambda row: self.calculate_result(row['score'], row['date'], team_name), axis=1)
        wins_last_5 = sum(results_last_5 == 'W')
        draws_last_5 = sum(results_last_5 == 'D')
        losses_last_5 = sum(results_last_5 == 'L')

        possession = team_stats['Poss_'].mean() if not team_stats['Poss_'].isnull().all() else 0.0
        ga_per_90 = team_stats['Per 90 Minutes_G+A'].mean() if not team_stats['Per 90 Minutes_G+A'].isnull().all() else 0.0
        performance_ga = team_stats['Performance_G+A'].mean() if not team_stats['Performance_G+A'].isnull().all() else 0.0
        xg = team_stats['Expected_xG'].mean() if not team_stats['Expected_xG'].isnull().all() else 0.0
        xag = team_stats['Expected_xAG'].mean() if not team_stats['Expected_xAG'].isnull().all() else 0.0
        prgc = team_stats['Progression_PrgC'].mean() if not team_stats['Progression_PrgC'].isnull().all() else 0.0
        prgp = team_stats['Progression_PrgP'].mean() if not team_stats['Progression_PrgP'].isnull().all() else 0.0

        return {
            'goals_last_5': goals_last_5,
            'conceded_goals_last_5': conceded_goals_last_5,
            'xg_last_5': xg_last_5 if not pd.isnull(xg_last_5) else 0.0,
            'wins_last_5': wins_last_5,
            'draws_last_5': draws_last_5,
            'losses_last_5': losses_last_5,
            'possession': possession,
            'ga_per_90': ga_per_90,
            'performance_ga': performance_ga,
            'xg': xg,
            'xag': xag,
            'prgc': prgc,
            'prgp': prgp,
            'attack_rating': attack_rating if not pd.isnull(attack_rating) else 0.0,
            'midfield_rating': midfield_rating if not pd.isnull(midfield_rating) else 0.0,
            'defence_rating': defence_rating if not pd.isnull(defence_rating) else 0.0,
            'overall_rating': overall_rating if not pd.isnull(overall_rating) else 0.0,
        }

    def calculate_result(self, score, date, team_name):
        if pd.isnull(score):
            return 'D'

        home_score, away_score = map(int, score.split('–'))

        home_match = self.match_results[(self.match_results['date'] == date) & (self.match_results['home_team'] == team_name)]
        away_match = self.match_results[(self.match_results['date'] == date) & (self.match_results['away_team'] == team_name)]

        if not home_match.empty:
            return 'W' if home_score > away_score else 'L' if home_score < away_score else 'D'
        elif not away_match.empty:
            return 'W' if away_score > home_score else 'L' if away_score < home_score else 'D'
        else:
            logger.warning(f"Матч для команды {team_name} на дату {date} не найден.")
            return 'D'

    def fetch_match_data(self, home_team, away_team, date):
        self.label_encoder = joblib.load(self.encoder_path)
        self.standard_scaler = joblib.load(self.scaler_path)

        home_stats = self.calculate_team_stats(home_team, date)
        away_stats = self.calculate_team_stats(away_team, date)

        if not home_stats or not away_stats:
            logger.error(f"Не удалось получить статистику для команд {home_team} и {away_team} на дату {date}.")
            return {}

        match_data = {
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            **{f'home_{key}': value for key, value in home_stats.items()},
            **{f'away_{key}': value for key, value in away_stats.items()},
        }

        match_data_encoded = match_data.copy()
        match_data_encoded['home_team'] = self.label_encoder.transform([match_data['home_team']])[0]
        match_data_encoded['away_team'] = self.label_encoder.transform([match_data['away_team']])[0]

        numerical_features = [f'home_{key}' for key in home_stats.keys()] + [f'away_{key}' for key in away_stats.keys()]
        match_data_numerical = pd.DataFrame([match_data_encoded])[numerical_features]
        scaled_features = self.standard_scaler.transform(match_data_numerical)

        for i, feature in enumerate(numerical_features):
            match_data_encoded[feature] = scaled_features[0, i]
            
        match_data_encoded.pop('date', None)
        match_data_encoded.pop('home_team', None)
        match_data_encoded.pop('away_team', None)
        match_data_encoded_df = pd.DataFrame([match_data_encoded])

        return match_data_encoded_df


if __name__ == "__main__":
    dp = DataPrepare()
    
    # get train data
    # dp.prepare_train_data()
    # dp.save_preprocessors()
    
    # get match_data
    # home_team = "Manchester United"
    # away_team = "Liverpool"
    # date = datetime(2025, 3, 1)
    # match_data = dp.fetch_match_data(home_team, away_team, date)
    # logger.debug(f"Fetched match data: {len(match_data)}")