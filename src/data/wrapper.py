import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.config import ConfigManager
from src.logger import logger

class DataWrapper:
    def __init__(self, leagues, seasons):
        self.leagues = leagues
        self.seasons = seasons
        
        config_manager = ConfigManager()
        
        self.raw_data_dir = config_manager.raw_data_dir
        self.fbref_data_dir = config_manager.fbref_data_dir
        self.sofifa_data_dir = config_manager.sofifa_data_dir

        logger.debug(f"Initialized DataWrapper with leagues: {self.leagues} and seasons: {self.seasons}")
        self.fbref = sd.FBref(leagues=self.leagues, seasons=self.seasons, data_dir=self.fbref_data_dir)
        self.sofifa = sd.SoFIFA(leagues=self.leagues, versions='latest', data_dir=self.sofifa_data_dir)
        
        logger.info("Starting to fetch data")
        
        logger.debug("Fetching FBRef data")
        self.fetch_match_results()
        self.fetch_team_stats()
        
        logger.debug("Fetching SoFIFA data")
        self.fetch_team_ratings()
    
    def fetch_match_results(self):
        ttl = timedelta(hours=8)
        cache_file_path = Path(self.raw_data_dir) / "match_results.json"
        
        logger.debug("Checking cached match results")
        if cache_file_path.exists():
            file_mod_time = datetime.fromtimestamp(cache_file_path.stat().st_mtime)
            if datetime.now() - file_mod_time < ttl:
                logger.debug("Using cached match results")
                return
        
        logger.debug("Fetching match results")
        try:
            match_results = self.fbref.read_schedule()
            fields_to_drop = ['attendance', 'venue', 'referee', 'match_report', 'notes', 'game_id']
            match_results.drop(columns=fields_to_drop, errors='ignore', inplace=True)
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file_path, 'w') as f:
                match_results.to_json(f, orient='records', indent=4)
        except Exception as e:
            logger.error(f"Error while fetching match results: {e}")
    
    def fetch_team_stats(self):
        ttl = timedelta(hours=8)
        cache_file_path = Path(self.raw_data_dir) / "team_stats.json"
        
        logger.debug("Checking cached team stats")
        
        if cache_file_path.exists():
            file_mod_time = datetime.fromtimestamp(cache_file_path.stat().st_mtime)
            if datetime.now() - file_mod_time < ttl:
                logger.debug("Using cached team stats")
                return
        
        logger.debug("Fetching team stats")
        try:
            team_stats = self.fbref.read_team_season_stats()
            if isinstance(team_stats, pd.DataFrame) and isinstance(team_stats.columns, pd.MultiIndex):
                team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns.values]
                
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file_path, 'w') as f:
                team_stats.to_json(f, orient='records', indent=4)
        except Exception as e:
            logger.error(f"Error while fetching team stats: {e}")
    
    def fetch_team_ratings(self):
        ttl = timedelta(hours=168)
        cache_file_path = Path(self.raw_data_dir) / "team_ratings.json"
        
        logger.debug("Checking cached team ratings")

        if cache_file_path.exists():
            file_mod_time = datetime.fromtimestamp(cache_file_path.stat().st_mtime)
            if datetime.now() - file_mod_time < ttl:
                logger.debug("Using cached team ratings")
                return

        logger.debug("Fetching team ratings")
        try:
            sofifa_ratings = self.sofifa.read_team_ratings()
            
            # Если sofifa_ratings имеет MultiIndex, сбросьте его
            if isinstance(sofifa_ratings.index, pd.MultiIndex):
                sofifa_ratings = sofifa_ratings.reset_index()
            
            # Удаляем ненужные поля
            fields_to_drop = [
                'fifa_edition', 'update', 'transfer_budget', 'build_up_speed', 'build_up_dribbling', 'build_up_passing',
                'build_up_positioning', 'chance_creation_crossing', 'chance_creation_passing', 'chance_creation_shooting',
                'chance_creation_positioning', 'defence_aggression', 'defence_pressure', 'defence_team_width',
                'defence_defender_line', 'defence_domestic_prestige', 'international_prestige', 'players',
                'starting_xi_average_age', 'whole_team_average_age'
            ]
            sofifa_ratings.drop(columns=fields_to_drop, errors='ignore', inplace=True)
            
            # Сохраняем обновленные данные в кэш-файл
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file_path, 'w') as f:
                sofifa_ratings.to_json(f, orient='records', indent=4)
        except Exception as e:
            logger.error(f"Error while fetching team ratings: {e}")
        
if __name__ == "__main__":
    leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
    seasons = ["2425"]
    data_wrapper = DataWrapper(leagues, seasons)