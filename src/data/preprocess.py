from pathlib import Path
from datetime import datetime, timedelta
import os
import re
from src.config import ConfigManager
from src.logger import logger
import pandas as pd
from fuzzywuzzy import process

class DataPreprocess:
    def __init__(self):       
        config_manager = ConfigManager()
        
        self.raw_data_dir = config_manager.raw_data_dir
        self.raw_match_results_path = Path(self.raw_data_dir) / "match_results.json"
        self.raw_team_stats_path = Path(self.raw_data_dir) / "team_stats.json"
        self.raw_team_ratings_path = Path(self.raw_data_dir) / "team_ratings.json"
        
        self.processed_data_dir = config_manager.processed_dir
        self.processed_match_results_path = Path(self.processed_data_dir) / "match_results.json"
        self.processed_team_stats_path = Path(self.processed_data_dir) / "team_stats.json"
        self.processed_team_ratings_path = Path(self.processed_data_dir) / "team_ratings.json"
        self.upcoming_matches_path = Path(self.processed_data_dir) / "upcoming_matches.json"
        
        logger.info("Starting to preprocess data")
        self.preprocess_data()  
      
    def preprocess_data(self):
        self.team_ratings = self.load_data(self.raw_team_ratings_path)
        self.team_stats = self.load_data(self.raw_team_stats_path)
        self.match_results = self.load_data(self.raw_match_results_path)
    
        self.team_names_list = self.create_team_names_list()
        self.add_team_names_to_team_stats()
        self.update_team_names_in_match_results()
        self.create_upcoming_matches() 
        
        self.save_data(self.team_ratings, self.processed_team_ratings_path)
        self.save_data(self.team_stats, self.processed_team_stats_path)
        self.save_data(self.match_results, self.processed_match_results_path)
               
        logger.debug(f"Preprocess data saved")
        
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

    def create_team_names_list(self):
        team_names_list = self.team_ratings['team'].dropna().unique().tolist()
        logger.debug(f"Created team_names_list with {len(team_names_list)} teams")
        return team_names_list

    def add_team_names_to_team_stats(self):
        def extract_team_name_from_url(url):
            if pd.isna(url):
                return None
            match = re.search(r'en/squads/[\w-]+/(.+?)-Stats', url)
            if match:
                return match.group(1)
            return None
        
        def map_team_name(team_name, team_names_list):
            if pd.isna(team_name):
                return None
            
            for name in team_names_list:
                if name.lower() in team_name.lower():
                    return name
            
            best_match, score = process.extractOne(team_name, team_names_list)
            if score > 60:
                return best_match
            
            return None
        
        self.team_stats['team_name_from_url'] = self.team_stats['url_'].apply(extract_team_name_from_url)
        self.team_stats['team'] = self.team_stats['team_name_from_url'].apply(lambda x: map_team_name(x, self.team_names_list))
        self.team_stats.drop(columns=['team_name_from_url', 'url_'], inplace=True)
        logger.debug("Added team_name to team_stats")

    def update_team_names_in_match_results(self):
        def map_team_name(team_name, team_names_list):
            if pd.isna(team_name):
                return None
            
            for name in team_names_list:
                if name.lower() in team_name.lower():
                    return name
            
            best_match, score = process.extractOne(team_name, team_names_list)
            if score > 60:
                return best_match
            
            return None
        
        self.match_results['home_team'] = self.match_results['home_team'].apply(lambda x: map_team_name(x, self.team_names_list))
        self.match_results['away_team'] = self.match_results['away_team'].apply(lambda x: map_team_name(x, self.team_names_list))
        
        logger.debug("Updated team names in match_results")

    def create_upcoming_matches(self):
        today = datetime.now().date()
        future_date = today + timedelta(days=7)
        
        self.match_results['date'] = pd.to_datetime(self.match_results['date'], errors='coerce')
        upcoming_matches = self.match_results[
            (self.match_results['date'].dt.date >= today) & (self.match_results['date'].dt.date <= future_date)
        ][['date', 'time', 'home_team', 'away_team']]
        
        team_league_dict = dict(zip(self.team_ratings['team'], self.team_ratings['league']))

        def get_league_for_team(team, team_league_dict):
            if pd.isna(team):
                return None
            return team_league_dict.get(team)
        
        upcoming_matches['league'] = upcoming_matches['home_team'].apply(lambda x: get_league_for_team(x, team_league_dict))        
        upcoming_matches.to_json(self.upcoming_matches_path, orient='records', indent=4)
        logger.debug(f"Saved upcoming matches to {self.upcoming_matches_path}")

if __name__ == "__main__":
    DataPreprocess()