from pathlib import Path
from datetime import datetime, timedelta
import os
import re
from src.config import ConfigManager
from src.logger import logger
from src.data.utils import save_data, load_data, create_team_names_list
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
        
        self.save_data = save_data
        self.load_data = load_data
        self.create_team_names_list = create_team_names_list
        
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

            if team_name == "Rennes":
                return "Stade Rennais FC"

            for name in team_names_list:
                if name.lower() == team_name.lower():
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
        # Создаем словарь для кэширования результатов
        team_name_cache = {}

        def map_team_name(team_name, team_names_list):
            # Возврат None для пустых значений
            if pd.isna(team_name):
                return None
            
            if team_name == "Rennes":
                return "Stade Rennais FC"

            # Используем кэш для уже обработанных команд
            if team_name in team_name_cache:
                return team_name_cache[team_name]
            
            # Проверка на точное совпадение в нижнем регистре
            team_name_lower = team_name.lower()
            for name in team_names_list:
                if name.lower() == team_name_lower:
                    team_name_cache[team_name] = name
                    return name
            
            # Проверка на вхождение подстроки
            for name in team_names_list:
                if name.lower() in team_name_lower:
                    team_name_cache[team_name] = name
                    return name
            
            # Если не найдено, используем нечеткое сопоставление
            best_match, score = process.extractOne(team_name, team_names_list)
            result = best_match if score > 60 else None
            team_name_cache[team_name] = result
            return result

        # Преобразуем team_names_list в список, если это не список
        if not isinstance(self.team_names_list, list):
            team_names_list = list(self.team_names_list)
        else:
            team_names_list = self.team_names_list

        # Векторизуем обработку колонок для ускорения
        self.match_results['home_team'] = self.match_results['home_team'].map(
            lambda x: map_team_name(x, team_names_list) if not pd.isna(x) else None
        )
        self.match_results['away_team'] = self.match_results['away_team'].map(
            lambda x: map_team_name(x, team_names_list) if not pd.isna(x) else None
        )

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
    
    