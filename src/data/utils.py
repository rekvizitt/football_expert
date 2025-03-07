from pathlib import Path
from fuzzywuzzy import process
from src.config import ConfigManager
from src.logger import logger
import pandas as pd

def load_data(path):
    if not path.exists():
        logger.error(f"File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_json(path)
    except Exception as e:
        logger.error(f"Error while loading data from {path}: {e}")
        return pd.DataFrame()

def save_data(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_json(path, orient='records', indent=4)
        logger.debug(f"Data saved to {path}")
    except Exception as e:
        logger.error(f"Error while saving data to {path}: {e}")

def create_team_names_list():
    config_manager = ConfigManager()
    raw_data_dir = config_manager.raw_data_dir
    raw_team_ratings_path = Path(raw_data_dir) / "team_ratings.json"
    team_ratings = load_data(raw_team_ratings_path)
    team_names_list = team_ratings['team'].dropna().unique().tolist()
    return team_names_list

def create_team_league_dict():
    config_manager = ConfigManager()
    raw_data_dir = config_manager.raw_data_dir
    raw_team_ratings_path = Path(raw_data_dir) / "team_ratings.json"
    team_ratings = load_data(raw_team_ratings_path)
    team_league_dict = {
        row['team']: row['league']
        for _, row in team_ratings.dropna(subset=['team', 'league']).iterrows()
    }
    return team_league_dict

def find_team_name(team_name: str):
    team_names_list = create_team_names_list()
    
    if pd.isna(team_name):
        return None
            
    for name in team_names_list:
        if name.lower() in team_name.lower():
            return name
            
    best_match, score = process.extractOne(team_name, team_names_list)
    if score > 60:
        return best_match
    
    return None

# if __name__ == '__main__':
#     team_name = find_team_name('dorntumnt')
#     logger.debug(team_name)