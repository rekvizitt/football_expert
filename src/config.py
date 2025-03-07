from pathlib import Path

class ConfigManager:
    def __init__(self):
        # Определение базовых путей
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.data_dir / 'models'
        self.metrics_dir = self.data_dir / 'metrics'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.train_data_dir = self.data_dir / 'train'
        self.fbref_data_dir = self.raw_data_dir / 'fbref'
        self.sofifa_data_dir = self.raw_data_dir / 'sofifa'

        # Создание директорий, если они не существуют
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.train_data_dir.mkdir(parents=True, exist_ok=True)
        self.fbref_data_dir.mkdir(parents=True, exist_ok=True)
        self.sofifa_data_dir.mkdir(parents=True, exist_ok=True)

        # Дополнительные настройки
        self.logging_level = 'DEBUG'
        
        # Конфигурация полей
        self.categorical_cols = ['home_team', 'away_team']
        self.numerical_cols = [
            'home_goals_last_5', 'away_goals_last_5',
            'home_conceded_goals_last_5', 'away_conceded_goals_last_5',
            'home_xg_last_5', 'away_xg_last_5',
            'home_wins_last_5', 'away_wins_last_5',
            'home_draws_last_5', 'away_draws_last_5',
            'home_losses_last_5', 'away_losses_last_5',
            'home_possession', 'away_possession',
            'home_ga_per_90', 'away_ga_per_90',
            'home_performance_ga', 'away_performance_ga',
            'home_xg', 'away_xg',
            'home_xag', 'away_xag',
            'home_prgc', 'away_prgc',
            'home_prgp', 'away_prgp',
            'home_attack_rating', 'away_attack_rating',
            'home_midfield_rating', 'away_midfield_rating',
            'home_defence_rating', 'away_defence_rating',
            'home_overall_rating', 'away_overall_rating',
            'home_days_since_last_match', 'away_days_since_last_match'
        ]