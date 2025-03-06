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