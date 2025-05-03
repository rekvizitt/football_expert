import unittest
from src.data.wrapper import DataWrapper
from src.config import ConfigManager

class TestDataWrapper(unittest.TestCase):
    def setUp(self):
        # Инициализация объектов перед каждым тестом
        self.leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
        self.seasons = ["2425"]
        self.data_wrapper = DataWrapper(leagues=self.leagues, seasons=self.seasons)
        self.config_manager = ConfigManager()

    def test_directories_exist(self):
        # Проверка, что все необходимые директории существуют
        self.assertTrue(self.data_wrapper.raw_data_dir.exists())
        self.assertTrue(self.data_wrapper.fbref_data_dir.exists())
        self.assertTrue(self.data_wrapper.sofifa_data_dir.exists())

    def test_upcoming_matches_cache_path(self):
        # Проверка, что путь к кэшу предстоящих матчей корректен
        expected_path = self.config_manager.get_raw_data_dir() / 'upcoming_matches.json'
        self.assertEqual(self.data_wrapper.upcoming_matches_cache_path, expected_path)

    def test_fbref_initialization(self):
        # Проверка, что объект fbref инициализирован корректно
        self.assertIsNotNone(self.data_wrapper.fbref)
        self.assertEqual(self.data_wrapper.fbref.leagues, self.leagues)
        self.assertEqual(self.data_wrapper.fbref.seasons, self.seasons)

    def test_sofifa_initialization(self):
        # Проверка, что объект sofifa инициализирован корректно
        self.assertIsNotNone(self.data_wrapper.sofifa)
        self.assertEqual(self.data_wrapper.sofifa.leagues, self.leagues)

if __name__ == '__main__':
    unittest.main()
