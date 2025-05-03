import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import ConfigManager
from src.logger import logger
from src.data.utils import save_data, load_data

class DataPrepare:
    def __init__(self):
        config_manager = ConfigManager()

        self.train_data_dir = config_manager.train_data_dir
        self.processed_data_dir = config_manager.processed_dir

        self.team_ratings_path = Path(self.processed_data_dir) / "team_ratings.json"
        self.team_stats_path = Path(self.processed_data_dir) / "team_stats.json"
        self.match_results_path = Path(self.processed_data_dir) / "match_results.json"

        self.encoder_path = Path(self.train_data_dir) / "onehot_encoder.pkl"
        self.scaler_path = Path(self.train_data_dir) / "standard_scaler.pkl"

        self.save_data = save_data
        self.load_data = load_data
        
        self.team_ratings = self.load_data(self.team_ratings_path)
        self.team_stats = self.load_data(self.team_stats_path)
        self.match_results = self.load_data(self.match_results_path)
        
        self._team_stats_cache = {}
        self._team_stats_indexed = None
        self._team_ratings_indexed = None
        self._matches_by_team = None
        
        self.categorical_cols = config_manager.categorical_cols
        self.numerical_cols = config_manager.numerical_cols

    def calculate_team_stats(self, team_name, date):
        """
        Расчет статистики команды.

        :param team_name: Название команды.
        :param date: Дата матча.
        :return: Словарь со статистикой команды.
        """
        # Используем кэширование с помощью lru_cache для предотвращения повторных вычислений
        # Создаем ключ кэша из team_name и date
        cache_key = (team_name, date)
        
        # Проверяем, есть ли уже в кэше результат
        if cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]
        
        # Создаем кэш, если его еще нет
        if not self._team_stats_cache:
            self._team_stats_cache = {}
        
        # Индексация для более быстрого поиска
        if self._team_stats_indexed is None:
            self._team_stats_indexed = self.team_stats.set_index('team')
            self._team_ratings_indexed = self.team_ratings.set_index('team')
        
        # Получаем статистику команды с помощью индекса
        try:
            team_stats = self._team_stats_indexed.loc[team_name]
        except KeyError:
            logger.warning(f"Статистика для команды {team_name} отсутствует.")
            self._team_stats_cache[cache_key] = {}
            return {}
        
        # Получаем рейтинги команды
        try:
            team_rating = self._team_ratings_indexed.loc[team_name]
        except KeyError:
            logger.warning(f"Рейтинги для команды {team_name} отсутствуют.")
            self._team_stats_cache[cache_key] = {}
            return {}
        
        # Извлекаем рейтинги один раз
        attack_rating = team_rating.get('attack', 0.0)
        midfield_rating = team_rating.get('midfield', 0.0)
        defence_rating = team_rating.get('defence', 0.0)
        overall_rating = team_rating.get('overall', 0.0)
        
        # Фильтруем последние 5 матчей более эффективно
        if self._matches_by_team is None:
            # Создаем кэш для быстрого доступа к матчам команды
            self._matches_by_team = {}
            for team in set(self.match_results['home_team']).union(set(self.match_results['away_team'])):
                self._matches_by_team[team] = self.match_results[
                    (self.match_results['home_team'] == team) | 
                    (self.match_results['away_team'] == team)
                ].sort_values(by='date', ascending=False)
        
        # Получаем последние 5 матчей до указанной даты
        try:
            team_matches = self._matches_by_team[team_name]
            last_5_matches = team_matches[team_matches['date'] < date].head(5)
        except KeyError:
            # Если команда не найдена в кэше
            last_5_matches = self.match_results[
                (self.match_results['home_team'] == team_name) | 
                (self.match_results['away_team'] == team_name)
            ]
            last_5_matches = last_5_matches[last_5_matches['date'] < date].sort_values(by='date', ascending=False).head(5)
        
        if last_5_matches.empty:
            logger.warning(f"Нет матчей для команды {team_name} за последние 5 матчей до {date}.")
            self._team_stats_cache[cache_key] = {}
            return {}
        
        # Вычисляем все метрики векторизованно за один проход
        goals_last_5 = 0
        conceded_goals_last_5 = 0
        xg_values = []
        results = []
        
        for _, row in last_5_matches.iterrows():
            if pd.isna(row['score']):
                continue
                
            score_parts = row['score'].split('\u2013')
            if len(score_parts) != 2:
                continue
                
            try:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
            except ValueError:
                continue
                
            if row['home_team'] == team_name:
                goals_last_5 += home_score
                conceded_goals_last_5 += away_score
                xg_values.append(row.get('home_xg', 0))
                
                if home_score > away_score:
                    results.append('W')
                elif home_score == away_score:
                    results.append('D')
                else:
                    results.append('L')
            else:
                goals_last_5 += away_score
                conceded_goals_last_5 += home_score
                xg_values.append(row.get('away_xg', 0))
                
                if away_score > home_score:
                    results.append('W')
                elif away_score == home_score:
                    results.append('D')
                else:
                    results.append('L')
        
        # Подсчитываем результаты
        wins_last_5 = results.count('W')
        draws_last_5 = results.count('D')
        losses_last_5 = results.count('L')
        xg_last_5 = sum(xg_values) / len(xg_values) if xg_values else 0.0
        
        # Извлекаем остальные статистики более эффективно, избегая многократных вычислений mean()
        stats_means = {}
        for stat in ['Poss_', 'Per 90 Minutes_G+A', 'Performance_G+A', 'Expected_xG', 
                    'Expected_xAG', 'Progression_PrgC', 'Progression_PrgP']:
            if stat in team_stats.index and not pd.isnull(team_stats[stat]):
                stats_means[stat] = team_stats[stat]
            else:
                stats_means[stat] = 0.0
        
        # Формируем результат
        result = {
            'goals_last_5': goals_last_5,
            'conceded_goals_last_5': conceded_goals_last_5,
            'xg_last_5': xg_last_5,
            'wins_last_5': wins_last_5,
            'draws_last_5': draws_last_5,
            'losses_last_5': losses_last_5,
            'possession': stats_means.get('Poss_', 0.0),
            'ga_per_90': stats_means.get('Per 90 Minutes_G+A', 0.0),
            'performance_ga': stats_means.get('Performance_G+A', 0.0),
            'xg': stats_means.get('Expected_xG', 0.0),
            'xag': stats_means.get('Expected_xAG', 0.0),
            'prgc': stats_means.get('Progression_PrgC', 0.0),
            'prgp': stats_means.get('Progression_PrgP', 0.0),
            'attack_rating': attack_rating,
            'midfield_rating': midfield_rating,
            'defence_rating': defence_rating,
            'overall_rating': overall_rating,
        }
        
        # Добавляем разницу в днях между текущим и предыдущим матчем
        result['days_since_last_match'] = self.days_since_last_match(date, team_matches)
        
        # Сохраняем в кэше для повторного использования
        self._team_stats_cache[cache_key] = result
        
        return result
    
    def days_since_last_match(self, current_date, matches):
        team_matches = matches[matches['date'] < current_date]
        if team_matches.empty:
            return None
        last_match_date = team_matches['date'].max()
        return (current_date - last_match_date).days
    
    def get_train_data(self):
        # Преобразование даты один раз для всего DataFrame
        self.match_results['date'] = pd.to_datetime(self.match_results['date'], errors='coerce')
        
        # Отфильтруем сразу записи с отсутствующим score
        valid_matches = self.match_results.dropna(subset=['score']).copy()
        
        # Кэш для статистики команд
        team_stats_cache = {}
        match_data_list = []
        
        # Сортируем данные по дате для правильного расчета разницы между матчами
        valid_matches = valid_matches.sort_values(by='date')
        
        # Обработка каждого матча
        for _, match in valid_matches.iterrows():
            date = match['date']
            home_team = match['home_team']
            away_team = match['away_team']
            score = match['score']
            
            # Разбиение строки с результатом на домашний и гостевой счет
            score_parts = score.split('\u2013')
            if len(score_parts) != 2:
                logger.warning(f"Некорректный формат score: {score}")
                continue
            
            try:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
            except ValueError:
                logger.warning(f"Некорректный формат score: {score}")
                continue
            
            # Определение результата матча
            if home_score > away_score:
                result = 0  # Победа домашней команды
            elif home_score == away_score:
                result = 1  # Ничья
            else:
                result = 2  # Победа гостевой команды
                
            # Получаем статистику из кэша или вычисляем
            if (home_team, date) not in team_stats_cache:
                team_stats_cache[(home_team, date)] = self.calculate_team_stats(home_team, date)
            if (away_team, date) not in team_stats_cache:
                team_stats_cache[(away_team, date)] = self.calculate_team_stats(away_team, date)
            
            home_stats = team_stats_cache[(home_team, date)]
            away_stats = team_stats_cache[(away_team, date)]
            
            if not home_stats or not away_stats:
                continue
            
            # Создание записи данных
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'result': result
            }
            
            # Быстрое добавление статистик команд с префиксами
            match_data.update({f'home_{key}': value for key, value in home_stats.items()})
            match_data.update({f'away_{key}': value for key, value in away_stats.items()})
            
            match_data_list.append(match_data)
        
        # Создание DataFrame из собранных данных
        self.train_data = pd.DataFrame(match_data_list)

        logger.debug(f"Train data prepared with {len(self.train_data)} records")
    
    def clean_data(self):
        """
        Очистка данных: удаление пропущенных значений и дубликатов.
        :return: Очищенный DataFrame.
        """
        try:
            self.train_data.dropna(inplace=True)
            self.train_data.fillna(0, inplace=True)
            logger.debug('Данные очищены')
        except Exception as e:
            logger.warning(f'Ошибка очистки данных: {e}')
            return None

    def encode_categorical_data(self):
        """
        Кодирование категориальных переменных.
        :return: DataFrame с закодированными категориальными переменными.
        """
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = pd.DataFrame(encoder.fit_transform(self.train_data[self.categorical_cols]))
            encoded_data.columns = encoder.get_feature_names_out(self.categorical_cols)

            # Обновляем список категориальных колонок, если появились новые
            new_categorical_cols = list(set(encoded_data.columns) - set(self.train_data.columns))
            if new_categorical_cols:
                logger.debug(f'Обнаружены новые категориальные поля: {new_categorical_cols}')

            self.train_data = self.train_data.drop(self.categorical_cols, axis=1)
            self.train_data = pd.concat([self.train_data, encoded_data], axis=1)

            # Сохраняем encoder
            self.encoder_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(encoder, self.encoder_path)
            logger.debug(f'OneHotEncoder сохранен в {self.encoder_path}')
            logger.debug('Категориальные переменные закодированы')
        except Exception as e:
            logger.warning(f'Ошибка кодирования категориальных переменных: {e}')
            return None

    def normalize_numerical_data(self):
        """
        Нормализация числовых данных.
        :return: DataFrame с нормализованными числовыми данными.
        """
        try:
            scaler = StandardScaler()
            numerical_data = self.train_data[self.numerical_cols]

            # Обновляем список числовых колонок, если появились новые
            new_numerical_cols = list(set(numerical_data.columns) - set(self.numerical_cols))
            if new_numerical_cols:
                logger.debug(f'Обнаружены новые числовые поля: {new_numerical_cols}')
                self.numerical_cols.extend(new_numerical_cols)

            self.train_data[self.numerical_cols] = scaler.fit_transform(numerical_data)

            # Сохраняем scaler
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            logger.debug(f'StandardScaler сохранен в {self.scaler_path}')
            logger.debug('Числовые данные нормализованы')
        except Exception as e:
            logger.warning(f'Ошибка нормализации числовых данных: {e}')
            return None

    def split_train_data(self):
        """
        Разделение данных на обучающую и тестовую выборки.
        :return: X_train, X_test, y_train, y_test.
        """
        try:
            X = self.train_data.drop('result', axis=1)
            y = self.train_data['result']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            logger.debug('Данные разделены на обучающую и тестовую выборки')
            self.save_train_data(X_train, X_test, y_train, y_test)
        except Exception as e:
            logger.warning(f'Ошибка разделения данных: {e}')
            return None, None, None, None
    
    def save_train_data(self, X_train, X_test, y_train, y_test):
        """
        Сохранение данных в JSON-файлы.
        :param X_train: Обучающая выборка признаков.
        :param X_test: Тестовая выборка признаков.
        :param y_train: Обучающая выборка целевой переменной.
        :param y_test: Тестовая выборка целевой переменной.
        """
        try:
            X_train.to_json(self.train_data_dir / 'X_train.json', orient='records', indent=4)
            X_test.to_json(self.train_data_dir / 'X_test.json', orient='records', indent=4)
            y_train.to_json(self.train_data_dir / 'y_train.json', orient='records', indent=4)
            y_test.to_json(self.train_data_dir / 'y_test.json', orient='records', indent=4)
            logger.debug('Данные сохранены в JSON-файлы')
        except Exception as e:
            logger.warning(f'Ошибка сохранения данных: {e}')

    def prepare_train_data(self):
        """
        Подготовка данных.
        """
        logger.info('Начало подготовки данных')
        self.get_train_data()
        self.normalize_numerical_data()
        self.encode_categorical_data()       
        self.clean_data()
        self.split_train_data()
        logger.debug("Подготовка данных завершена")

    def fetch_match_data(self, home_team, away_team, date):
        """
        Подготовка данных для предсказания результата матча между домашней и гостевой командами.

        :param home_team: Название домашней команды.
        :param away_team: Название гостевой команды.
        :param date: Дата матча.
        :return: Кортеж (сырые данные, закодированные данные) или None в случае ошибки.
        """
        # Преобразование даты в datetime, если это необходимо
        date = pd.to_datetime(date, errors='coerce')
        
        if pd.isnull(date):
            logger.error("Некорректная дата.")
            return None
        
        # Получаем статистику из кэша или вычисляем
        if (home_team, date) not in self._team_stats_cache:
            home_stats = self.calculate_team_stats(home_team, date)
        else:
            home_stats = self._team_stats_cache[(home_team, date)]
        
        if (away_team, date) not in self._team_stats_cache:
            away_stats = self.calculate_team_stats(away_team, date)
        else:
            away_stats = self._team_stats_cache[(away_team, date)]
        
        if not home_stats or not away_stats:
            logger.error("Не удалось получить статистику для одной из команд.")
            return None
        
        # Создание записи данных
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
        }
        
        # Быстрое добавление статистик команд с префиксами
        match_data.update({f'home_{key}': value for key, value in home_stats.items()})
        match_data.update({f'away_{key}': value for key, value in away_stats.items()})

        # Добавление разницы в днях между текущим и предыдущим матчем (TODO: check this because has some problems видимо добавляется два раза)
        # match_data['days_since_home_last_match'] = self.days_since_last_match(date, self._matches_by_team.get(home_team, pd.DataFrame()))
        # match_data['days_since_away_last_match'] = self.days_since_last_match(date, self._matches_by_team.get(away_team, pd.DataFrame()))
        
        # Преобразование данных в DataFrame для применения кодирования и нормализации
        match_data_df = pd.DataFrame([match_data])
        
        encoded_match_data_df = self.encode_match_data(match_data_df)
    
        if encoded_match_data_df is None:
            return None
        
        return match_data_df, encoded_match_data_df

    def encode_match_data(self, match_data_df):
        """
        Преобразует сырые данные матча в закодированный формат для модели.
        
        :param match_data_df: DataFrame с сырыми данными матча
        :return: DataFrame с закодированными данными или None в случае ошибки
        """
        try:
            encoder = joblib.load(self.encoder_path)
            scaler = joblib.load(self.scaler_path)
        except Exception as e:
            logger.warning(f'Ошибка загрузки моделей: {e}')
            return None
        
        # Применение кодирования категориальных переменных
        categorical_cols = [col for col in match_data_df.columns if col in self.categorical_cols]
        encoded_data = pd.DataFrame(encoder.transform(match_data_df[categorical_cols]), 
                                columns=encoder.get_feature_names_out(self.categorical_cols))
        encoded_match_data_df = pd.concat([match_data_df.drop(categorical_cols, axis=1), encoded_data], axis=1)
        
        # Применение нормализации числовых данных
        actual_numerical_cols = [col for col in self.numerical_cols if col in match_data_df.columns]
        encoded_match_data_df[actual_numerical_cols] = scaler.transform(match_data_df[actual_numerical_cols])

        return encoded_match_data_df

if __name__ == "__main__":
    dp = DataPrepare()

    # Подготовка данных для обучения
    dp.prepare_train_data()

    # Пример получения данных для матча
    home_team = "Manchester United"
    away_team = "Liverpool"
    date = datetime(2025, 3, 1)
    match_data, encoded_match_data = dp.fetch_match_data(home_team, away_team, date)
    logger.debug(f"Fetched match data: {match_data}")