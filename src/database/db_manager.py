import csv
import io
import json
import sqlite3
import os
from src.logger import logger
import src.data.utils as utils
from src.data.prepare_data import DataPrepare
from contextlib import contextmanager
from datetime import datetime
import shutil

dp = DataPrepare()

class DataBaseManager:
    def __init__(self, db_path='football_expert.db'):
        self.db_path = db_path
        self._initialize_db()
   
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
   
    def _initialize_db(self):
        with self._get_connection() as conn:
            # Таблица лиг
            conn.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                country TEXT
            )
            """)
            
            # Таблица команд
            conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                league_id INTEGER REFERENCES leagues(league_id)
            )
            """)
           
            # Таблица матчей
            conn.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                league_id INTEGER NOT NULL REFERENCES leagues(league_id),
                match_date TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'upcoming'
            )
            """)
           
            # Таблица статистики (разделена на домашнюю и выездную)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS match_stats (
                stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL REFERENCES matches(match_id),
                team_id INTEGER NOT NULL REFERENCES teams(team_id),
                is_home BOOLEAN NOT NULL,
                goals_scored_last5 INT,
                goals_conceded_last5 INT,
                xg_last5 DECIMAL(3,2),
                wins_last5 INT,
                draws_last5 INT,
                losses_last5 INT,
                possession DECIMAL(4,2),
                ga_per90 DECIMAL(4,2),
                performance_ga INT,
                xg_total DECIMAL(5,1),
                xag_total DECIMAL(5,1),
                prgc INT,
                prgp INT,
                attack_rating INT,
                midfield_rating INT,
                defense_rating INT,
                overall_rating INT
            )
            """)
           
            # Таблица предсказаний
            conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL REFERENCES matches(match_id),
                home_win_prob DECIMAL(5,4) NOT NULL,
                draw_prob DECIMAL(5,4) NOT NULL,
                away_win_prob DECIMAL(5,4) NOT NULL,
                predicted_winner TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)
           
            conn.commit()

    def get_tables_info(self):
        """Получение информации о таблицах в базе данных"""
        tables_info = []
        with self._get_connection() as conn:
            # Получаем список всех таблиц
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                # Количество записей
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                records_count = cursor.fetchone()[0]
                
                # Приблизительный размер таблицы (примерная оценка)
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Получаем первую строку для оценки размера
                sample_query = f"SELECT * FROM {table} LIMIT 1"
                try:
                    cursor = conn.execute(sample_query)
                    row = cursor.fetchone()
                    row_size = 0
                    if row:
                        for i, value in enumerate(row):
                            if value is not None:
                                row_size += len(str(value))
                except sqlite3.Error:
                    row_size = 100  # Если не удалось получить, используем примерное значение
                
                # Оценка размера таблицы
                estimated_size = (row_size * records_count) / 1024  # в КБ
                
                tables_info.append({
                    'name': table,
                    'records': records_count,
                    'size': round(estimated_size, 2)
                })
        
        return tables_info

    def get_total_tables_count(self):
        """Получение общего количества таблиц в базе данных"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
            return cursor.fetchone()[0]

    def get_total_records_count(self):
        """Получение общего количества записей в базе данных"""
        total_records = 0
        with self._get_connection() as conn:
            # Получаем список всех таблиц
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    records_count = cursor.fetchone()[0]
                    total_records += records_count
                except sqlite3.Error as e:
                    logger.error(f"Error counting records in {table}: {e}")
        
        return total_records

    def backup_database(self):
        """Создание резервной копии базы данных"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{os.path.splitext(self.db_path)[0]}_backup_{timestamp}.db"
            
            # Создаем копию файла
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backup created at {backup_path}")
            return {'success': True, 'backup_path': backup_path}
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_database(self):
        """Оптимизация базы данных"""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                conn.execute("PRAGMA optimize")
                conn.commit()
            
            logger.info("Database optimized successfully")
            return {'success': True}
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return {'success': False, 'error': str(e)}

    def get_table_data(self, table_name, limit=100):
        """Получение данных из таблицы"""
        try:
            with self._get_connection() as conn:
                # Проверка существования таблицы
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cursor.fetchone():
                    return {'success': False, 'error': f"Table '{table_name}' does not exist"}
                
                # Получение структуры таблицы
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = [{'name': col[1], 'type': col[2]} for col in cursor.fetchall()]
                
                # Получение данных
                cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                rows = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'success': True,
                    'columns': columns,
                    'data': rows,
                    'total_records': self._count_records(table_name)
                }
        except Exception as e:
            logger.error(f"Error getting data from table {table_name}: {e}")
            return {'success': False, 'error': str(e)}

    def _count_records(self, table_name):
        """Подсчет количества записей в таблице"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cursor.fetchone()[0]
        except Exception:
            return 0

    def get_match_date_or_today(self, home_team, away_team):
        upcoming_matches = dp.match_results
        match = upcoming_matches[
            (upcoming_matches['home_team'] == home_team) &
            (upcoming_matches['away_team'] == away_team)
        ]
        
        if not match.empty:
            return match.iloc[0]['date'].date()
        else:
            return datetime.date.today()

    def fill_database(self, custom_data=False):
        """Заполнение базы данных тестовыми данными с логированием"""
        logger.debug("Starting database fill process")
        try:
            with self._get_connection() as conn:
                if not custom_data:
                    # Заполняем таблицу лиг
                    leagues_data = [
                        (1, "ENG-Premier League", "England"),
                        (2, "ESP-La Liga", "Spain"),
                        (3, "FRA-Ligue 1", "France"),
                        (4, "GER-Bundesliga", "Germany"),
                        (5, "ITA-Serie A", "Italy")
                    ]
                    
                    logger.debug(f"Preparing to insert leagues data: {leagues_data}")
                    
                    for league in leagues_data:
                        conn.execute(
                            "INSERT OR IGNORE INTO leagues (league_id, name, country) VALUES (?, ?, ?)",
                            league
                        )
                    league_name_to_id = {name: id for id, name, _ in leagues_data}
                    logger.debug(f"Created league name to ID mapping: {league_name_to_id}")
                
                    # Получаем словарь команд и лиг (где ключи - названия лиг)
                    logger.debug("Fetching team-league dictionary from utils")
                    teams_leagues_dict = utils.create_team_league_dict()
                    logger.debug(f"Received teams_leagues_dict: {teams_leagues_dict}")
                    
                    # Заполняем таблицу команд на основе teams_leagues_dict
                    teams_data = []
                    for team_name, league_name in teams_leagues_dict.items():
                        if league_name in league_name_to_id:
                            league_id = league_name_to_id[league_name]
                            teams_data.append((team_name, league_id))

                    logger.debug(f"Preparing to insert {len(teams_data)} teams")
                    
                    for team_name, league_id in teams_data:
                        conn.execute(
                            "INSERT OR IGNORE INTO teams (name, league_id) VALUES (?, ?)",
                            (team_name, league_id)
                        )
                    
                    # Получаем данные о завершенных матчах
                    logger.debug("Fetching train data from dp")
                    dp.get_train_data()
                    all_finished_matches = dp.train_data
                    # Получаем ID команд из базы
                    cursor = conn.execute("SELECT team_id, name, league_id FROM teams")
                    teams = {row[1]: {"id": row[0], "league_id": row[2]} for row in cursor.fetchall()}
                    logger.debug(f"Fetched teams from DB: {teams}")
                    
                    # Создаем список матчей на основе all_finished_matches (первые 10 для примера)
                    matches_to_add = all_finished_matches.head(10)
                    logger.debug(f"Preparing to process {len(matches_to_add)} matches")
                    logger.debug(f"Matches columns: {matches_to_add.columns}")
                    
                    for i, match in enumerate(matches_to_add.iterrows(), 1):
                        _, match_data = match
                        home_team = match_data['home_team']
                        away_team = match_data['away_team']
                        match_date = self.get_match_date_or_today(home_team, away_team)
                        
                        logger.debug(f"\nProcessing match {i}/{len(matches_to_add)}: {home_team} vs {away_team} on {match_date}")
                        
                        if home_team in teams and away_team in teams:
                            home_id = teams[home_team]["id"]
                            away_id = teams[away_team]["id"]
                            league_id = teams[home_team]["league_id"]
                            
                            logger.debug(f"Team IDs - Home: {home_id}, Away: {away_id}, League: {league_id}")
                            
                            # Получаем данные матча
                            logger.debug(f"Fetching match data for {home_team} vs {away_team}")
                            match_data_df, _ = dp.fetch_match_data(home_team, away_team, match_date)
                            
                            if match_data_df is not None and not match_data_df.empty:
                                logger.debug(f"Successfully fetched match data with {len(match_data_df)} rows")
                                
                                # Добавляем матч
                                cursor = conn.execute(
                                    """INSERT INTO matches 
                                    (home_team_id, away_team_id, league_id, match_date) 
                                    VALUES (?, ?, ?, ?)""",
                                    (home_id, away_id, league_id, match_date)
                                )
                                match_id = cursor.lastrowid
                                logger.debug(f"Inserted match with ID: {match_id}")
                                
                                # Извлекаем статистику из match_data_df
                                if len(match_data_df) > 0:
                                    match_data = match_data_df.iloc[0].to_dict()
                                    
                                   # Extract home team stats
                                    home_stats = {
                                        'goals_scored_last5': match_data['home_goals_last_5'],
                                        'goals_conceded_last5': match_data['home_conceded_goals_last_5'],
                                        'xg_last5': match_data['home_xg_last_5'],
                                        'wins_last5': match_data['home_wins_last_5'],
                                        'draws_last5': match_data['home_draws_last_5'],
                                        'losses_last5': match_data['home_losses_last_5'],
                                        'possession': match_data['home_possession'],
                                        'ga_per90': match_data['home_ga_per_90'],
                                        'performance_ga': match_data['home_performance_ga'],
                                        'xg_total': match_data['home_xg'],
                                        'xag_total': match_data['home_xag'],
                                        'prgc': match_data['home_prgc'],
                                        'prgp': match_data['home_prgp'],
                                        'attack_rating': match_data['home_attack_rating'],
                                        'midfield_rating': match_data['home_midfield_rating'],
                                        'defense_rating': match_data['home_defence_rating'],
                                        'overall_rating': match_data['home_overall_rating'],
                                        'days_since_last_match': match_data['home_days_since_last_match'],
                                    }

                                    # Extract away team stats
                                    away_stats = {
                                        'goals_scored_last5': match_data['away_goals_last_5'],
                                        'goals_conceded_last5': match_data['away_conceded_goals_last_5'],
                                        'xg_last5': match_data['away_xg_last_5'],
                                        'wins_last5': match_data['away_wins_last_5'],
                                        'draws_last5': match_data['away_draws_last_5'],
                                        'losses_last5': match_data['away_losses_last_5'],
                                        'possession': match_data['away_possession'],
                                        'ga_per90': match_data['away_ga_per_90'],
                                        'performance_ga': match_data['away_performance_ga'],
                                        'xg_total': match_data['away_xg'],
                                        'xag_total': match_data['away_xag'],
                                        'prgc': match_data['away_prgc'],
                                        'prgp': match_data['away_prgp'],
                                        'attack_rating': match_data['away_attack_rating'],
                                        'midfield_rating': match_data['away_midfield_rating'],
                                        'defense_rating': match_data['away_defence_rating'],
                                        'overall_rating': match_data['away_overall_rating'],
                                        'days_since_last_match': match_data['away_days_since_last_match'],
                                    }

                                    
                                    logger.debug(f"Home stats: {home_stats}")
                                    logger.debug(f"Away stats: {away_stats}")
                                else:
                                    logger.debug("No match data available")

                                
                                # Добавляем статистику для домашней команды
                                conn.execute(
                                    """INSERT INTO match_stats 
                                    (match_id, team_id, is_home, goals_scored_last5, goals_conceded_last5, 
                                        xg_last5, wins_last5, draws_last5, losses_last5, possession, 
                                        ga_per90, performance_ga, xg_total, xag_total, prgc, prgp, 
                                        attack_rating, midfield_rating, defense_rating, overall_rating) 
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        match_id, home_id, True,
                                        home_stats.get('goals_scored_last5', 0),
                                        home_stats.get('goals_conceded_last5', 0),
                                        home_stats.get('xg_last5', 0),
                                        home_stats.get('wins_last5', 0),
                                        home_stats.get('draws_last5', 0),
                                        home_stats.get('losses_last5', 0),
                                        home_stats.get('possession', 50),
                                        home_stats.get('ga_per90', 1.0),
                                        home_stats.get('performance_ga', 75),
                                        home_stats.get('xg_total', 0),
                                        home_stats.get('xag_total', 0),
                                        home_stats.get('prgc', 0),
                                        home_stats.get('prgp', 0),
                                        home_stats.get('attack_rating', 75),
                                        home_stats.get('midfield_rating', 75),
                                        home_stats.get('defense_rating', 75),
                                        home_stats.get('overall_rating', 75)
                                    )
                                )
                                logger.debug("Inserted home team stats")
                                
                                # Добавляем статистику для выездной команды
                                conn.execute(
                                    """INSERT INTO match_stats 
                                    (match_id, team_id, is_home, goals_scored_last5, goals_conceded_last5, 
                                        xg_last5, wins_last5, draws_last5, losses_last5, possession, 
                                        ga_per90, performance_ga, xg_total, xag_total, prgc, prgp, 
                                        attack_rating, midfield_rating, defense_rating, overall_rating) 
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        match_id, away_id, False,
                                        away_stats.get('goals_scored_last5', 0),
                                        away_stats.get('goals_conceded_last5', 0),
                                        away_stats.get('xg_last5', 0),
                                        away_stats.get('wins_last5', 0),
                                        away_stats.get('draws_last5', 0),
                                        away_stats.get('losses_last5', 0),
                                        away_stats.get('possession', 50),
                                        away_stats.get('ga_per90', 1.0),
                                        away_stats.get('performance_ga', 75),
                                        away_stats.get('xg_total', 0),
                                        away_stats.get('xag_total', 0),
                                        away_stats.get('prgc', 0),
                                        away_stats.get('prgp', 0),
                                        away_stats.get('attack_rating', 75),
                                        away_stats.get('midfield_rating', 75),
                                        away_stats.get('defense_rating', 75),
                                        away_stats.get('overall_rating', 75)
                                    )
                                )
                                logger.debug("Inserted away team stats")
                                
                                # # Добавляем предсказание для матча (примерные значения)
                                # predicted_winner = "home" if home_stats.get('overall_rating', 0) > away_stats.get('overall_rating', 0) else "away"
                                # conn.execute(
                                #     """INSERT INTO predictions 
                                #     (match_id, home_win_prob, draw_prob, away_win_prob, predicted_winner) 
                                #     VALUES (?, ?, ?, ?, ?)""",
                                #     (match_id, 0.45, 0.30, 0.25, predicted_winner)
                                # )
                                # logger.debug(f"Inserted prediction with winner: {predicted_winner}")
                            else:
                                logger.warning(f"No match data found for {home_team} vs {away_team}")
                        else:
                            logger.warning(f"Teams not found in database: {home_team} and/or {away_team}")
                else:
                    data_type, data_content = custom_data
                    logger.debug(f"Processing custom data of type: {data_type}")
                    
                    if data_type == 'json':
                        try:
                            data = json.loads(data_content)
                            logger.debug(f"Parsed JSON data: {data}")
                            
                            for table_name, rows in data.items():
                                if not isinstance(rows, list):
                                    logger.warning(f"Invalid data format for table {table_name}, expected list of dicts")
                                    continue
                                    
                                if not rows:
                                    logger.debug(f"No data to insert for table {table_name}")
                                    continue
                                    
                                # Получаем список колонок из первого элемента
                                columns = list(rows[0].keys())
                                columns_str = ', '.join(columns)
                                placeholders = ', '.join(['?'] * len(columns))
                                
                                logger.debug(f"Preparing to insert into {table_name} with columns: {columns_str}")
                                
                                # Вставляем данные
                                for row in rows:
                                    values = [row.get(col) for col in columns]
                                    conn.execute(
                                        f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                                        values
                                    )
                                logger.debug(f"Inserted {len(rows)} rows into {table_name}")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON data: {e}")
                            return {'success': False, 'error': f"Invalid JSON format: {str(e)}"}
                            
                    elif data_type == 'csv':
                        try:
                            # Читаем CSV данные
                            csv_data = io.StringIO(data_content)
                            reader = csv.DictReader(csv_data)
                            
                            # Группируем данные по таблицам (предполагаем, что имя таблицы в первом столбце)
                            data_by_table = {}
                            for row in reader:
                                table_name = row.get('table_name')
                                if not table_name:
                                    logger.warning("CSV row missing 'table_name' field")
                                    continue
                                    
                                if table_name not in data_by_table:
                                    data_by_table[table_name] = []
                                
                                # Удаляем table_name из данных
                                row_data = {k: v for k, v in row.items() if k != 'table_name'}
                                data_by_table[table_name].append(row_data)
                            
                            # Вставляем данные в соответствующие таблицы
                            for table_name, rows in data_by_table.items():
                                if not rows:
                                    continue
                                    
                                columns = list(rows[0].keys())
                                columns_str = ', '.join(columns)
                                placeholders = ', '.join(['?'] * len(columns))
                                
                                logger.debug(f"Preparing to insert into {table_name} with columns: {columns_str}")
                                
                                for row in rows:
                                    values = [row.get(col) for col in columns]
                                    conn.execute(
                                        f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                                        values
                                    )
                                logger.debug(f"Inserted {len(rows)} rows into {table_name}")
                                
                        except Exception as e:
                            logger.error(f"Error processing CSV data: {e}")
                            return {'success': False, 'error': f"CSV processing error: {str(e)}"}
                            
                    else:
                        logger.error(f"Unsupported data type: {data_type}")
                        return {'success': False, 'error': f"Unsupported data type: {data_type}"}
                    
                conn.commit()
                logger.debug("Database fill completed successfully")
                return {'success': True}
                
        except Exception as e:
            logger.error(f"Error filling database: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
 
    def add_match_prediction(self, home_team, away_team, prediction_data):
        logger.debug(f"Adding prediction for match: {home_team} vs {away_team}")
        try:
            with self._get_connection() as conn:
                # Получаем ID матча
                cursor = conn.execute(
                    """SELECT match_id FROM matches 
                    WHERE home_team_id = (SELECT team_id FROM teams WHERE name = ?)
                    AND away_team_id = (SELECT team_id FROM teams WHERE name = ?)
                    ORDER BY match_date DESC LIMIT 1""",
                    (home_team, away_team)
                )
                match = cursor.fetchone()
                
                if not match:
                    logger.error(f"Match not found in database: {home_team} vs {away_team}")
                    return {'success': False, 'error': 'Match not found'}
                
                match_id = match[0]
                
                # Добавляем предсказание
                conn.execute(
                    """INSERT OR REPLACE INTO predictions 
                    (match_id, home_win_prob, draw_prob, away_win_prob, predicted_winner) 
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        match_id,
                        prediction_data['predictions']['home_win'],
                        prediction_data['predictions']['draw'],
                        prediction_data['predictions']['away_win'],
                        prediction_data['winner']
                    )
                )
                
                conn.commit()
                logger.debug(f"Successfully added prediction for match ID {match_id}")
                return {'success': True}
                
        except Exception as e:
            logger.error(f"Error adding match prediction: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
           
if __name__ == "__main__":
    database = DataBaseManager()
    database.fill_database()