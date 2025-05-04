import sqlite3
import os
import datetime
from src.logger import logger

class DataBaseManager:
    def __init__(self, db_name='football_expert.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        # Таблица команд
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL UNIQUE
        )
        """)
        
        # Таблица лиг
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS leagues (
            league_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL UNIQUE
        )
        """)
        
        # Таблица матчей
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            match_date DATE NOT NULL,
            league_id INTEGER NOT NULL,
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (league_id) REFERENCES leagues(league_id)
        )
        """)
        
        # Таблица статистики (разделена на домашнюю и выездную)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_stats (
            stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
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
            overall_rating INT,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
        """)
        
        # Таблица предсказаний
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            home_win_prob DECIMAL(4,2),
            draw_prob DECIMAL(4,2),
            away_win_prob DECIMAL(4,2),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
        """)
        
        # Таблица результатов
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            prediction_id INTEGER,
            is_home_win BOOLEAN,
            is_draw BOOLEAN,
            is_away_win BOOLEAN,
            FOREIGN KEY (match_id) REFERENCES matches(match_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
            CHECK (is_home_win + is_draw + is_away_win = 1)  -- только один может быть TRUE
        )
        """)
        
        self.conn.commit()
        
    def insert_data(self, table_name, data):
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cursor.execute(query, data)
        self.conn.commit()

    def fetch_data(self, table_name, condition=""):
        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()
        
    def get_db_path(self):
        """Возвращает абсолютный путь к файлу базы данных"""
        return os.path.abspath(self.db_name)
    
    def get_teams_from_db(self):
        """Получает список всех команд из базы данных"""
        teams = self.fetch_data('teams')
        return [team[1] for team in teams] 

    def get_leagues_from_db(self):
        """Получает список всех лиг из базы данных"""
        leagues = self.fetch_data('leagues')
        return [league[1] for league in leagues]  # league[1] - это имя лиги

    def get_upcoming_matches_from_db(self, league=None):
        """Получает предстоящие матчи из базы данных"""
        today = datetime.date.today().strftime('%Y-%m-%d')
        query = f"""
        SELECT t1.name as home_team, t2.name as away_team, m.match_date, l.name as league
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.team_id
        JOIN teams t2 ON m.away_team_id = t2.team_id
        JOIN leagues l ON m.league_id = l.league_id
        WHERE m.match_date >= '{today}'
        """
        if league:
            query += f" AND l.name = '{league}'"
        
        self.cursor.execute(query)
        matches = self.cursor.fetchall()

        return [
            {
                'home_team': match[0],
                'away_team': match[1],
                'date': match[2],
                'league': match[3]
            }
            for match in matches
        ]

    def save_prediction_to_db(self, home_team, away_team, match_date, predictions):
        """Сохраняет предсказание в базу данных"""
        try:
            # Получаем ID матча
            match_id = self.fetch_data(
                'matches',
                f"home_team_id = (SELECT team_id FROM teams WHERE name = '{home_team}') "
                f"AND away_team_id = (SELECT team_id FROM teams WHERE name = '{away_team}') "
                f"AND match_date = '{match_date}'"
            )
            
            if not match_id:
                logger.warning(f"Матч {home_team} vs {away_team} не найден в базе данных")
                return False
            
            match_id = match_id[0][0]
            
            # Вставляем предсказание
            self.insert_data(
                'predictions',
                (None, match_id, predictions['home_win'], predictions['draw'], predictions['away_win'])
            )
            
            logger.info(f"Предсказание для матча {home_team} vs {away_team} сохранено в базу данных")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении предсказания: {e}")
            return False