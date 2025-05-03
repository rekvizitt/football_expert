import sqlite3

class DataBaseManager:
    def __init__(self, db_name='football_expert.db'):
        self.conn = sqlite3.connect(db_name)
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