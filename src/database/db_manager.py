import sqlite3

class DataBaseMananger:
    def __init__(self, db_name='football_expert.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        
    def create_table(self, table_name, columns):
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.cursor.execute(query)
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