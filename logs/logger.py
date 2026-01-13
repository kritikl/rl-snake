import sqlite3
import datetime
import os


class BaseLogger:
    """Base class for SQLite loggers."""
    def __init__(self, db_path, table_name):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """To be implemented in subclasses."""
        raise NotImplementedError

    def close(self):
        """Close the database connection."""
        self.conn.close()


class TrainingLogger(BaseLogger):
    """Logs detailed training metrics for each episode."""
    def __init__(self, db_path="logs/training.db"):
        super().__init__(db_path, "training_logs")

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                episode INTEGER,
                reward REAL,
                score INTEGER,
                loops_detected INTEGER,
                missed_foods INTEGER,
                length INTEGER,
                total_steps INTEGER
            )
        """)
        self.conn.commit()

    def log(self, model_name, episode, reward, score, loops_detected, missed_foods, length, total_steps):
        """Log one training episode."""
        timestamp = datetime.datetime.now().isoformat()
        self.cursor.execute("""
            INSERT INTO training_logs 
                (timestamp, model_name, episode, reward, score, loops_detected, missed_foods, length, total_steps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, model_name, episode, reward, score, loops_detected, missed_foods, length, total_steps))
        self.conn.commit()


class EvaluationLogger(BaseLogger):
    """Logs periodic evaluation results."""
    def __init__(self, db_path="logs/evaluation.db"):
        super().__init__(db_path, "evaluation_logs")

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                mean_reward REAL,
                std_reward REAL,
                episodes_evaluated INTEGER
            )
        """)
        self.conn.commit()

    def log(self, model_name, mean_reward, std_reward, episodes_evaluated):
        """Log an evaluation summary."""
        timestamp = datetime.datetime.now().isoformat()
        self.cursor.execute("""
            INSERT INTO evaluation_logs 
                (timestamp, model_name, mean_reward, std_reward, episodes_evaluated)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, model_name, mean_reward, std_reward, episodes_evaluated))
        self.conn.commit()
