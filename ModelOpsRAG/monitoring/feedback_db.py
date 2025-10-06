# monitoring/feedback_db.py
from pathlib import Path
import sqlite3, json, time

class FeedbackStore:
    def __init__(self, db_path: str):
        self.path = Path(db_path)
        # Asegura que exista el directorio
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Inicializa la tabla
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                event TEXT NOT NULL,
                payload TEXT
            )
            """)

    def record(self, event: str, payload: dict | None = None):
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT INTO events (ts, event, payload) VALUES (?, ?, ?)",
                (time.time(), event, json.dumps(payload or {}))
            )
