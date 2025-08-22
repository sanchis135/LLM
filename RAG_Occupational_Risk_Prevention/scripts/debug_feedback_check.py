# debug_feedback_check.py
from monitoring.logger import DB_PATH
import sqlite3, pathlib

db = pathlib.Path(DB_PATH)
print("DB:", db.resolve())
with sqlite3.connect(str(db)) as conn:
    for row in conn.execute("SELECT id, feedback, feedback_text FROM interactions ORDER BY id DESC LIMIT 10"):
        print(row)