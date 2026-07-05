import sqlite3

from config import DB

def get_connection():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return (conn)

def init_db():
    with get_connection() as db:
        cur = db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                lr_prediction TEXT,
                lr_fake_prob REAL,
                lr_real_prob REAL,
                lr_explanations TEXT,
                bert_prediction TEXT,
                bert_fake_prob REAL,
                bert_real_prob REAL,
                ollama_summary TEXT,
                created_at DATETIME
            );
        """)