from __future__ import annotations
import sqlite3, time, pathlib
from typing import List, Dict, Optional

DB_PATH = pathlib.Path("data/monitoring/telemetry.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- columnas esperadas en la versión actual ---
EXPECTED_COLUMNS = {
    "id", "ts_utc", "query", "retriever", "topk", "fanout",
    "latency_ms", "provider", "model", "answer",
    "sources", "ctx_len", "feedback", "feedback_text",
    # opcionales que puede que uses en métricas
    "latency_ms_retrieval", "latency_ms_llm",
}

def _table_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("PRAGMA table_info(interactions)")
    return {row[1] for row in cur.fetchall()}

def _init_and_migrate() -> None:
    with sqlite3.connect(str(DB_PATH)) as conn:
        # crea si no existe (esquema completo actual)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_utc REAL NOT NULL,
          query TEXT,
          retriever TEXT,
          topk INTEGER,
          fanout INTEGER,
          latency_ms INTEGER,
          provider TEXT,
          model TEXT,
          answer TEXT,
          sources TEXT,
          ctx_len INTEGER,
          feedback INTEGER DEFAULT 0,
          feedback_text TEXT,
          latency_ms_retrieval REAL,
          latency_ms_llm REAL
        )
        """)
        conn.commit()

        # migración: añade cualquier columna que falte
        cols = _table_columns(conn)
        def add_col(name: str, sql_type: str, default_sql: str = None):
            if name not in cols:
                conn.execute(f"ALTER TABLE interactions ADD COLUMN {name} {sql_type}")
                if default_sql is not None:
                    conn.execute(f"UPDATE interactions SET {name} = {default_sql} WHERE {name} IS NULL")
        # esenciales para tu error
        add_col("sources", "TEXT", "NULL")
        add_col("ctx_len", "INTEGER", "0")
        add_col("feedback", "INTEGER", "0")
        add_col("feedback_text", "TEXT", "NULL")
        # opcionales para métricas
        add_col("latency_ms_retrieval", "REAL", "NULL")
        add_col("latency_ms_llm", "REAL", "NULL")

        conn.commit()

_init_and_migrate()

def log_interaction(
    query: str,
    retriever: str,
    topk: int,
    fanout: int,
    latency_ms: int,
    provider: str,
    model: str,
    answer: Optional[str],
    sources: List[str],
    ctx_len: int,
    latency_ms_retrieval: Optional[float] = None,
    latency_ms_llm: Optional[float] = None,
) -> int:
    ts = time.time()
    # guardamos sources como texto simple separado por ' | '
    src = " | ".join(sources or [])
    with sqlite3.connect(str(DB_PATH)) as conn:
        cols = _table_columns(conn)

        # construimos el INSERT según columnas disponibles (por si vienes de un esquema antiguo)
        base_cols = ["ts_utc","query","retriever","topk","fanout","latency_ms","provider","model","answer"]
        base_vals = [ts, query, retriever, topk, fanout, latency_ms, provider, model, answer]

        if "sources" in cols:
            base_cols += ["sources"]
            base_vals += [src]
        if "ctx_len" in cols:
            base_cols += ["ctx_len"]
            base_vals += [ctx_len]
        if "feedback" in cols:
            base_cols += ["feedback"]
            base_vals += [0]
        if "feedback_text" in cols:
            base_cols += ["feedback_text"]
            base_vals += [None]
        if "latency_ms_retrieval" in cols:
            base_cols += ["latency_ms_retrieval"]
            base_vals += [latency_ms_retrieval]
        if "latency_ms_llm" in cols:
            base_cols += ["latency_ms_llm"]
            base_vals += [latency_ms_llm]

        placeholders = ",".join(["?"] * len(base_cols))
        sql = f"INSERT INTO interactions ({', '.join(base_cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, base_vals)
        conn.commit()
        return cur.lastrowid

def update_feedback(interaction_id: int, value: int, comment: str = "") -> int:
    value = 1 if value > 0 else (-1 if value < 0 else 0)
    with sqlite3.connect(str(DB_PATH)) as conn:
        cur = conn.execute(
            "UPDATE interactions SET feedback=?, feedback_text=? WHERE id=?",
            (value, comment or None, int(interaction_id)),
        )
        conn.commit()
        return cur.rowcount  # 1 si se actualizó, 0 si no encontró la fila
    
def get_last_interactions(n: int = 5) -> list[dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, ts_utc, query, feedback, feedback_text FROM interactions ORDER BY id DESC LIMIT ?",
            (int(n),),
        ).fetchall()
    return [dict(r) for r in rows]

def recent(limit: int = 100) -> List[Dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, ts_utc, query, retriever, topk, fanout,
                   latency_ms, provider, model,
                   COALESCE(ctx_len, 0) AS ctx_len,
                   COALESCE(feedback, 0) AS feedback,
                   feedback_text
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
    return [dict(r) for r in rows]
