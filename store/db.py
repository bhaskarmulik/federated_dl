from __future__ import annotations
import sqlite3, pathlib
from typing import Iterator, Tuple

def open_db(path: str | pathlib.Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute('PRAGMA journal_mode=WAL')
    return con

def init_schema(con: sqlite3.Connection) -> None:
    con.executescript('''
    CREATE TABLE IF NOT EXISTS runs(
        run_id TEXT PRIMARY KEY,
        created_at TEXT,
        config_json TEXT,
        dataset TEXT,
        topology TEXT,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS rounds(
        run_id TEXT, round_no INTEGER, start_ts TEXT, end_ts TEXT,
        selected_clients_json TEXT, committed_clients_json TEXT,
        staleness_bound INTEGER, trigger_reason TEXT
    );
    CREATE TABLE IF NOT EXISTS updates(
        run_id TEXT, round_no INTEGER, client_id TEXT, n_samples INTEGER,
        base_round INTEGER, sha256 TEXT, received_ts TEXT, accepted INTEGER, reason TEXT
    );
    ''')
    con.commit()
