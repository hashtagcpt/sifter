import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "sifter.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Universe table
    c.execute('''
        CREATE TABLE IF NOT EXISTS universe (
            ticker TEXT PRIMARY KEY
        )
    ''')
    
    # Bayesian views table
    c.execute('''
        CREATE TABLE IF NOT EXISTS bayesian_views (
            ticker TEXT PRIMARY KEY,
            subjective_prob REAL,
            confidence REAL,
            updated_at TIMESTAMP
        )
    ''')
    
    # Analyst ratings history
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyst_ratings_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            strong_buy INT,
            buy INT,
            hold INT,
            sell INT,
            strong_sell INT,
            rating_date TEXT,
            fetched_at TIMESTAMP
        )
    ''')
    
    # AI Macro scores history
    c.execute('''
        CREATE TABLE IF NOT EXISTS ai_macro_scores_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT,
            score REAL,
            news_context TEXT,
            fetched_at TIMESTAMP
        )
    ''')
    
    # Fundamental cache
    c.execute('''
        CREATE TABLE IF NOT EXISTS fundamental_cache (
            ticker TEXT,
            data_type TEXT,
            data_json TEXT,
            updated_at TIMESTAMP,
            PRIMARY KEY (ticker, data_type)
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Universe ---
def save_universe(tickers: List[str]):
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM universe')
    c.executemany('INSERT INTO universe (ticker) VALUES (?)', [(t,) for t in tickers])
    conn.commit()
    conn.close()

def get_universe() -> List[str]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT ticker FROM universe')
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

# --- Bayesian Views ---
def save_bayesian_view(ticker: str, prob: float, conf: float):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''
        INSERT OR REPLACE INTO bayesian_views (ticker, subjective_prob, confidence, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (ticker, prob, conf, now))
    conn.commit()
    conn.close()

def get_bayesian_views() -> tuple[dict, dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT ticker, subjective_prob, confidence FROM bayesian_views')
    rows = c.fetchall()
    conn.close()
    
    views = {}
    confidences = {}
    for r in rows:
        views[r[0]] = r[1]
        confidences[r[0]] = r[2]
    return views, confidences

# --- Analyst Ratings ---
def save_analyst_rating(ticker: str, data: dict):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''
        INSERT INTO analyst_ratings_history 
        (ticker, strong_buy, buy, hold, sell, strong_sell, rating_date, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker, 
        data.get("strongBuy", 0), 
        data.get("buy", 0), 
        data.get("hold", 0), 
        data.get("sell", 0), 
        data.get("strongSell", 0), 
        data.get("date", ""), 
        now
    ))
    conn.commit()
    conn.close()

def get_latest_analyst_rating(ticker: str) -> Optional[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT strong_buy, buy, hold, sell, strong_sell, rating_date 
        FROM analyst_ratings_history 
        WHERE ticker = ? 
        ORDER BY fetched_at DESC LIMIT 1
    ''', (ticker,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "strongBuy": row[0],
            "buy": row[1],
            "hold": row[2],
            "sell": row[3],
            "strongSell": row[4],
            "date": row[5]
        }
    return None

def get_analyst_rating_history(ticker: str) -> List[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT strong_buy, buy, hold, sell, strong_sell, rating_date, fetched_at 
        FROM analyst_ratings_history 
        WHERE ticker = ? 
        ORDER BY fetched_at ASC
    ''', (ticker,))
    rows = c.fetchall()
    conn.close()
    return [{
        "strongBuy": r[0],
        "buy": r[1],
        "hold": r[2],
        "sell": r[3],
        "strongSell": r[4],
        "date": r[5],
        "fetched_at": r[6]
    } for r in rows]

# --- AI Macro Scores ---
def save_ai_macro_scores(sector_scores: dict, news_context: str):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    for sector, score in sector_scores.items():
        c.execute('''
            INSERT INTO ai_macro_scores_history (sector, score, news_context, fetched_at)
            VALUES (?, ?, ?, ?)
        ''', (sector, float(score), news_context, now))
    conn.commit()
    conn.close()

def get_ai_macro_score_history(sector: str) -> List[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT score, news_context, fetched_at 
        FROM ai_macro_scores_history 
        WHERE sector = ? 
        ORDER BY fetched_at ASC
    ''', (sector,))
    rows = c.fetchall()
    conn.close()
    return [{"score": r[0], "news_context": r[1], "fetched_at": r[2]} for r in rows]

# --- Fundamental Cache ---
def get_fundamental_cache(ticker: str, data_type: str, ttl_hours: int = 24) -> Optional[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT data_json, updated_at 
        FROM fundamental_cache 
        WHERE ticker = ? AND data_type = ?
    ''', (ticker, data_type))
    row = c.fetchone()
    conn.close()
    
    if row:
        updated_at = datetime.fromisoformat(row[1])
        if (datetime.now() - updated_at).total_seconds() < ttl_hours * 3600:
            return json.loads(row[0])
    return None

def save_fundamental_cache(ticker: str, data_type: str, data: dict):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''
        INSERT OR REPLACE INTO fundamental_cache (ticker, data_type, data_json, updated_at)
        VALUES (?, ?, ?, ?)
    ''', (ticker, data_type, json.dumps(data), now))
    conn.commit()
    conn.close()
