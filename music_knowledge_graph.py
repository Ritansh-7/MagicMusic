import sqlite3

DB_PATH = "music_knowledge.db"

def init_graph():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS song_theme (song_id TEXT, theme TEXT, confidence REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS song_intent (song_id TEXT, intent TEXT, confidence REAL)")
    conn.commit()
    conn.close()

def add_song_theme(song_id: str, theme: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO song_theme VALUES (?, ?, ?)", (song_id, theme, confidence))
    conn.commit()
    conn.close()