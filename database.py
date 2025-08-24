# database.py
import sqlite3
import os

DB_PATH = 'corrosion.db'

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_image TEXT NOT NULL,
            result_image TEXT NOT NULL,
            result_text TEXT,
            high_severity INTEGER DEFAULT 0,
            medium_severity INTEGER DEFAULT 0,
            low_severity INTEGER DEFAULT 0,
            confirmed BOOLEAN DEFAULT FALSE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def save_detection(original_img, result_img, result_text, high, med, low):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections 
        (original_image, result_image, result_text, high_severity, medium_severity, low_severity)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (original_img, result_img, result_text, high, med, low))
    conn.commit()
    conn.close()