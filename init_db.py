import sqlite3
conn = sqlite3.connect('corrosion.db')
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
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        comments TEXT DEFAULT '',
        custom_name TEXT DEFAULT ''
    )
''')
conn.commit()
conn.close()
print("✅ Database created!")