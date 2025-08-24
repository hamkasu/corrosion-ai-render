import sqlite3

conn = sqlite3.connect('corrosion.db')
c = conn.cursor()
c.execute("PRAGMA table_info(detections)")
columns = [col[1] for col in c.fetchall()]
if 'timestamp' not in columns:
    c.execute("ALTER TABLE detections ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
    print("✅ Added timestamp column")
conn.commit()
conn.close()