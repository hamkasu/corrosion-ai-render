import sqlite3

try:
    conn = sqlite3.connect('corrosion.db')
    c = conn.cursor()
    c.execute("PRAGMA table_info(detections)")
    cols = c.fetchall()
    print("✅ Database exists. Columns:")
    for col in cols:
        print(f" - {col[1]} ({col[2]})")
    conn.close()
except Exception as e:
    print("❌ DB Error:", str(e))