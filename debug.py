# Run this in Python to debug
import sqlite3
conn = sqlite3.connect('corrosion.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT * FROM detections")
rows = c.fetchall()
for r in rows:
    print(dict(r))