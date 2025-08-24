# test_server.py - Minimal Flask test
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello! Flask server is running!"

if __name__ == '__main__':
    print("🌍 Starting test server...")
    app.run(host='127.0.0.1', port=5000, debug=False)