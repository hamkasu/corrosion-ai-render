# auth.py - User authentication
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def check_password(pw_hash, password):
        return check_password_hash(pw_hash, password)

# In-memory user (use database in production)
users = [
    User(id=1, username="admin", password_hash=generate_password_hash("admin123"))
]

def get_user(username):
    return next((u for u in users if u.username == username), None)