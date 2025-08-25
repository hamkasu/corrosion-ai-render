# app.py - Corrosion Detection AI Application

from flask import Flask, request, redirect, url_for, send_file, render_template_string
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import sqlite3
import cv2
import numpy as np
import base64
from io import BytesIO
import pytz
from datetime import datetime
import zipfile
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Flask App Setup
# ===========================
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['REPORT_FOLDER'] = 'static/reports'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'] + '/markup', exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# ===========================
# Configurable Timezone
# ===========================
DEFAULT_TIMEZONE = 'Asia/Kuala_Lumpur'
tz = pytz.timezone(DEFAULT_TIMEZONE)

# ===========================
# User Authentication
# ===========================
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def check_password(pw_hash, password):
        return check_password_hash(pw_hash, password)

users = [
    User(id=1, username="admin", password_hash=generate_password_hash("admin123"))
]

def get_user(username):
    return next((u for u in users if u.username == username), None)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return next((u for u in users if str(u.id) == user_id), None)

# ===========================
# Initialize Database
# ===========================
def init_db():
    try:
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
        
        # Check for existing columns and add if needed
        c.execute("PRAGMA table_info(detections)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'timestamp' not in columns:
            c.execute("ALTER TABLE detections ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
        if 'comments' not in columns:
            c.execute("ALTER TABLE detections ADD COLUMN comments TEXT DEFAULT ''")
        if 'custom_name' not in columns:
            c.execute("ALTER TABLE detections ADD COLUMN custom_name TEXT DEFAULT ''")
            
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing database: {str(e)}")

def init_deletion_log():
    try:
        conn = sqlite3.connect('corrosion.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS deletion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                original_image TEXT NOT NULL,
                deleted_by TEXT NOT NULL,
                deleted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (detection_id) REFERENCES detections (id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✅ Deletion log table created")
    except Exception as e:
        logger.error(f"❌ Error creating deletion log: {str(e)}")

# Initialize database
init_db()
init_deletion_log()

# ===========================
# Load Your Custom YOLO Model
# ===========================
MODEL_PATH = 'best.pt'

model = None
try:
    model = YOLO(MODEL_PATH)
    model.to('cpu')
    logger.info("✅ YOLO model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    # Fallback to Roboflow model if local model fails
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY"))
        project = rf.workspace("hamka-corrosion").project("corrosion-detection-xjdlv")
        model = project.version(1).model
        logger.info("✅ Roboflow model loaded as fallback")
    except Exception as e:
        logger.error(f"❌ Error loading Roboflow model: {str(e)}")

# ===========================
# Prediction Function
# ===========================
def predict_image(filepath):
    if model is None or not os.path.exists(filepath):
        return "no_detection.jpg", "Model not loaded", 0, 0, 0

    try:
        image = Image.open(filepath).convert("RGB")
        image = image.resize((640, 640), Image.Resampling.LANCZOS)
        results = model(
            source=image,
            conf=0.3,
            iou=0.2,
            max_det=10,
            agnostic_nms=True,
            retina_masks=True
        )

        det = results[0].boxes
        masks = results[0].masks
        if masks is None or len(det) == 0:
            return "no_detection.jpg", "No Corrosion Detected", 0, 0, 0

        mask_data = masks.data.cpu().numpy()
        scores = det.conf.cpu().numpy()
        indices = scores.argsort()[::-1]
        mask_data = mask_data[indices]

        merged_mask = mask_data[0].copy()
        for i in range(1, len(mask_data)):
            overlap = (merged_mask > 0.5) & (mask_data[i] > 0.5)
            if overlap.sum() == 0:
                merged_mask += mask_data[i]
            else:
                merged_mask = ((merged_mask + mask_data[i]) > 0).astype(np.float32)

        from torch import tensor
        merged_mask_tensor = tensor(merged_mask).unsqueeze(0)
        results[0].masks.data = merged_mask_tensor

        final_mask = (merged_mask > 0.5).astype('uint8')
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr_array = np.array(image)
        severities = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = bgr_array[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            avg_intensity = roi.mean(axis=(0,1)).mean()
            if avg_intensity < 40:
                continue
            area = cv2.contourArea(cnt)
            if area > 8000:
                severities.append("High")
            elif area > 3000:
                severities.append("Medium")
            else:
                severities.append("Low")

        high = severities.count("High")
        med = severities.count("Medium")
        low = severities.count("Low")
        num_detections = len(severities)

        if num_detections == 0:
            return "no_detection.jpg", "No Corrosion Detected", 0, 0, 0

        result_text = f"Corrosion Detected: PASS ({num_detections} spot(s))<br>Severity: High={high}, Medium={med}, Low={low}"

        result_image = results[0].plot()
        result_pil = Image.fromarray(result_image).resize(image.size)

        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_pil.save(result_path)

        return result_filename, result_text, high, med, low

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return "no_detection.jpg", "Error analyzing image", 0, 0, 0

# ===========================
# Summary Stats
# ===========================
def get_summary_stats():
    try:
        conn = sqlite3.connect('corrosion.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM detections")
        rows = c.fetchall()
        conn.close()
        total = len(rows)
        confirmed = sum(1 for r in rows if r['confirmed'])
        high = sum(r['high_severity'] for r in rows)
        med = sum(r['medium_severity'] for r in rows)
        low = sum(r['low_severity'] for r in rows)
        return {'total': total, 'confirmed': confirmed, 'high': high, 'med': med, 'low': low}
    except Exception as e:
        logger.error(f"❌ Error getting summary stats: {str(e)}")
        return {'total': 0, 'confirmed': 0, 'high': 0, 'med': 0, 'low': 0}

# ===========================
# Routes
# ===========================
@app.route('/')
def home():
    stats = get_summary_stats()
    is_authenticated = current_user.is_authenticated
    username = current_user.username if is_authenticated else ""

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📸 Corrosion Inspector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {{ --primary: #d9534f; --secondary: #5bc0de; --success: #5cb85c; --warning: #f0ad4e; --dark: #2c3e50; }}
            body {{ background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); font-family: 'Segoe UI', sans-serif; color: #333; }}
            .navbar {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); box-shadow: 0 2px 20px rgba(0,0,0,0.1); }}
            .card {{ border: none; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); transition: transform 0.3s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .btn {{ border-radius: 50px; padding: 12px 24px; font-weight: 600; }}
            .btn-primary {{ background: var(--primary); border: none; }}
            .btn-secondary {{ background: var(--secondary); border: none; }}
            .btn-success {{ background: var(--success); border: none; }}
            .btn-warning {{ background: var(--warning); border: none; }}
            .btn-outline-dark {{ color: var(--dark); border-color: var(--dark); }}
            .btn-outline-dark:hover {{ background: var(--dark) !important; color: white !important; }}
            .btn-outline-danger {{ color: #d9534f; border-color: #d9534f; }}
            .btn-outline-danger:hover {{ background: #d9534f !important; color: white !important; }}
            .stats-card {{ background: white; border-left: 5px solid var(--primary); }}
            video {{ border-radius: 15px; max-height: 300px; object-fit: cover; }}
            .feature-icon {{ font-size: 2rem; margin-bottom: 10px; color: var(--primary); }}
            .alert {{ border-radius: 10px; }}
            .comment-box {{ border: 2px solid #dee2e6; border-radius: 10px; padding: 15px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
            .comment-box textarea {{ resize: vertical; min-height: 100px; border: 1px solid #ced4da; border-radius: 8px; }}
            .img-container {{ position: relative; display: inline-block; }}
            #markupCanvas {{ position: absolute; top: 10px; left: 10px; cursor: crosshair; }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light sticky-top">
            <div class="container">
                <a class="navbar-brand d-flex align-items-center" href="/">
    			<img src="/static/images/logo.png" alt="Logo" width="40" height="40" class="me-2 rounded">
    			<div>
        		<div class="fw-bold">Calmic Sdn Bhd</div>
        		<div style="font-size: 0.8em; opacity: 0.9;">Corrosion Inspector</div>
    		</div>
	</a>
                <div class="ms-auto">
                    {''
                      f'<span class="navbar-text me-2"><i class="fas fa-user"></i> Hello, {username}</span>'
                      f'<a href="/logout" class="btn btn-outline-danger btn-sm">Logout</a>'
                    if is_authenticated else
                      '<a href="/login" class="btn btn-outline-dark btn-sm me-2">Login</a>'
                    }
                    <a href="/reports" class="btn btn-primary btn-sm">📋 Reports</a>
                </div>
            </div>
        </nav>

        <div class="container py-4">
            <div class="row mb-4">
                <div class="col-md-3 col-6">
                    <div class="card stats-card p-3 text-center">
                        <h6>Total Inspections</h6>
                        <h3 class="text-primary">{stats['total']}</h3>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="card stats-card p-3 text-center">
                        <h6>Confirmed</h6>
                        <h3 class="text-success">{stats['confirmed']}</h3>
                    </div>
                </div>
                <div class="col-md-6 col-12">
                    <div class="card stats-card p-3">
                        <h6>Severity Breakdown</h6>
                        <div class="d-flex justify-content-around">
                            <span class="badge bg-danger">High: {stats['high']}</span>
                            <span class="badge bg-warning text-dark">Medium: {stats['med']}</span>
                            <span class="badge bg-success">Low: {stats['low']}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-lg-4 col-md-6">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <div class="feature-icon"><i class="fas fa-upload"></i></div>
                            <h5 class="card-title">Upload Image</h5>
                            <p class="text-muted">Upload a single image</p>
                            <form method="POST" enctype="multipart/form-data" action="/upload" class="mt-3">
                                <input type="file" name="file" accept="image/*" required class="form-control mb-2">
                                <button type="submit" class="btn btn-primary w-100">Analyze</button>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 col-md-6">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <div class="feature-icon"><i class="fas fa-mobile-alt"></i></div>
                            <h5 class="card-title">Phone Camera</h5>
                            <p class="text-muted">Use rear camera</p>
                            <button onclick="startMobileCamera()" class="btn btn-secondary w-100 mt-3">📷 Open Camera</button>
                            <video id="mobileVideo" autoplay playsinline class="w-100 mt-2" style="display:none;"></video>
                            <div id="mobileCaptureUI" style="display:none;" class="mt-2">
                                <button onclick="captureMobilePhoto()" class="btn btn-success btn-sm me-2">📸 Take</button>
                                <button onclick="stopMobileCamera()" class="btn btn-sm">❌ Close</button>
                            </div>
                            <img id="mobilePhoto" class="w-100 mt-2" style="display:none;" />
                            <div id="mobilePhotoActions" style="display:none;" class="mt-2">
                                <button onclick="submitMobilePhoto()" class="btn btn-primary btn-sm me-2">✅ Analyze</button>
                                <button onclick="retakeMobilePhoto()" class="btn btn-sm">🔄 Retake</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 col-md-6">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <div class="feature-icon"><i class="fas fa-laptop"></i></div>
                            <h5 class="card-title">Laptop Camera</h5>
                            <p class="text-muted">Use front camera</p>
                            <button onclick="startLaptopCamera()" class="btn btn-warning w-100 mt-3">📷 Open Camera</button>
                            <video id="laptopVideo" autoplay playsinline class="w-100 mt-2" style="display:none;"></video>
                            <div id="laptopCaptureUI" style="display:none;" class="mt-2">
                                <button onclick="captureLaptopPhoto()" class="btn btn-success btn-sm me-2">📸 Take</button>
                                <button onclick="stopLaptopCamera()" class="btn btn-sm">❌ Close</button>
                            </div>
                            <img id="laptopPhoto" class="w-100 mt-2" style="display:none;" />
                            <div id="laptopPhotoActions" style="display:none;" class="mt-2">
                                <button onclick="submitLaptopPhoto()" class="btn btn-primary btn-sm me-2">✅ Analyze</button>
                                <button onclick="retakeLaptopPhoto()" class="btn btn-sm">🔄 Retake</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 col-md-6">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <div class="feature-icon"><i class="fas fa-file-pdf"></i></div>
                            <h5 class="card-title">Reports</h5>
                            <p class="text-muted">View and export</p>
                            <a href="/reports" class="btn btn-success w-100 mb-2">📋 View Reports</a>
                            <a href="/download_all_pdfs" class="btn btn-outline-dark w-100">📥 Download All</a>
                        </div>
                    </div>
                </div>
            </div>

            <div id="result" class="mt-4"></div>
        </div>

        <footer class="container text-center mt-5 text-muted">
            <p>Calmic Sdn Bhd | Corrosion Detection AI © 2025 | Built with Ultralytics YOLOv8</p>
        </footer>

        <script>
            let mobileStream = null, laptopStream = null;
            const result = document.getElementById('result');

            function startMobileCamera() {{
                navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "environment" }} }}).then(s => {{
                    mobileStream = s;
                    document.getElementById('mobileVideo').srcObject = s;
                    document.getElementById('mobileVideo').style.display = 'block';
                    document.getElementById('mobileCaptureUI').style.display = 'block';
                }}).catch(err => {{
                    result.innerHTML = `<div class="alert alert-danger">❌ Camera denied: ${{err.message}}</div>`;
                }});
            }}

            function stopMobileCamera() {{
                if (mobileStream) mobileStream.getTracks().forEach(t => t.stop());
                document.getElementById('mobileVideo').style.display = 'none';
                document.getElementById('mobileCaptureUI').style.display = 'none';
            }}

            function captureMobilePhoto() {{
                const c = document.createElement('canvas');
                c.width = 640; c.height = 640;
                c.getContext('2d').drawImage(document.getElementById('mobileVideo'), 0, 0, 640, 640);
                document.getElementById('mobilePhoto').src = c.toDataURL('image/jpeg');
                document.getElementById('mobilePhoto').style.display = 'block';
                document.getElementById('mobilePhotoActions').style.display = 'block';
                stopMobileCamera();
            }}

            function retakeMobilePhoto() {{
                document.getElementById('mobilePhoto').style.display = 'none';
                document.getElementById('mobilePhotoActions').style.display = 'none';
            }}

            function submitMobilePhoto() {{
                fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ image: document.getElementById('mobilePhoto').src }})
                }}).then(res => res.json()).then(data => {{
                    result.innerHTML = `
                        <div class="card">
                            <div class="card-body">
                                <h5>✅ Result</h5>
                                <p>${{data.severity}}</p>
                                <img src="${{data.result_image}}" class="w-100" style="border-radius:10px;">
                                <div class="mt-3">
                                    <a href="/" class="btn btn-outline-dark">🏠 Home</a>
                                    <a href="/reports" class="btn btn-primary">📋 Reports</a>
                                </div>
                            </div>
                        </div>
                    `;
                }}).catch(err => {{
                    result.innerHTML = `<div class="alert alert-danger">❌ Error: ${{err.message}}</div>`;
                }});
            }}

            function startLaptopCamera() {{
                navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user" }} }}).then(s => {{
                    laptopStream = s;
                    document.getElementById('laptopVideo').srcObject = s;
                    document.getElementById('laptopVideo').style.display = 'block';
                    document.getElementById('laptopCaptureUI').style.display = 'block';
                }}).catch(err => {{
                    result.innerHTML = `<div class="alert alert-danger">❌ Camera denied: ${{err.message}}</div>`;
                }});
            }}

            function stopLaptopCamera() {{
                if (laptopStream) laptopStream.getTracks().forEach(t => t.stop());
                document.getElementById('laptopVideo').style.display = 'none';
                document.getElementById('laptopCaptureUI').style.display = 'none';
            }}

            function captureLaptopPhoto() {{
                const c = document.createElement('canvas');
                c.width = 640; c.height = 640;
                c.getContext('2d').drawImage(document.getElementById('laptopVideo'), 0, 0, 640, 640);
                document.getElementById('laptopPhoto').src = c.toDataURL('image/jpeg');
                document.getElementById('laptopPhoto').style.display = 'block';
                document.getElementById('laptopPhotoActions').style.display = 'block';
                stopLaptopCamera();
            }}

            function retakeLaptopPhoto() {{
                document.getElementById('laptopPhoto').style.display = 'none';
                document.getElementById('laptopPhotoActions').style.display = 'none';
            }}

            function submitLaptopPhoto() {{
                fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ image: document.getElementById('laptopPhoto').src }})
                }}).then(res => res.json()).then(data => {{
                    result.innerHTML = `
                        <div class="card">
                            <div class="card-body">
                                <h5>✅ Result</h5>
                                <p>${{data.severity}}</p>
                                <img src="${{data.result_image}}" class="w-100" style="border-radius:10px;">
                                <div class="mt-3">
                                    <a href="/" class="btn btn-outline-dark">🏠 Home</a>
                                    <a href="/reports" class="btn btn-primary">📋 Reports</a>
                                </div>
                            </div>
                        </div>
                    `;
                }}).catch(err => {{
                    result.innerHTML = `<div class="alert alert-danger">❌ Error: ${{err.message}}</div>`;
                }});
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and User.check_password(user.password_hash, password):
            login_user(user)
            return redirect('/')
        return 'Invalid credentials', 401
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: #f8f9fa; }}
            .card {{ border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container py-5">
            <a href="/" class="btn btn-outline-primary btn-sm mb-3">Home</a>
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h3>Admin Login</h3>
                            <form method="POST">
                                <input type="text" name="username" placeholder="Username" required class="form-control mb-3">
                                <input type="password" name="password" placeholder="Password" required class="form-control mb-3">
                                <button type="submit" class="btn btn-primary w-100">Login</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_filename, result_text, high, med, low = predict_image(filepath)

        if result_filename is None or result_filename == "":
            result_filename = "no_detection.jpg"
            fallback_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            if not os.path.exists(fallback_path):
                Image.new("RGB", (640, 640), (200, 200, 200)).save(fallback_path)

        timestamp = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

        try:
            conn = sqlite3.connect('corrosion.db')
            c = conn.cursor()
            c.execute("SELECT comments, custom_name FROM detections WHERE result_image = ?", (result_filename,))
            row = c.fetchone()
            existing_comment = row['comments'] if row else ''
            existing_custom_name = row['custom_name'] if row and row['custom_name'] else ''

            if row:
                c.execute('''
                    UPDATE detections SET
                    original_image = ?, result_text = ?, high_severity = ?, medium_severity = ?, low_severity = ?,
                    timestamp = ?, confirmed = COALESCE(confirmed, 0)
                    WHERE result_image = ?
                ''', (filename, result_text, high, med, low, timestamp, result_filename))
            else:
                c.execute('''
                    INSERT INTO detections 
                    (original_image, result_image, result_text, high_severity, medium_severity, low_severity, timestamp, comments, custom_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (filename, result_filename, result_text, high, med, low, timestamp, '', ''))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ DB Save failed: {str(e)}")

        safe_comment = (existing_comment or '').replace('"', '&quot;').replace("'", "\\'")
        safe_custom_name = (existing_custom_name or '').replace('"', '&quot;').replace("'", "\\'")
        safe_result_filename = result_filename.replace('"', '&quot;').replace("'", "\\'")

        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>✅ Detection Result</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
                    font-family: 'Segoe UI', sans-serif;
                    color: #333;
                }}
                .card {{
                    border: none;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .img-container {{
                    background: #f8f9fa;
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;
                    position: relative;
                    display: inline-block;
                }}
                .img-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                #markupCanvas {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    cursor: crosshair;
                    border-radius: 8px;
                }}
                .btn {{
                    border-radius: 50px;
                    padding: 10px 20px;
                    font-weight: 600;
                }}
                footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
                .comment-box {{
                    border: 2px solid #dee2e6;
                    border-radius: 10px;
                    padding: 15px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }}
                .tool-btn {{
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container py-4">
                <a href="/" class="btn btn-outline-primary btn-sm mb-3">🏠 Home</a>

                <div class="row justify-content-center">
                    <div class="col-lg-10">
                        <div class="card shadow-sm">
                            <div class="card-body text-center">
                                <h1 class="text-success mb-4">
                                    <i class="fas fa-check-circle"></i> Detection Complete
                                </h1>

                                <div class="alert alert-success mb-4">
                                    <h5>📋 Inspection Result</h5>
                                    <p class="mb-0" style="font-size:1.1em;">{result_text.replace('<br>', '<br>')}</p>
                                </div>

                                <div class="row g-4 mb-4">
                                    <div class="col-md-6">
                                        <div class="img-container">
                                            <h6>📸 Original Image</h6>
                                            <img src="/static/uploads/{filename}" alt="Original">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="img-container" id="detectedContainer">
                                            <h6>✏️ Detected & Markup</h6>
                                            <img id="detectedImage" src="/static/results/{result_filename}" alt="Detected" onload="initCanvas(this)">
                                            <canvas id="markupCanvas"></canvas>
                                        </div>
                                        <div class="mt-2">
                                            <button onclick="setDrawMode('pen')" class="btn btn-success tool-btn"><i class="fas fa-pen"></i></button>
                                            <button onclick="setDrawMode('erase')" class="btn btn-danger tool-btn"><i class="fas fa-eraser"></i></button>
                                            <button onclick="clearCanvas()" class="btn btn-warning tool-btn"><i class="fas fa-trash"></i></button>
                                            <button onclick="saveMarkup('{safe_result_filename}')" class="btn btn-primary">💾 Save Markup</button>
                                        </div>
                                        <div class="mt-3">
                                            <a href="/download_marked/{safe_result_filename}" class="btn btn-info">
                                                <i class="fas fa-download"></i> 📥 Download Marked Image
                                            </a>
                                        </div>
                                    </div>
                                </div>

                                <div class="comment-box mb-4">
                                    <form id="renameForm" class="w-100">
                                        <label for="custom_name" class="form-label"><strong>📋 Rename This Report</strong></label>
                                        <input type="text" 
                                               name="custom_name" 
                                               id="custom_name" 
                                               class="form-control mb-3"
                                               placeholder="e.g., Pipe_Joint_Inspection_Aug25"
                                               value="{safe_custom_name}">
                                        <button type="submit" class="btn btn-primary w-100">
                                            <i class="fas fa-edit"></i> Save Report Name
                                        </button>
                                    </form>
                                </div>

                                <div class="comment-box mb-4">
                                    <form id="commentForm" class="w-100">
                                        <label for="comment" class="form-label"><strong>Comments & Observations</strong></label>
                                        <textarea name="comment" id="comment" class="form-control mb-3" placeholder="e.g., Location: Pipe elbow, Suspected cause: Moisture ingress, Action: Schedule repair">{safe_comment}</textarea>
                                        <button type="submit" class="btn btn-primary w-100">
                                            <i class="fas fa-save"></i> Save Comment
                                        </button>
                                    </form>
                                </div>

                                <div class="mt-4 p-3 bg-light rounded">
                                    <p class="mb-3"><strong>Was this result correct?</strong></p>
                                    <div class="d-flex justify-content-center gap-3 flex-wrap">
                                        <a href="/confirm/{result_filename}?correct=true" class="btn btn-success">
                                            <i class="fas fa-thumbs-up"></i> Yes, Correct
                                        </a>
                                        <a href="/confirm/{result_filename}?correct=false" class="btn btn-danger">
                                            <i class="fas fa-thumbs-down"></i> No, Incorrect
                                        </a>
                                    </div>
                                </div>

                                <div class="mt-4">
                                    <a href="/" class="btn btn-outline-primary me-2 mb-2">
                                        <i class="fas fa-upload"></i> Upload Another
                                    </a>
                                    <a href="/reports" class="btn btn-primary mb-2">
                                        <i class="fas fa-list"></i> View All Reports
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <footer class="mt-5">
                    Calmic Sdn Bhd|Corrosion Detection AI © 2025 | Powered by Ultralytics YOLOv8
                </footer>
            </div>

            <script>
                let canvas, ctx;
                let isDrawing = false;
                let drawMode = 'pen';
                let lastX = 0;
                let lastY = 0;

                function initCanvas(img) {{
                    const container = document.getElementById('detectedContainer');
                    canvas = document.getElementById('markupCanvas');
                    ctx = canvas.getContext('2d');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;

                    const displayWidth = img.clientWidth;
                    const displayHeight = img.clientHeight;
                    const scale = {{
                        x: canvas.width / displayWidth,
                        y: canvas.height / displayHeight
                    }};

                    canvas.style.width = displayWidth + 'px';
                    canvas.style.height = displayHeight + 'px';

                    const saved = localStorage.getItem('markup_' + '{result_filename}');
                    if (saved) {{
                        const imgData = new Image();
                        imgData.onload = function() {{
                            ctx.drawImage(imgData, 0, 0, canvas.width, canvas.height);
                        }};
                        imgData.src = saved;
                    }}

                    function getMousePos(e) {{
                        const rect = canvas.getBoundingClientRect();
                        return {{
                            x: (e.clientX - rect.left) * scale.x,
                            y: (e.clientY - rect.top) * scale.y
                        }};
                    }}

                    canvas.addEventListener('mousedown', (e) => {{
                        const pos = getMousePos(e);
                        isDrawing = true;
                        [lastX, lastY] = [pos.x, pos.y];
                    }});

                    canvas.addEventListener('mousemove', (e) => {{
                        if (!isDrawing) return;
                        const pos = getMousePos(e);
                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(pos.x, pos.y);
                        ctx.strokeStyle = drawMode === 'erase' ? 'white' : 'red';
                        ctx.lineWidth = drawMode === 'erase' ? 20 : 3;
                        ctx.lineCap = 'round';
                        ctx.globalCompositeOperation = drawMode === 'erase' ? 'destination-out' : 'source-over';
                        ctx.stroke();
                        [lastX, lastY] = [pos.x, pos.y];
                    }});

                    canvas.addEventListener('mouseup', () => isDrawing = false);
                    canvas.addEventListener('mouseout', () => isDrawing = false);

                    canvas.addEventListener('touchstart', (e) => {{
                        e.preventDefault();
                        const touch = e.touches[0];
                        const mouseEvent = new MouseEvent('mousedown', {{
                            clientX: touch.clientX,
                            clientY: touch.clientY
                        }});
                        canvas.dispatchEvent(mouseEvent);
                    }});

                    canvas.addEventListener('touchmove', (e) => {{
                        e.preventDefault();
                        const touch = e.touches[0];
                        const mouseEvent = new MouseEvent('mousemove', {{
                            clientX: touch.clientX,
                            clientY: touch.clientY
                        }});
                        canvas.dispatchEvent(mouseEvent);
                    }});

                    canvas.addEventListener('touchend', (e) => {{
                        e.preventDefault();
                        const mouseEvent = new MouseEvent('mouseup');
                        canvas.dispatchEvent(mouseEvent);
                    }});
                }}

                function setDrawMode(mode) {{
                    drawMode = mode;
                }}

                function clearCanvas() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }}

                function saveMarkup(resultFilename) {{
                    const dataUrl = canvas.toDataURL('image/png');
                    localStorage.setItem('markup_' + resultFilename, dataUrl);

                    fetch('/save_markup', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            image_name: resultFilename,
                            markup_data: dataUrl
                        }})
                    }}).then(res => res.json())
                      .then(data => {{
                        alert(data.success ? '✅ Markup saved!' : '❌ Save failed');
                      }});
                }}

                document.getElementById('renameForm').addEventListener('submit', function(e) {{
                    e.preventDefault();
                    const formData = new FormData(this);
                    const customName = formData.get('custom_name');

                    fetch('/rename_report/{result_filename}', {{
                        method: 'POST',
                        body: new URLSearchParams({{ custom_name: customName }})
                    }}).then(res => res.text())
                      .then(html => {{
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        alert(doc.querySelector('.alert-success') ? '✅ Name saved!' : '❌ Save failed');
                      }});
                }});

                document.getElementById('commentForm').addEventListener('submit', function(e) {{
                    e.preventDefault();
                    const formData = new FormData(this);
                    const comment = formData.get('comment');

                    fetch('/save_comment/{result_filename}', {{
                        method: 'POST',
                        body: new URLSearchParams({{ comment: comment }})
                    }}).then(res => res.text())
                      .then(html => {{
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        alert(doc.querySelector('.alert-success') ? '✅ Comment saved!' : '❌ Save failed');
                      }});
                }});
            </script>
        </body>
        </html>
        '''
    return "Upload failed", 400

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        filename = f"mobile_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        result_filename, result_text, high, med, low = predict_image(filepath)

        if result_filename is None:
            result_filename = "no_detection.jpg"
            fallback_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            if not os.path.exists(fallback_path):
                Image.new("RGB", (640, 640), (200, 200, 200)).save(fallback_path)

        timestamp = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

        try:
            conn = sqlite3.connect('corrosion.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO detections (original_image, result_image, result_text, high_severity, medium_severity, low_severity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, result_filename, result_text, high, med, low, timestamp))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB Save failed: {str(e)}")

        return {
            "result_image": f"/static/results/{result_filename}",
            "severity": result_text
        }

    except Exception as e:
        logger.error(f"❌ API Error: {str(e)}")
        return {"error": str(e)}, 500

@app.route('/confirm/<result_filename>')
def confirm(result_filename):
    correct = request.args.get('correct') == 'true'
    try:
        conn = sqlite3.connect('corrosion.db')
        c = conn.cursor()
        c.execute("UPDATE detections SET confirmed = ? WHERE result_image = ?", (correct, result_filename))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"❌ DB Update failed: {str(e)}")

    if correct:
        title = "Thank You!"
        message = "Your feedback helps improve the AI's accuracy. We appreciate your input!"
        icon = "✅"
        alert_class = "success"
    else:
        title = "Feedback Received"
        message = "We're sorry the result wasn't accurate. Your feedback helps us improve."
        icon = "❌"
        alert_class = "danger"

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Submitted</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
                font-family: 'Segoe UI', sans-serif;
                min-height: 100vh;
                display: flex;
                align-items: center;
            }}
            .card {{
                border: none;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                color: #6c757d;
                font-size: 0.9em;
            }}
            .btn {{
                border-radius: 50px;
                padding: 10px 25px;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn btn-outline-primary btn-sm mb-3">🏠 Home</a>

            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card shadow text-center p-5">
                        <div style="font-size: 4rem; margin-bottom: 15px;">{icon}</div>
                        <h1 class="mb-4">{title}</h1>
                        <div class="alert alert-{alert_class} p-4 mb-4">
                            <p class="mb-0"><strong>Feedback:</strong> {"✅ Correct detection" if correct else "❌ Incorrect detection"}</p>
                        </div>
                        <p class="lead mb-4">{message}</p>
                        <div class="mt-4">
                            <a href="/" class="btn btn-outline-primary me-2 mb-2">Upload New Image</a>
                            <a href="/reports" class="btn btn-primary mb-2">View All Reports</a>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="mt-5">
                Calmic Sdn Bhd|Corrosion Detection AI © 2025 | Feedback helps train smarter AI
            </footer>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

@app.route('/reports')
@login_required
def view_reports():
    search = request.args.get('search', '').lower()
    min_date = request.args.get('min_date', '')
    max_date = request.args.get('max_date', '')
    severity_filter = request.args.get('severity', '')

    try:
        conn = sqlite3.connect('corrosion.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if search:
            query += " AND original_image LIKE ?"
            params.append(f"%{search}%")
        if min_date:
            query += " AND timestamp >= ?"
            params.append(min_date)
        if max_date:
            query += " AND timestamp <= ?"
            params.append(max_date)

        c.execute(query, params)
        rows = c.fetchall()
        conn.close()

        filtered_rows = []
        for row in rows:
            has_high = row['high_severity'] > 0
            has_med = row['medium_severity'] > 0
            has_low = row['low_severity'] > 0

            if severity_filter == 'high' and not has_high: continue
            elif severity_filter == 'med' and not has_med: continue
            elif severity_filter == 'low' and not has_low: continue
            filtered_rows.append(row)

        rows = filtered_rows

    except Exception as e:
        return f"<div class='alert alert-danger'>❌ Database error: {str(e)}</div>"

    form_html = f'''
    <div class="card shadow-sm mb-4">
        <div class="card-body">
            <h5 class="card-title mb-3">🔍 Filter Reports</h5>
            <form method="GET" class="row g-3">
                <div class="col-md-3">
                    <input type="text" name="search" placeholder="Search by filename" value="{search}" class="form-control" />
                </div>
                <div class="col-md-3">
                    <input type="date" name="min_date" value="{min_date}" class="form-control" />
                </div>
                <div class="col-md-3">
                    <input type="date" name="max_date" value="{max_date}" class="form-control" />
                </div>
                <div class="col-md-3">
                    <select name="severity" class="form-select">
                        <option value="">All Severities</option>
                        <option value="high" {"selected" if severity_filter=="high" else ""}>High Only</option>
                        <option value="med" {"selected" if severity_filter=="med" else ""}>Medium Only</option>
                        <option value="low" {"selected" if severity_filter=="low" else ""}>Low Only</option>
                    </select>
                </div>
                <div class="col-12">
                    <button type="submit" class="btn btn-primary">Apply Filters</button>
                    <a href="/reports" class="btn btn-outline-secondary">Reset</a>
                </div>
            </form>
        </div>
    </div>
    '''

    table_rows = ""
    for row in rows:
        confirmed = "✅ Yes" if row['confirmed'] else "❌ No"
        severity_badges = ""
        if row['high_severity'] > 0:
            severity_badges += f'<span class="badge bg-danger">High: {row["high_severity"]}</span> '
        if row['medium_severity'] > 0:
            severity_badges += f'<span class="badge bg-warning text-dark">Med: {row["medium_severity"]}</span> '
        if row['low_severity'] > 0:
            severity_badges += f'<span class="badge bg-success">Low: {row["low_severity"]}</span>'

        safe_filename = row['original_image'].replace("'", "\\'")
        display_name = row['custom_name'] or row['original_image']
        table_rows += f"""
        <tr>
            <td><input type="checkbox" class="delete-checkbox" data-id="{row['id']}" data-filename="{safe_filename}"></td>
            <td>{row['id']}</td>
            <td><small>{display_name}</small></td>
            <td>{severity_badges}</td>
            <td>{confirmed}</td>
            <td><small>{row['timestamp']}</small></td>
            <td><small>{row['comments'] or 'No comment'}</small></td>
            <td>
                <div class="d-flex gap-2">
                    <a href="/download_pdf/{row['id']}" class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-file-pdf"></i> PDF
                    </a>
                    <a href="#" onclick="confirmDelete({row['id']}, '{safe_filename}')" class="btn btn-sm btn-outline-danger">
                        <i class="fas fa-trash"></i> Delete
                    </a>
                </div>
            </td>
        </tr>
        """

    if not table_rows:
        table_rows = '''
        <tr>
            <td colspan="8" class="text-center text-muted">No reports found matching your criteria.</td>
        </tr>
        '''

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📋 Inspection Reports</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {{ background: #f8f9fa; font-family: 'Segoe UI', sans-serif; }}
            .header {{ background: white; border-bottom: 1px solid #dee2e6; padding: 15px 0; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .table {{ background: white; }}
            .table th {{ background: #f1f3f5; font-weight: 600; }}
            .badge {{ font-size: 0.8em; padding: 0.5em 0.8em; }}
            .btn i {{ margin-right: 5px; }}
            footer {{ margin-top: 50px; text-align: center; color: #6c757d; font-size: 0.9em; }}
            .bulk-actions {{ margin-bottom: 1rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="d-flex justify-content-between align-items-center">
                    <h1>📋 Inspection Reports</h1>
                    <div>
                        <a href="/" class="btn btn-outline-secondary btn-sm">
                            <i class="fas fa-home"></i> Home
                        </a>
                        <a href="/download_all_pdfs" class="btn btn-warning btn-sm ms-2">
                            <i class="fas fa-download"></i> All PDFs
                        </a>
                        <a href="/logout" class="btn btn-danger btn-sm ms-2">
                            <i class="fas fa-sign-out-alt"></i> Logout
                        </a>
                    </div>
                </div>
            </div>

            {form_html}

            <div class="bulk-actions">
                <button id="bulkDeleteBtn" class="btn btn-outline-danger btn-sm" disabled>
                    <i class="fas fa-trash"></i> Delete Selected
                </button>
                <span id="selectedCount" class="ms-2 text-muted"></span>
            </div>

            <div class="card shadow-sm">
                <div class="table-responsive">
                    <table class="table table-hover mb-0">
                        <thead>
                            <tr>
                                <th><input type="checkbox" id="select-all"></th>
                                <th>ID</th>
                                <th>Report Name</th>
                                <th>Severity</th>
                                <th>Confirmed</th>
                                <th>Timestamp</th>
                                <th>Comments</th>
                                <th>Report</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <footer class="mt-4">
                Calmic Sdn Bhd|Corrosion Detection AI © 2025 | Total: {len(rows)} report(s)
            </footer>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const selectAll = document.getElementById('select-all');
            const checkboxes = document.querySelectorAll('.delete-checkbox');
            const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
            const selectedCount = document.getElementById('selectedCount');

            function updateBulkActionState() {{
                const checked = document.querySelectorAll('.delete-checkbox:checked');
                const count = checked.length;
                bulkDeleteBtn.disabled = count === 0;
                selectedCount.textContent = count > 0 ? `${{count}} selected` : '';
            }}

            selectAll.addEventListener('change', function() {{
                checkboxes.forEach(cb => cb.checked = this.checked);
                updateBulkActionState();
            }});

            checkboxes.forEach(cb => {{
                cb.addEventListener('change', updateBulkActionState);
            }});

            function confirmDelete(id, filename) {{
                if (confirm(`Are you sure you want to delete report #${{id}} (${{filename}})? This cannot be undone.`)) {{
                    deleteReports([id]);
                }}
            }}

            document.getElementById('bulkDeleteBtn').addEventListener('click', function() {{
                const checked = document.querySelectorAll('.delete-checkbox:checked');
                const ids = Array.from(checked).map(cb => parseInt(cb.dataset.id));
                const filenames = Array.from(checked).map(cb => cb.dataset.filename);
                const list = filenames.slice(0, 5).join(', ') + (filenames.length > 5 ? '...' : '');
                if (confirm(`Are you sure you want to delete these ${{filenames.length}} report(s)?\\n\\n${{list}}\\n\\nThis cannot be undone.`)) {{
                    deleteReports(ids);
                }}
            }});

            function deleteReports(ids) {{
                fetch('/delete', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ ids: ids }})
                }})
                .then(res => res.json())
                .then(data => {{
                    if (data.success) {{
                        alert(`✅ Successfully deleted ${{data.results.success.length}} report(s)!`);
                        location.reload();
                    }} else {{
                        alert('❌ Delete failed: ' + (data.error || 'Unknown error'));
                    }}
                }})
                .catch(err => {{
                    alert('❌ Network error: ' + err.message);
                }});
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/delete', methods=['POST'])
@login_required
def delete_reports():
    data = request.get_json()
    ids_to_delete = data.get('ids', [])
    results = {'success': [], 'failed': []}

    if not ids_to_delete:
        return {"success": False, "error": "No IDs provided"}, 400

    try:
        conn = sqlite3.connect('corrosion.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        for detection_id in ids_to_delete:
            c.execute("SELECT id, original_image, result_image FROM detections WHERE id = ?", (detection_id,))
            row = c.fetchone()
            if not row:
                results['failed'].append({'id': detection_id, 'error': 'Not found'})
                continue

            c.execute('''
                INSERT INTO deletion_logs (detection_id, original_image, deleted_by)
                VALUES (?, ?, ?)
            ''', (row['id'], row['original_image'], current_user.username))

            try:
                os.remove(os.path.join('static/uploads', row['original_image']))
                os.remove(os.path.join('static/results', row['result_image']))
            except Exception as e:
                logger.warning(f"⚠️ Could not delete files for ID {detection_id}: {e}")

            c.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
            results['success'].append(detection_id)

        conn.commit()
        conn.close()
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Delete failed: {str(e)}")
        return {"success": False, "error": str(e)}, 500

@app.route('/download_all_pdfs')
@login_required
def download_all_pdfs():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        reports_dir = 'static/reports'
        if os.path.exists(reports_dir):
            for pdf_file in os.listdir(reports_dir):
                if pdf_file.endswith('.pdf'):
                    zf.write(os.path.join(reports_dir, pdf_file), pdf_file)
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='all_corrosion_reports.zip'
    )

@app.route('/download_pdf/<int:detection_id>')
@login_required
def download_pdf(detection_id):
    try:
        conn = sqlite3.connect('corrosion.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return "Report not found", 404

        # Create PDF report
        orig_path = os.path.join('static', 'uploads', row['original_image'])
        result_path = os.path.join('static', 'results', row['result_image'])
        pdf_filename = f"report_{detection_id}.pdf"
        pdf_path = os.path.join('static', 'reports', pdf_filename)
        
        # Debug: Check if files exist
        logger.info(f"Original image: {orig_path} exists: {os.path.exists(orig_path)}")
        logger.info(f"Result image: {result_path} exists: {os.path.exists(result_path)}")

        # Generate PDF with images
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(0, 0, 128)
            pdf.cell(0, 10, "Corrosion Inspection Report", ln=True, align='C')
            pdf.ln(5)

            # Add metadata
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0, 0, 0)
            display_name = row['custom_name'] or row['original_image']
            pdf.cell(0, 8, f"Image: {display_name}", ln=True)
            pdf.cell(0, 8, f"Generated on: {pdf_filename}", ln=True)
            pdf.ln(5)

            # Add detection result
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, "Detection Result:", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.set_text_color(0, 0, 0)
            clean_text = row['result_text'].replace('<br>', '\n').replace('Corrosion Detected: PASS ', '').replace('Severity: ', '')
            pdf.multi_cell(0, 6, clean_text)
            pdf.ln(5)

            # Add Comments
            if row['comments'].strip():
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(0, 100, 0)
                pdf.cell(0, 8, "Comments & Observations:", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, row['comments'])
                pdf.ln(5)

            # Add Images
            y = pdf.get_y() + 10
            img_width = 90

            # Check and add original image
            if os.path.exists(orig_path):
                try:
                    pdf.image(orig_path, x=10, y=y, w=img_width)
                except Exception as e:
                    logger.error(f"❌ Error adding original image: {str(e)}")
                    pdf.cell(90, 40, "Original image not found", border=1)
            else:
                pdf.cell(90, 40, "Original image missing", border=1)

            # Check and add result image
            if os.path.exists(result_path):
                try:
                    pdf.image(result_path, x=105, y=y, w=img_width)
                except Exception as e:
                    logger.error(f"❌ Error adding result image: {str(e)}")
                    pdf.cell(90, 40, "Detected image not found", border=1)
            else:
                pdf.cell(90, 40, "Detected image missing", border=1)

            # Labels
            pdf.set_y(y + 85)
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(90, 6, "Original Image", align='C')
            pdf.cell(90, 6, "Detected Corrosion", align='C')

            pdf.output(pdf_path)
            
        except Exception as e:
            logger.error(f"❌ PDF generation error: {str(e)}")
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container py-5">
                    <a href="/" class="btn btn-outline-primary btn-sm mb-3">Home</a>
                    <div class="alert alert-danger">
                        <h4>❌ Error Generating PDF</h4>
                        <p>{str(e)}</p>
                    </div>
                    <a href="/reports" class="btn btn-secondary">← Back to Reports</a>
                </div>
            </body>
            </html>
            '''

        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📄 PDF Generated</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
                    font-family: 'Segoe UI', sans-serif;
                    color: #333;
                }}
                .card {{
                    border: none;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .icon-large {{
                    font-size: 3rem;
                    color: #28a745;
                }}
                .btn {{
                    border-radius: 50px;
                    padding: 10px 25px;
                    font-weight: 600;
                }}
                footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container py-5">
                <a href="/" class="btn btn-outline-primary btn-sm mb-3">🏠 Home</a>

                <div class="row justify-content-center">
                    <div class="col-lg-8">
                        <div class="card shadow text-center p-5">
                            <div class="icon-large">
                                <i class="fas fa-file-pdf"></i>
                            </div>
                            <h1 class="mb-4">📄 PDF Report Generated!</h1>
                            <p class="lead mb-4">
                                Your corrosion inspection report has been successfully created.
                                Click the button below to download it.
                            </p>

                            <p>
                                <a href="/static/reports/{pdf_filename}" target="_blank" class="btn btn-success btn-lg">
                                    <i class="fas fa-download"></i> Download Report
                                </a>
                            </p>

                            <div class="mt-4">
                                <a href="/reports" class="btn btn-outline-secondary me-2 mb-2">
                                    Back to Reports
                                </a>
                                <a href="/" class="btn btn-outline-dark mb-2">
                                    Home
                                </a>
                            </div>
                        </div>
                    </div>
                </div>

                <footer class="mt-5">
                    Calmic Sdn Bhd|Corrosion Detection AI © 2025 | Report: #{detection_id}
                </footer>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        '''

    except Exception as e:
        logger.error(f"❌ Download PDF error: {str(e)}")
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container py-5">
                <a href="/" class="btn btn-outline-primary btn-sm mb-3">Home</a>
                <div class="alert alert-danger">
                    <h4>❌ Error Generating PDF</h4>
                    <p>{str(e)}</p>
                </div>
                <a href="/reports" class="btn btn-secondary">← Back to Reports</a>
            </div>
        </body>
        </html>
        '''

@app.route('/save_markup', methods=['POST'])
def save_markup():
    try:
        data = request.get_json()
        image_name = data['image_name']
        markup_data = data['markup_data'].split(',')[1]
        image_data = base64.b64decode(markup_data)

        markup_dir = os.path.join(app.config['RESULT_FOLDER'], 'markup')
        os.makedirs(markup_dir, exist_ok=True)
        markup_path = os.path.join(markup_dir, f"markup_{image_name}")

        with open(markup_path, 'wb') as f:
            f.write(image_data)

        conn = sqlite3.connect('corrosion.db')
        c = conn.cursor()
        c.execute("UPDATE detections SET comments = comments || '\n[Markup saved]' WHERE result_image = ?", (image_name,))
        conn.commit()
        conn.close()

        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Save markup failed: {str(e)}")
        return {"success": False, "error": str(e)}, 500

@app.route('/download_marked/<result_filename>')
@login_required
def download_marked_image(result_filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    markup_path = os.path.join(app.config['RESULT_FOLDER'], 'markup', f'markup_{result_filename}')
    
    try:
        base = Image.open(result_path).convert("RGBA")
        if os.path.exists(markup_path):
            overlay = Image.open(markup_path).convert("RGBA")
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            combined = Image.alpha_composite(base, overlay)
            temp_buffer = BytesIO()
            combined.convert("RGB").save(temp_buffer, "JPEG")
            temp_buffer.seek(0)
            return send_file(temp_buffer, mimetype='image/jpeg', as_attachment=True, download_name=f"marked_{result_filename}")
        else:
            return send_file(result_path, as_attachment=True)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/rename_report/<result_filename>', methods=['POST'])
def rename_report(result_filename):
    custom_name = request.form.get('custom_name', '').strip()
    if not custom_name:
        custom_name = ''

    try:
        conn = sqlite3.connect('corrosion.db')
        c = conn.cursor()
        c.execute("UPDATE detections SET custom_name = ? WHERE result_image = ?", (custom_name, result_filename))
        conn.commit()
        conn.close()
        message = "✅ Report name saved successfully!"
        alert_class = "success"
    except Exception as e:
        message = f"❌ Error saving name: {str(e)}"
        alert_class = "danger"

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>✅ Name Saved</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); font-family: 'Segoe UI', sans-serif; }}
            .alert {{ border-radius: 10px; }}
            footer {{ margin-top: 50px; text-align: center; color: #6c757d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container py-4">
            <a href="/" class="btn btn-outline-primary btn-sm mb-3">🏠 Home</a>
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="alert alert-{alert_class} text-center">
                        <h4>{message}</h4>
                    </div>
                    <div class="text-center mt-4">
                        <a href="/reports" class="btn btn-primary me-2">📋 Reports</a>
                        <a href="/" class="btn btn-outline-dark">🏠 Home</a>
                    </div>
                </div>
            </div>
            <footer class="mt-5"> Calmic Sdn Bhd|Corrosion Detection AI © 2025</footer>
        </div>
    </body>
    </html>
    '''

@app.route('/save_comment/<result_filename>', methods=['POST'])
def save_comment(result_filename):
    comment = request.form.get('comment', '').strip()
    try:
        conn = sqlite3.connect('corrosion.db')
        c = conn.cursor()
        c.execute("UPDATE detections SET comments = ? WHERE result_image = ?", (comment, result_filename))
        conn.commit()
        conn.close()
        message = "✅ Comment saved successfully!"
        alert_class = "success"
    except Exception as e:
        message = f"❌ Error saving comment: {str(e)}"
        alert_class = "danger"

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>✅ Comment Saved</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); font-family: 'Segoe UI', sans-serif; }}
            .alert {{ border-radius: 10px; }}
            footer {{ margin-top: 50px; text-align: center; color: #6c757d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container py-4">
            <a href="/" class="btn btn-outline-primary btn-sm mb-3">🏠 Home</a>
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="alert alert-{alert_class} text-center">
                        <h4>{message}</h4>
                    </div>
                    <div class="text-center mt-4">
                        <a href="/reports" class="btn btn-primary me-2">📋 Reports</a>
                        <a href="/" class="btn btn-outline-dark">🏠 Home</a>
                    </div>
                </div>
            </div>
            <footer class="mt-5">Calmic Sdn Bhd|Corrosion Detection AI © 2025</footer>
        </div>
    </body>
    </html>
    '''

# ===========================
# Run the App
# ===========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)