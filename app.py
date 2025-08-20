# app.py - Web-Enabled Corrosion Detection with Supabase

from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import requests
from dotenv import load_dotenv
import base64

# ===========================
# Load Environment Variables
# ===========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}

# ===========================
# Flask App Setup
# ===========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===========================
# Load YOLO Model
# ===========================
try:
    model = YOLO('best.pt')
    model.to('cpu')
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

# ===========================
# Helper: Upload Image to Supabase Storage
# ===========================
def upload_to_supabase(image_path, filename):
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
    with open(image_path, "rb") as f:
        response = requests.post(url, headers=HEADERS, data=f)
    if response.status_code == 200:
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
    else:
        print("❌ Upload failed:", response.text)
        return None

# ===========================
# Helper: Save Result to Database
# ===========================
def save_to_database(filename, result, image_url):
    url = f"{SUPABASE_URL}/rest/v1/corrosion_uploads"
    data = {
        "filename": filename,
        "result": result,
        "image_url": image_url
    }
    response = requests.post(url, headers=HEADERS, json=data)
    return response.status_code == 201

# ===========================
# Prediction Function
# ===========================
def predict_image(filepath):
    if not model:
        return None, "Model not loaded"

    try:
        image = Image.open(filepath)
        results = model(image, conf=0.25)
        num_detections = len(results[0].boxes)

        # Save result image with boxes
        result_img = results[0].plot()
        result_pil = Image.fromarray(result_img)
        
        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        result_pil.save(result_path)

        # Upload to Supabase
        public_url = upload_to_supabase(result_path, result_filename)
        if not public_url:
            return None, "Failed to upload image"

        # Save to DB
        success = save_to_database(result_filename, f"Detected {num_detections} spot(s)" if num_detections > 0 else "No corrosion", public_url)

        result_text = f"Corrosion Detected ✅ ({num_detections} spot(s))" if num_detections > 0 else "No Corrosion ✅"
        return result_filename, result_text, public_url

    except Exception as e:
        print("❌ Error:", str(e))
        return None, "Analysis failed", None

# ===========================
# Routes
# ===========================
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    if file:
        # Save locally first
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        # Predict
        result_filename, result_text, public_url = predict_image(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return render_template('result.html', 
                             result=result_text, 
                             image_url=public_url)

# ===========================
# Run App
# ===========================
if __name__ == '__main__':
    app.run(debug=True)