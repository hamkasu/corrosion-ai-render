# app.py - Corrosion Detection using Roboflow Inference API

from flask import Flask, request, render_template, redirect, url_for
import requests
import os
import uuid
from PIL import Image
from io import BytesIO

# ===========================
# Roboflow Configuration
# ===========================
ROBOFLOW_API_KEY = "UEVy3RH1ekFLVJYMztXn"           # Your API key
ROBOFLOW_MODEL_ID = "corrosion-detection-xjdlv/1"   # Project ID / Version
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

# ===========================
# Supabase Configuration
# ===========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "your-anon-key")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "corrosion-images")

HEADERS = {
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
}

# ===========================
# Flask App Setup
# ===========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===========================
# Helper: Upload Image to Supabase Storage
# ===========================
def upload_to_supabase(image_path, filename):
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
    try:
        with open(image_path, "rb") as f:
            response = requests.post(url, headers=HEADERS, data=f)
        if response.status_code == 200:
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        else:
            print("❌ Upload failed:", response.text)
            return None
    except Exception as e:
        print("❌ File error:", e)
        return None

# ===========================
# Helper: Save Result to Supabase DB
# ===========================
def save_to_database(filename, result, image_url):
    url = f"{SUPABASE_URL}/rest/v1/corrosion_uploads"
    data = {
        "filename": filename,
        "result": result,
        "image_url": image_url
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.status_code == 201

# ===========================
# Prediction: Use Roboflow Inference API
# ===========================
def predict_with_roboflow(image_path):
    try:
        # Open image and send to Roboflow
        with open(image_path, "rb") as img_file:
            response = requests.post(
                ROBOFLOW_URL,
                params={"api_key": ROBOFLOW_API_KEY},
                files={"file": img_file}
            )

        # Parse JSON response
        if response.status_code == 200:
            result = response.json()

            # Extract predictions
            predictions = result.get("predictions", [])
            num_detections = len(predictions)

            # Draw bounding boxes using Roboflow's annotate feature
            # This returns image with boxes
            image_response = requests.post(
                ROBOFLOW_URL,
                params={
                    "api_key": ROBOFLOW_API_KEY,
                    "format": "image",
                    "labels": "true"
                },
                files={"file": open(image_path, "rb")}
            )

            if image_response.status_code == 200:
                # Save annotated image
                annotated_img = Image.open(BytesIO(image_response.content))
                result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                annotated_img.save(result_path)
                return result_filename, num_detections, result_path
            else:
                return None, num_detections, None
        else:
            print("❌ Inference error:", response.text)
            return None, 0, None

    except Exception as e:
        print("❌ API call failed:", str(e))
        return None, 0, None

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
        # Save uploaded image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        # Run inference on Roboflow
        result_filename, num_detections, result_path = predict_with_roboflow(temp_path)

        if result_filename is None:
            result_text = "Error analyzing image"
            public_url = temp_path  # Fallback
        else:
            # Upload annotated image to Supabase
            public_url = upload_to_supabase(result_path, result_filename)

            # Save to database
            result_text = f"Corrosion Detected ✅ ({num_detections} spot(s))" if num_detections > 0 else "No Corrosion ✅"
            save_to_database(result_filename, result_text, public_url)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if result_path and os.path.exists(result_path):
            os.remove(result_path)

        return render_template('result.html', result=result_text, image_url=public_url)

# ===========================
# Run App
# ===========================
if __name__ == '__main__':
    app.run(debug=True)