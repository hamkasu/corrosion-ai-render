# download_and_train.py

from roboflow import Roboflow

# === UPDATE THESE VALUES ===
API_KEY = "UEVy3RH1ekFLVJYMztXn"           # Paste your key from roboflow.com/account/api
WORKSPACE = "hamka-corrosion"       # From URL: app.roboflow.com/WORKSPACE/project
PROJECT = "corrosion-detection-xjdlv"         # Your project name
VERSION = 1
# ===========================

print("📡 Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)

print("📂 Loading project...")
project = rf.workspace(WORKSPACE).project(PROJECT)

print("⬇️ Downloading dataset (YOLOv8 format)...")
try:
    dataset = project.version(VERSION).download("yolov8")
    print("🎉 Success! Dataset saved to:")
    print(dataset.location)

    # Start training
    print("🚀 Starting YOLOv8 training...")
    import os
    os.system(f"yolo task=detect mode=train model=yolov8s.pt data='{dataset.location}/data.yaml' epochs=50 imgsz=640")

except Exception as e:
    print("❌ Error:", str(e))
    print("💡 Common issues:")
    print("   - Wrong workspace/project name")
    print("   - Dataset not generated (click 'Generate' on Roboflow)")
    print("   - No labeled images")