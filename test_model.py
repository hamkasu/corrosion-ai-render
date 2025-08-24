from ultralytics import YOLO
import os

MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"❌ File not found: {MODEL_PATH}")
else:
    print(f"📄 File size: {os.path.getsize(MODEL_PATH) / 1e6:.2f} MB")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
        results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
        print("🚀 Test inference successful!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")