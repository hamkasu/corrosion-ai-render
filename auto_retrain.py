# auto_retrain.py - Auto-retrain using confirmed data
import sqlite3
import shutil
import os
from roboflow import Roboflow

def collect_confirmed_images():
    # Connect to DB
    conn = sqlite3.connect('corrosion.db')
    c = conn.cursor()
    c.execute("SELECT original_image, result_image FROM detections WHERE confirmed = 1")
    rows = c.fetchall()
    conn.close()

    # Copy to retraining folder
    retrain_dir = "retrain_dataset"
    os.makedirs(f"{retrain_dir}/images", exist_ok=True)
    for orig, result in rows:
        src = os.path.join('static/uploads', orig)
        dst = os.path.join(retrain_dir, 'images', orig)
        if os.path.exists(src):
            shutil.copy(src, dst)
    print(f"✅ Collected {len(rows)} confirmed images for retraining")
    return len(rows)

# Run if enough data
if __name__ == '__main__':
    count = collect_confirmed_images()
    if count >= 10:
        print("🚀 Ready to retrain! Upload to Roboflow...")
        # Optional: Use Roboflow API to upload and retrain
        # rf = Roboflow(api_key="YOUR_KEY")
        # project = rf.workspace("hamka-corrosion").project("corrosion-detection-xjdlv")
        # version = project.version(2)
        # version.upload_dataset(
        #     dataset_path="retrain_dataset",
        #     annotation_format="yolov8",
        #     model_format="yolov8-segmentation"
        # )
        # version.train()
    else:
        print("⏳ Not enough confirmed images yet (need 10+)")