# train_segmentation.py
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("hamka-corrosion").project("corrosion-detection-xjdlv")

# Download segmentation dataset
dataset = project.version(1).download("yolov8")

# Train with improved settings
import os
os.system(f"""
yolo task=segment mode=train 
    model=yolov8m-seg.pt 
    data={dataset.location}/data.yaml 
    epochs=150 
    imgsz=640 
    batch=16 
    lr0=0.01 
    augment=True 
    iou=0.8 
    patience=50
""")