import torch
import cv2
import numpy as np
import os
import pathlib
import time
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# POSIX path hack for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Monkeypatch torch.load for PyTorch 2.6+ compatibility with custom YOLOv5 serialization
_orig_load = torch.load
def new_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = new_load

# Initialize FastAPI application
app = FastAPI(
    title="Dental Caries Detection API",
    description="Deep Learning API for detecting dental cavities using YOLOv5",
    version="2.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv5 model
print("[INFO] Initializing model loading...")
script_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(script_dir, "yolov5-master")
model_path = os.path.join(script_dir, "last.pt")

try:
    # Load custom model locally using the local yolov5 codebase
    model = torch.hub.load(yolov5_path, 'custom', source='local', path=model_path, force_reload=True)
    classes = model.names
    print(f"[INFO] Model loaded successfully. Classes detected: {classes}")
    
    # Configure model for interactive detection
    model.conf = 0.10  # Low baseline confidence threshold to send all candidate detections to client
    model.iou = 0.45   # IoU threshold for NMS
    
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None
    classes = {}

# Endpoint to test API status
@app.get("/status")
async def get_status():
    return {
        "status": "online" if model is not None else "degraded (model not loaded)",
        "classes": list(classes.values())
    }

# Endpoint for image prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...), augment: bool = Query(default=False, description="Enable Test-Time Augmentation (TTA)")):
    if model is None:
        raise HTTPException(status_code=500, detail="Object detection model is not loaded on the server.")
    
    start_time = time.time()
    try:
        # Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Provided file is not a valid image format.")
            
        h, w, c = img.shape
        
        # Convert BGR to RGB for YOLOv5
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference at 1024px resolution (TTA if augment is True) for higher precision on small cavities
        results = model([img_rgb], size=1024, augment=augment)
        
        # Parse output coordinates and classes
        labels = results.xyxyn[0][:, -1].tolist()
        coords = results.xyxyn[0][:, :-1].tolist()
        
        detections = []
        for i in range(len(labels)):
            row = coords[i]
            conf_raw = float(row[4])
            
            # Get denormalized coordinates in pixels
            x1 = int(row[0] * w)
            y1 = int(row[1] * h)
            x2 = int(row[2] * w)
            y2 = int(row[3] * h)
            
            # Crop region to analyze pixel intensity (false positive filter for healthy bright teeth)
            # Clip coordinates to image boundaries
            cx1 = max(0, min(w - 1, x1))
            cy1 = max(0, min(h - 1, y1))
            cx2 = max(0, min(w - 1, x2))
            cy2 = max(0, min(h - 1, y2))
            
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(crop_gray)
                
                # Dark pixels (< 90) ratio
                dark_pixels = np.sum(crop_gray < 90)
                dark_ratio = dark_pixels / crop_gray.size
                
                # Bright pixels (> 185) ratio
                bright_pixels = np.sum(crop_gray > 185)
                bright_ratio = bright_pixels / crop_gray.size
                
                # Heuristic for bright healthy tooth: high mean intensity, low dark ratio, high bright ratio
                if mean_intensity > 158 and dark_ratio < 0.15 and bright_ratio > 0.30:
                    print(f"[INFO] Suppressing false-positive detection on healthy bright tooth: mean={mean_intensity:.1f}, dark={dark_ratio:.2f}, bright={bright_ratio:.2f}")
                    continue  # Skip/discard this false positive detection
            
            # Apply confidence calibration (power curve to boost under-confident medical model outputs)
            # A raw confidence of 0.43 becomes ~0.66, 0.57 becomes ~0.75, which matches human visual diagnostic expectations.
            conf = round(float(np.power(conf_raw, 0.45)), 4) if conf_raw > 0 else 0.0
            
            class_id = int(labels[i])
            class_name = classes.get(class_id, f"Class {class_id}")
            
            detections.append({
                "id": i,
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
        # Encode original image to base64
        _, buffer_orig = cv2.imencode('.jpg', img)
        orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return JSONResponse(content={
            "success": True,
            "width": w,
            "height": h,
            "detections": detections,
            "processing_time_ms": processing_time,
            "original_image": f"data:image/jpeg;base64,{orig_base64}"
        })
        
    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount static files directory
static_dir_path = os.path.join(script_dir, "static")
if not os.path.exists(static_dir_path):
    os.makedirs(static_dir_path)

app.mount("/static", StaticFiles(directory=static_dir_path), name="static")

# Serve Index Page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(static_dir_path, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Index page not found inside static directory.</h1>")
