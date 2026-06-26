# 🦷 Dental Caries Detection System

An advanced, deep-learning-powered clinical decision support system designed to assist dental professionals in identifying **carious lesions (cavities)** on dental radiographs (X-rays) and oral photographs. The system utilizes a custom-trained **YOLOv5** object detection model integrated into a lightweight **FastAPI** web service and an interactive local **OpenCV GUI** runner.

---

## 🚀 Key Features

*   **Custom-Trained YOLOv5 Model**: Optimized for detecting small carious lesions on dental radiographs.
*   **Dual Mode Execution**:
    1.  **FastAPI REST Web Service (`app.py`)**: A modern API server running inference at 1024px with CORS enabled, complete with an interactive drag-and-drop web frontend.
    2.  **Local OpenCV GUI (`deploy.py`)**: A desktop-friendly runner supporting real-time inference on images, video files, or direct live webcam feeds.
*   **Medical False-Positive Suppressor**: An intelligent pixel-intensity heuristic filter that analyzes cropped bounding box regions to suppress false positive detections on healthy, bright, highly reflective teeth surfaces.
*   **Confidence Power-Curve Calibration**: Uses a power-curve calibration (`confidence = conf_raw^0.45`) to boost under-confident model detections to match visual diagnostic expectations.
*   **Cross-Platform Path Resolver**: Monkeypatched for Windows/POSIX path compatibility, allowing seamless model loading across environments.

---

## 📁 Repository Structure

```text
├── app.py              # FastAPI REST API server & router
├── deploy.py           # Local OpenCV GUI testing script (Images, Video, Webcam)
├── last.pt             # Trained YOLOv5 PyTorch model weight file (14.3 MB)
├── .gitignore          # Excludes caches, tools, and temporary directories
├── static/             # Static assets folder (served at '/')
│   └── index.html      # Interactive web frontend interface
└── yolov5-master/      # Core YOLOv5 object detection engine (local codebase)
```

---

## 🛠️ Local Installation & Setup

### Prerequisites
*   **Python 3.8 to 3.11** installed on your system.
*   **Git** (for version control operations).

### 1. Clone & Set Up Directory
Open your terminal and navigate to the project directory:
```bash
cd D:\Project\Deploy
```

### 2. Create and Activate Virtual Environment
It is highly recommended to isolate your dependencies using a virtual environment:

*   **Windows (PowerShell)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies
Install PyTorch, OpenCV, FastAPI, and other core libraries:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy fastapi uvicorn pillow python-multipart
```
> **Note**: The command above installs the CPU version of PyTorch. If you have an NVIDIA GPU and want faster inference, install the CUDA-supported PyTorch version from [pytorch.org](https://pytorch.org/).

---

## 💻 Running the Project Locally

### Option 1: Run the Interactive FastAPI Web Server (Recommended)
This starts the backend REST API and hosts the web interface.

1.  Start the FastAPI server using Uvicorn:
    ```bash
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    ```
2.  Open your web browser and go to:
    **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
3.  Upload an X-ray or dental image (e.g., `Dental_Caries.jpeg`) to inspect bounding boxes, confidence scores, and diagnostic reports interactively.
4.  View API documentation and test endpoints at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Option 2: Run the Local OpenCV GUI Script
This script performs offline inference and displays the results in an OpenCV window.

1.  By default, `deploy.py` is configured to run inference on `Dental_Caries.jpeg`. Start it by running:
    ```bash
    python deploy.py
    ```
2.  An image window will pop up showing the bounding box predictions.
3.  **Keyboard Controls**:
    *   Press the **`ESC`** key to save the annotated output (`final_output.jpg`) and close the window.
4.  **Webcam & Video Feeds**:
    *   To analyze webcams or video files, uncomment the corresponding lines in the `main` block at the bottom of `deploy.py`:
        ```python
        # For video file analysis:
        # main(vid_path="path_to_video.mp4", vid_out="result_video.mp4")
        
        # For live webcam feed:
        # main(vid_path=0, vid_out="webcam_result.mp4")
        ```

---

## ⚡ API Endpoints

The FastAPI app exposes the following REST endpoints:

*   **`GET /status`**: Returns the health state of the API and the list of classes supported by the model.
*   **`POST /predict`**: Accepts an image file and returns coordinates, labels, and calibrated confidence levels in JSON format alongside the original image base64 buffer.
    *   *Parameters*: `augment` (Boolean, default: `false`) - set to `true` to run Test-Time Augmentation (TTA) for increased accuracy on smaller cavities.

---

## 🧠 Diagnostic Enhancements Details

### 1. False-Positive Reflector
Because dental radiographs frequently contain bright, highly reflective white areas (e.g., healthy tooth structures, dental fillings, crowns), object detectors can sometimes misidentify these as carious lesions. 
The system mitigates this by cropping each bounding box region, translating it to grayscale, and extracting the **mean pixel intensity**, **dark pixel ratio**, and **bright pixel ratio**. If the cropped region exceeds bright thresholds and lacks dark textures, the detection is bypassed as a healthy tooth.

### 2. Confidence Calibration
PyTorch models trained on specific medical datasets can often display conservative confidence metrics. The backend runs raw outputs through a power curve calibration function:
$$Confidence_{calibrated} = (Confidence_{raw})^{0.45}$$
This maps a conservative $0.43$ raw score to a visually representative $\sim 0.66$ score, aligning the output with standard clinical scales while maintaining relative probability rankings.
