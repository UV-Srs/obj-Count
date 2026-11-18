# Vision-intelligence — FastAPI Backend

A high-performance Python REST API for automated thermal insulation pipe detection and counting. Powered by a lightweight ONNX Runtime inference engine with our proprietary **Two-Stage Hybrid Neural-Geometric Pipeline** (concentric bore suppression, cardboard vent-hole gating, and interstitial void filtering).

---

## 🚀 Run Commands

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

*(Note: Production runs on ultra-lightweight `onnxruntime` ~85 MB, fully compatible with serverless platforms like Vercel and AWS Lambda).*

### 2. Start the Backend Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/api/docs`
- **Health Check:** `http://127.0.0.1:8000/api/health`

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Upload an image file (`file`, multipart/form-data) to count pipes. Returns total count, bounding boxes, confidence metrics, and base64 annotated image. Supports `.jpg`, `.png`, `.webp`, `.jfif`, `.gfif`, `.bmp`, `.tiff`. |
| `GET` | `/api/model/info` | Query current active engine (`yolov8m` 1024px or `yolov8n` 640px), native resolution, and status. |
| `POST` | `/api/model/switch` | Dynamically switch the active AI engine between YOLOv8 Medium (1024px) and YOLOv8 Nano (640px). |
| `GET` | `/api/samples` | Return list of built-in demo warehouse sample photos. |
| `GET` | `/api/sample-image/{sample_id}` | Fetch a demo sample photo directly by ID. |
| `POST` | `/api/detect-sample` | Run pipe detection directly on a built-in demo sample without uploading. |
| `GET` | `/api/health` | Health status and supported formats check. |
| `GET` | `/api/docs` | Interactive Swagger API explorer. |

---

## 🧪 Standalone Python Usage

You can use the detection engine directly in any Python script without running the web server:

```python
from backend.insulation_detector import InsulationPipeDetector

# Load active high-resolution 1024px engine
detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_m_1024.onnx")

# Run detection on any image file or NumPy array
annotated_image, detections = detector.detect("sample_test/box_touching_grid_70.jpg", conf=0.10)

print(f"Total pipes counted: {len(detections)}")
for d in detections:
    print(f"Pipe #{d['id']}: [{d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}] - Conf: {d.get('conf', 0.0):.2f}")
```
