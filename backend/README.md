# PipeCount - FastAPI Backend

A high-performance Python REST API for automated thermal insulation pipe detection and counting. Powered by a lightweight ONNX Runtime inference engine with built-in concentric double-suppression and cardboard/watermark distractor filtering.

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
Alternatively, on Windows double-click:
```powershell
.\run_backend.bat
```

- **API Base:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/api/docs`
- **Alternative ReDoc Docs:** `http://127.0.0.1:8000/api/redoc`
- **Health Check:** `http://127.0.0.1:8000/api/health`

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Upload an image file (`file`, multipart/form-data) to count pipes. Returns total count, bounding boxes, confidence metrics, and base64 annotated image. Supports `.jpg`, `.png`, `.webp`, `.jfif`, `.gfif`, `.bmp`, `.tiff`, `.gif`. |
| `GET` | `/api/health` | Health status and supported formats check. |
| `GET` | `/api/docs` | Interactive Swagger API explorer. |

---

## 🧪 Standalone Python Usage

You can use the detection engine directly in any Python script without running the web server:

```python
from insulation_detector import InsulationPipeDetector

# Load model (automatically loads ONNX or PyTorch weights)
detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_clean.onnx")

# Run detection on any image file or NumPy array
annotated_image, detections = detector.detect("sample_test/box_touching_grid_70.jpg", conf=0.35)

print(f"Total pipes counted: {len(detections)}")
```
