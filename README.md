# PipeCount - Insulation Pipe Detection Studio

A production-grade Object Detection suite specialized for **Thermal Insulation Pipe Counting**, featuring a clean **React Frontend** dashboard, a high-performance **FastAPI Backend**, and a lightweight **ONNX Runtime** inference engine with built-in concentric double-suppression.

---

## 🚀 Run Commands (Frontend & Backend)

### Quick Start: Launch Both Together (Windows)
Double-click [**`run_all.bat`**](run_all.bat) or run from your terminal:
```powershell
.\run_all.bat
```
This automatically boots the FastAPI backend (Port 8000), starts the Vite dev server (Port 5173), and opens `http://localhost:5173` in your browser.

---

### Option A: Running the Backend (FastAPI)

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server:**
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *(Or double-click `run_backend.bat`)*

- **Backend API:** `http://127.0.0.1:8000`
- **Swagger Documentation:** `http://127.0.0.1:8000/api/docs`
- **Health Check:** `http://127.0.0.1:8000/api/health`

---

### Option B: Running the Frontend (React + Vite)

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   *(Or double-click `run_frontend.bat`)*

- **Web Dashboard:** `http://localhost:5173`
- *(Vite automatically proxies `/api/*` requests to the FastAPI backend on port 8000).*

---

### Option C: Building Frontend for Production
```bash
cd frontend
npm run build
```
Builds the optimized single-page application into `frontend/dist/`.

---

## ☁️ Vercel Deployment

This repository is pre-configured for multi-service deployment on Vercel via [`vercel.json`](vercel.json):

- **Frontend Service**: React + Vite SPA at `frontend/` (routed to `/(.*)`).
- **Backend Service**: FastAPI Serverless Function at `backend/` (routed to `/api/(.*)`).
- **Lightweight Dependencies**: Uses `onnxruntime` (~85 MB total bundle size), resolving AWS Lambda / Vercel `LAMBDA_SIZE_EXCEEDED` limits.

To deploy with Vercel CLI:
```bash
vercel
```
Or import the GitHub repository into your Vercel Dashboard.

---

## 📁 Repository Structure

```
c:\AI\
├── backend/
│   ├── main.py                  # FastAPI server with /api/detect & /api/health
│   ├── insulation_detector.py   # ONNX Runtime & PyTorch inference engine
│   ├── concentric_nms.py        # 4-pass anti-double & distractor filter
│   ├── requirements.txt         # Lightweight serverless dependencies (onnxruntime)
│   ├── run_backend.bat          # One-click backend launcher
│   └── models/
│       ├── best_insulation_pipe_clean.onnx  # Production ONNX model (12 MB)
│       └── best_insulation_pipe_clean.pt    # PyTorch model weights
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Dashboard UI (drag-and-drop, stacked results)
│   │   ├── App.css              # Clean industrial dashboard styling
│   │   └── main.jsx             # React entrypoint
│   ├── package.json             # React 19 & Vite dependencies
│   ├── vite.config.js           # Vite config with API reverse proxy
│   └── run_frontend.bat         # One-click frontend launcher
│
├── run_all.bat                  # Concurrently launches backend & frontend
├── run_backend.bat              # Root launcher for backend
├── run_frontend.bat             # Root launcher for frontend
├── vercel.json                  # Vercel multi-service routing configuration
└── README.md                    # Project documentation
```

---

## 🛡️ Anti-Double & Distractor Suppression Pipeline

Hollow insulation foam pipes in warehouse environments create unique computer vision challenges:
1. **Concentric Double Detection**: Standard NMS double-counts the outer foam ring and the dark hollow inner hole. An **Intersection-over-Smaller (IoS)** suppression pass eliminates any duplicate inner bore box when $\ge 50\%$ is nested inside a larger concentric box.
2. **Cardboard Ventilation Holes**: Kraft cardboard boxes have round punch holes that mimic pipe faces. **Outer Margin Chromatic Analysis** evaluates color signatures ($R > 110, R - B > 18, G - B > 6$) to reject cardboard background false alarms.
3. **Dark Shadow Cavities & Watermarks**: Pitch-black gaps between receding tubes and watermark stamps are rejected via relative contrast and size distribution checks.
4. **Axial Cylinder Bodies**: Elongated side tubes in perspective angles are suppressed when circular pipe end-faces are present, preserving real neighbor pipes in touching grids.

---

## 📦 Standalone Python Integration

To use the pipe detection model in any Python script:

```python
from insulation_detector import InsulationPipeDetector

# Automatically loads ONNX (or PyTorch if available)
detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_clean.onnx")

# Run detection on any image format (.jpg, .png, .jfif, .gfif, .webp, .bmp, etc.)
annotated_img, detections = detector.detect("box_touching_grid_70.jpg", conf=0.35)

print(f"Total pipes detected: {len(detections)}")
```
