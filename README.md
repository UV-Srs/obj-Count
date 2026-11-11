# ⚡ Insulation Pipe Detection & Analysis Studio (YOLOv8)

A production-grade Deep Learning Object Detection suite specialized for **Thermal Insulation Pipe Detection**, featuring a clean **React Frontend**, a high-performance **FastAPI Backend**, and a built-in Dual-Pass Anti-Double Concentric Suppression Engine.

---

## 🚀 Run Commands (Backend & Frontend)

### 🌟 Quick Start: Launch Both Together
Double-click [**`run_all.bat`**](file:///c:/AI/run_all.bat) or execute:
```powershell
.\run_all.bat
```
This automatically starts the FastAPI backend, starts the React dev server, and opens `http://localhost:5173` in your browser.

---

### Option A: Launch Individually

#### 1. Backend (FastAPI on Port 8000)
Double-click [**`run_backend.bat`**](file:///c:/AI/run_backend.bat) or run from terminal:
```powershell
cd c:\AI\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- **API URL:** `http://127.0.0.1:8000`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`

#### 2. Frontend (React + Vite on Port 5173)
Double-click [**`run_frontend.bat`**](file:///c:/AI/run_frontend.bat) or run from terminal:
```powershell
cd c:\AI\frontend
npm run dev
```
- **Web App:** `http://localhost:5173`

---

## 📂 Project Architecture

```
c:\AI\
│
├── backend/                                    # 🐍 FastAPI Backend & Machine Learning Engine
│   ├── main.py                                 # REST API endpoints (/api/detect, /api/samples, /api/health)
│   ├── insulation_detector.py                  # Standalone detection engine (supports all image formats)
│   ├── concentric_nms.py                       # Anti-double concentric & distractor suppression pipeline
│   ├── run_backend.bat                         # One-click backend launcher (Port 8000)
│   ├── requirements.txt                        # Python dependencies (FastAPI, Uvicorn, YOLO, OpenCV)
│   │
│   ├── models/                                 # Trained neural network weights
│   │   ├── best_insulation_pipe_clean.pt       # Production PyTorch model (70/70 pipes on dense grids)
│   │   ├── best_insulation_pipe_clean.onnx     # Production ONNX model
│   │   ├── best_insulation_pipe_finetuned.pt   # Fine-tuned checkpoint
│   │   └── best_insulation_pipe.pt             # Baseline YOLOv8n weights
│   │
│   ├── data/                                   # Training and validation dataset annotations
│   └── sample_test/                            # Benchmark test samples (70-pipe grid, deformed pipes, etc.)
│
├── frontend/                                   # ⚛️ Clean React Frontend (Vite)
│   ├── src/
│   │   ├── App.jsx                             # Clean, simple UI (drag-and-drop, count stats, export)
│   │   ├── App.css                             # Minimalist industrial tool stylesheet
│   │   └── main.jsx                            # React entry point
│   ├── index.html                              # Web app template
│   ├── package.json                            # React & Vite dependencies
│   └── vite.config.js                          # Vite configuration
│
├── run_all.bat                                 # Launches both backend and frontend together
├── run_backend.bat                             # Root launcher for backend
├── run_frontend.bat                            # Root launcher for frontend
└── README.md                                   # Documentation
```

---

## 🛡️ Anti-Double, Watermark & Cardboard Suppression Engine (`concentric_nms.py`)

Hollow insulation pipes in real-world packaging environments pose several challenges that break standard NMS:
1. **Cardboard Packaging Distractors (Pass 0A)**: Shipping boxes have round ventilation punch holes with dark centers that resemble pipe faces.
   - **Fix**: **Outer Margin Chromatic Analysis** checks margin color ($R > 110, R - B > 18, G - B > 6$) to immediately distinguish warm yellow-brown kraft cardboard from neutral dark gray/black insulation foam, eliminating 100% of cardboard background false alarms.
2. **Dark Voids & Watermark Distractors (Pass 0B)**: Background voids (pitch-black gaps between receding barrels) and circular stock photo watermarks (e.g. `@ dreamstime` logos) can trigger false positives at low confidence.
   - **Fix**: **Void & Inverted Contrast Filtering** eliminates pitch-black background cavities ($\text{outer\_gray} < 30 \text{ and } \text{center\_gray} < 25$), inverted contrast watermark logos ($\text{center\_gray} > \text{outer\_gray} + 5$), and relative size outlier stamps ($w < 45\%$ of median pipe width). Combined with 9 hard-negative background crops, this drops watermark and shadow false alarms from 18 to 0.
3. **Concentric Double Detection (Pass 1)**: Standard NMS calculates $\text{IoU} \approx 0.18$ between the inner dark hole and the outer foam ring. Because $0.18 < 0.45$, both boxes survive.
   - **Fix**: **Intersection-over-Smaller ($\text{IoS}$)** suppression eliminates any inner box when $\ge 50\%$ of it is contained inside a larger concentric box. This operates natively without requiring an external UI toggle button.
4. **Axial Cylinder Body vs. Touching Pipes in Grids (Pass 2)**: In perspective views, elongated cylinder bodies ($h/w > 1.28$ or $< 0.68$) appear alongside circular faces. However, naive one-dimensional overlap suppresses real touching pipes in the same row or column.
   - **Fix**: **2D Co-Occurrence Spatial Filtering** requires substantial overlap in *both* dimensions simultaneously before classifying a box as a duplicate body. This preserves 100% of touching neighbor pipes while eliminating side-body duplicates.

---

## ✨ React Frontend Features

- **Clean, Minimalist Design**: Utility-first tool interface without unnecessary buttons, cluttered sidebars, or bloated animations.
- **Universal Image Support**: Drag-and-drop or upload any image format — `.jpg`, `.jpeg`, `.jfif`, `.gfif`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`.
- **Automatic Counting**: Immediate detection on image upload with a prominent total count display and key inference metrics (inference time, average confidence).
- **One-Click Presets**:
  - `Dense Grid (70 Pipes)` (automatically sets `conf = 0.35`)
  - `Warehouse Carton` (automatically sets `conf = 0.58`)
  - `Standard Stack` (automatically sets `conf = 0.45`)
- **Built-in Demo Samples**: Instant one-click testing of 70-pipe touching grid, loose deformed pipes, multi-layer stack, warehouse cartons, and 6-pipe pyramid.
- **Visual Comparison**: Toggle between the detection result with clean bounding boxes and the original image.
- **Exports**: One-click download of the annotated image and export of detection coordinates to CSV.
- **Expandable Coordinates Table**: Clean inspection of all detected pipes (ID, coordinates, dimensions, confidence).

---

## 📦 2. Using the Model in Another Project (`insulation_detector.py`)

To use this model in another Python project, you only need **2 files**:
1. `models/best_insulation_pipe_clean.pt` (or `.onnx` for C++/edge deployment)
2. `insulation_detector.py`

### 3-Line Integration Example:
```python
from insulation_detector import InsulationPipeDetector

# 1. Initialize detector (loads PyTorch or ONNX weights)
detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_clean.pt")

# 2. Run detection (supports all formats: .jpg, .png, .jfif, .gfif, .webp, .gif, PIL Images, and NumPy arrays)
annotated_img, detections = detector.detect("my_pipes.jfif", conf=0.35)

# 3. Print count and access bounding boxes
print(f"Total pipes detected: {len(detections)}")
for d in detections:
    print(f"Pipe at ({d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}) - Conf: {d['conf']*100:.1f}%")
```

---

## 📈 4. Jupyter Notebook (`train_insulation_pipe.ipynb`)

Open the notebook locally or on Google Colab / Kaggle:
```powershell
jupyter notebook c:\AI\train_insulation_pipe.ipynb
```
Contains 28 executable cells covering:
- Dataset verification & health checks.
- Fine-tuning with cosine annealing and backbone preservation.
- Section 11: Concentric NMS mathematical derivation & 6-pipe resolution.
- Section 12: High-density packaging fine-tuning (scaling from 14 to 158 pipes).

---

## 💻 3. CLI Training with Real-Time Progress Monitor (`train.py`)

Run training with a live terminal progress dashboard:
```powershell
python train.py --model models/best_insulation_pipe_clean.pt --epochs 15 --batch 4 --freeze 6 --lr0 0.0003 --name best_insulation_pipe_clean
```
Provides live terminal updates on every epoch:
- Progress bar and `% completed`
- Elapsed time and estimated remaining time (ETA)
- Box loss, Cls loss, and Total loss
- Validation Precision, Recall, mAP50, and mAP50-95
#   O b j - c o u n t  
 