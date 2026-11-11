"""
Insulation Pipe Detection - FastAPI Backend
Provides clean REST API endpoints for image detection, sample loading, and exports.
"""

import os
import sys
import time
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
import numpy as np
import cv2

# Directory paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from insulation_detector import InsulationPipeDetector

# Register .jfif and .gfif with Pillow
try:
    Image.register_extension("JPEG", ".gfif")
    Image.register_extension("JPEG", ".jfif")
except Exception:
    pass

app = FastAPI(
    title="Insulation Pipe Detection API",
    version="2.0.0",
    description="Backend API for insulation pipe counting and defect detection.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Enable CORS for React frontend (Vite default is 5173, CRA is 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
@app.get("/api/")
def api_root():
    return {
        "message": "Insulation Pipe Detection API is running",
        "docs": "/api/docs",
        "health": "/api/health"
    }

# Initialize detector with default production weights (prefer ONNX for ultra-lightweight serverless)
ONNX_MODEL = BACKEND_DIR / "models" / "best_insulation_pipe_clean.onnx"
PT_MODEL = BACKEND_DIR / "models" / "best_insulation_pipe_clean.pt"
MODEL_PATH = str(ONNX_MODEL if ONNX_MODEL.exists() else PT_MODEL)
SAMPLE_DIR = BACKEND_DIR / "sample_test"

detector = None

def get_detector() -> InsulationPipeDetector:
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        detector = InsulationPipeDetector(MODEL_PATH)
    return detector

PRESET_CONFIGS = {
    "auto": {"conf": 0.35, "iou": 0.45, "name": "Auto (Optimized)"},
    "grid": {"conf": 0.35, "iou": 0.45, "name": "Dense Touching Grid"},
    "carton": {"conf": 0.58, "iou": 0.45, "name": "Warehouse Carton"},
    "stack": {"conf": 0.45, "iou": 0.45, "name": "Standard Bundle / Stack"},
}

SAMPLE_IMAGES = [
    {
        "id": "touching_grid",
        "name": "70-Pipe Touching Grid",
        "filename": "box_touching_grid_70.jpg",
        "preset": "grid",
        "description": "Dense grid of 70 touching pipes inside a carton"
    },
    {
        "id": "deformed_pipes",
        "name": "Loose Deformed Pipes",
        "filename": "box_deformed_pipes_clean.jpg",
        "preset": "grid",
        "description": "Loose, squished and oval insulation pipes"
    },
    {
        "id": "pipe_stack",
        "name": "Multi-Layer Stack",
        "filename": "pipe_stack_dreamstime.jpg",
        "preset": "stack",
        "description": "Warehouse multi-layer stack (watermark suppression test)"
    },
    {
        "id": "warehouse_box",
        "name": "Warehouse Box",
        "filename": "warehouse_box_uploaded.jpg",
        "preset": "carton",
        "description": "High-density warehouse carton packaging"
    },
    {
        "id": "crop_6",
        "name": "6-Pipe Pyramid",
        "filename": "crop_6_pipes.jpg",
        "preset": "stack",
        "description": "Standard 6-pipe pyramid bundle"
    }
]


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image to Base64 JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Could not encode image to JPEG")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes of any format (jfif, gfif, png, webp, bmp, gif, etc.) to RGB."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image format: {e}")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model": "YOLOv8 Insulation Pipe Clean",
        "supported_formats": [
            "jpg", "jpeg", "jfif", "gfif", "png", "webp",
            "bmp", "tiff", "tif", "gif", "avif", "ico"
        ]
    }


@app.get("/api/samples")
def get_samples():
    """Return list of demo sample images."""
    available = []
    for s in SAMPLE_IMAGES:
        fpath = SAMPLE_DIR / s["filename"]
        if fpath.exists():
            available.append(s)
    return available


@app.get("/api/sample-image/{sample_id}")
def get_sample_image(sample_id: str):
    """Serve a sample image directly."""
    for s in SAMPLE_IMAGES:
        if s["id"] == sample_id:
            fpath = SAMPLE_DIR / s["filename"]
            if not fpath.exists():
                raise HTTPException(status_code=404, detail="Sample image not found on disk")
            with open(fpath, "rb") as f:
                content = f.read()
            return Response(content=content, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample ID not found")


def process_image_bytes(
    file_bytes: bytes,
    preset: str = "auto",
    conf: Optional[float] = None,
    iou: Optional[float] = None
) -> Dict[str, Any]:
    """Core detection logic on raw image bytes."""
    det = get_detector()

    # Determine confidence threshold
    target_conf = conf
    target_iou = iou
    if target_conf is None or target_conf <= 0:
        cfg = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["auto"])
        target_conf = cfg["conf"]
    if target_iou is None or target_iou <= 0:
        target_iou = 0.45

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")

    rgb_img = decode_uploaded_image(file_bytes)
    h, w = rgb_img.shape[:2]

    # Run detection
    t0 = time.time()
    annotated_rgb, detections = det.detect(
        rgb_img,
        conf=target_conf,
        iou=target_iou,
        imgsz=640
    )
    t_elapsed_ms = round((time.time() - t0) * 1000, 1)

    # Convert annotated RGB to BGR for JPEG encoding
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    annotated_b64 = image_to_base64(annotated_bgr)

    # Compute confidence metrics
    confs = [d["conf"] for d in detections]
    highest_conf = round(max(confs), 3) if confs else 0.0
    avg_conf = round(float(np.mean(confs)), 3) if confs else 0.0

    # Format detection coordinates
    formatted_dets = []
    for i, d in enumerate(detections, 1):
        formatted_dets.append({
            "id": i,
            "x1": d["x1"],
            "y1": d["y1"],
            "x2": d["x2"],
            "y2": d["y2"],
            "width": d["x2"] - d["x1"],
            "height": d["y2"] - d["y1"],
            "conf": round(d["conf"], 3),
        })

    return {
        "count": len(detections),
        "metrics": {
            "highest_conf": highest_conf,
            "avg_conf": avg_conf,
            "inference_time_ms": t_elapsed_ms,
            "image_width": w,
            "image_height": h,
            "conf_threshold": target_conf,
            "preset": preset
        },
        "annotated_image": annotated_b64,
        "detections": formatted_dets
    }


@app.post("/api/detect")
async def detect_pipes(
    file: UploadFile = File(...),
    preset: str = Form("auto"),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None)
):
    """
    Detect insulation pipes in an uploaded image.
    Supports all formats (.jfif, .gfif, .webp, .png, .jpg, .bmp, .tiff, .gif).
    """
    file_bytes = await file.read()
    return process_image_bytes(
        file_bytes=file_bytes,
        preset=preset,
        conf=conf,
        iou=iou
    )


@app.post("/api/detect-sample")
async def detect_sample(
    sample_id: str = Form(...),
    preset: Optional[str] = Form(None),
    conf: Optional[float] = Form(None)
):
    """Run pipe detection directly on a built-in demo sample."""
    sample_obj = None
    for s in SAMPLE_IMAGES:
        if s["id"] == sample_id:
            sample_obj = s
            break

    if not sample_obj:
        raise HTTPException(status_code=404, detail="Sample ID not found")

    fpath = SAMPLE_DIR / sample_obj["filename"]
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Sample file missing from disk")

    chosen_preset = preset if preset and preset in PRESET_CONFIGS else sample_obj["preset"]
    with open(fpath, "rb") as f:
        file_bytes = f.read()

    return process_image_bytes(
        file_bytes=file_bytes,
        preset=chosen_preset,
        conf=conf
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
