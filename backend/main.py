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
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from PIL import Image, ImageOps
import numpy as np
import cv2
import uuid
import shutil

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

# Upload limits (protects server memory and blocks accidental massive uploads)
MAX_IMAGE_SIZE_MB = 25
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".gfif", ".png", ".webp", ".bmp", ".tiff", ".tif"}

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

# Model Architecture Registry
# ACTIVE_MODEL_PROFILE: "yolov8m" (High accuracy, 1024px) or "yolov8n" (Ultra-fast baseline, 640px)
# Can be changed here or set via PIPE_MODEL_PROFILE environment variable.
ACTIVE_MODEL_PROFILE = os.environ.get("PIPE_MODEL_PROFILE", "yolov8m").lower().strip()

MODEL_PROFILES = {
    "yolov8m": {
        "name": "YOLOv8 Medium (1024px)",
        "onnx": BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.onnx",
        "pt": BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.pt",
        "imgsz": 1024,
    },
    "yolov8n": {
        "name": "YOLOv8 Nano (640px)",
        "onnx": BACKEND_DIR / "models" / "best_insulation_pipe_clean.onnx",
        "pt": BACKEND_DIR / "models" / "best_insulation_pipe_clean.pt",
        "imgsz": 640,
    }
}

SAMPLE_DIR = BACKEND_DIR / "sample_test"

detector = None
current_profile_in_use = None

def resolve_model(profile_name: Optional[str] = None) -> Tuple[str, int, str]:
    """
    Resolve active model weights and input resolution.
    Gracefully falls back to yolov8n baseline if yolov8m weights are not yet present.
    """
    target = (profile_name or ACTIVE_MODEL_PROFILE).lower().strip()
    profile = MODEL_PROFILES.get(target, MODEL_PROFILES["yolov8m"])

    # Check for target model weights (prefer ONNX)
    if profile["onnx"].exists():
        return str(profile["onnx"]), profile["imgsz"], target
    if profile["pt"].exists():
        return str(profile["pt"]), profile["imgsz"], target

    # Seamless fallback to yolov8n baseline
    baseline = MODEL_PROFILES["yolov8n"]
    if baseline["onnx"].exists():
        return str(baseline["onnx"]), baseline["imgsz"], "yolov8n"
    if baseline["pt"].exists():
        return str(baseline["pt"]), baseline["imgsz"], "yolov8n"

    raise RuntimeError(f"No pipe detection weights found in {BACKEND_DIR / 'models'}")

def get_detector(profile_name: Optional[str] = None) -> InsulationPipeDetector:
    global detector, current_profile_in_use
    target_profile = (profile_name or ACTIVE_MODEL_PROFILE).lower().strip()

    if detector is None or (profile_name and profile_name != current_profile_in_use):
        model_path, imgsz, actual_profile = resolve_model(target_profile)
        detector = InsulationPipeDetector(model_path)
        current_profile_in_use = actual_profile
        print(f"[Model Registry] Active detector profile: {actual_profile} | Model: {model_path} | Native imgsz: {detector.native_imgsz}")
    return detector

PRESET_CONFIGS = {
    "auto": {"conf": 0.04, "iou": 0.45, "name": "Auto (Optimized Static)"},
    "grid": {"conf": 0.04, "iou": 0.45, "name": "Dense Touching Grid"},
    "carton": {"conf": 0.04, "iou": 0.45, "name": "Warehouse Carton"},
    "stack": {"conf": 0.04, "iou": 0.45, "name": "Standard Bundle / Stack"},
    "pyramid": {"conf": 0.04, "iou": 0.45, "name": "Pyramid Bundle"},
    "metal_dense": {"conf": 0.04, "iou": 0.45, "name": "Dense Steel & Scaffolding"},
    "strict": {"conf": 0.15, "iou": 0.45, "name": "Strict High-Precision"},
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
        "preset": "pyramid",
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
    det = get_detector()
    return {
        "status": "online",
        "model": f"YOLOv8 ({current_profile_in_use or ACTIVE_MODEL_PROFILE})",
        "imgsz": det.native_imgsz,
        "use_onnx": det.use_onnx,
        "supported_formats": [
            "jpg", "jpeg", "jfif", "gfif", "png", "webp",
            "bmp", "tiff", "tif", "gif", "avif", "ico"
        ]
    }


@app.get("/api/model/info")
def get_model_info():
    """Return currently active model profile, architecture details, and switchable profiles."""
    det = get_detector()
    return {
        "active_profile": current_profile_in_use or ACTIVE_MODEL_PROFILE,
        "configured_profile": ACTIVE_MODEL_PROFILE,
        "native_imgsz": det.native_imgsz,
        "model_path": det.model_path,
        "use_onnx": det.use_onnx,
        "available_profiles": {
            k: {
                "name": v["name"],
                "installed": v["onnx"].exists() or v["pt"].exists(),
                "imgsz": v["imgsz"],
                "weights": str(v["onnx"] if v["onnx"].exists() else v["pt"]) if (v["onnx"].exists() or v["pt"].exists()) else None
            }
            for k, v in MODEL_PROFILES.items()
        }
    }


@app.post("/api/model/switch")
async def switch_model_profile(
    request: Request,
    profile: Optional[str] = Form(None)
):
    """
    Switch active model dynamically between 'yolov8m' and 'yolov8n' without server restart.
    Accepts form data, JSON payload, or query parameter.
    """
    global ACTIVE_MODEL_PROFILE, detector, current_profile_in_use

    target = profile
    if not target:
        target = request.query_params.get("profile")
    if not target and request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            target = body.get("profile")
        except Exception:
            pass

    if not target:
        raise HTTPException(
            status_code=400,
            detail="Missing 'profile' parameter. Choose 'yolov8m' or 'yolov8n'."
        )

    target = target.lower().strip()
    if target not in MODEL_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile '{target}'. Choose from: {list(MODEL_PROFILES.keys())}"
        )

    ACTIVE_MODEL_PROFILE = target
    detector = None
    new_det = get_detector(target)
    return {
        "success": True,
        "requested_profile": target,
        "active_profile": current_profile_in_use,
        "native_imgsz": new_det.native_imgsz,
        "model_path": new_det.model_path,
        "model_name": MODEL_PROFILES[target]["name"]
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
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")

    # Enforce image upload limit (25 MB)
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {MAX_IMAGE_SIZE_MB} MB."
        )

    det = get_detector()

    # Determine confidence threshold
    target_conf = conf
    target_iou = iou
    if target_conf is None or target_conf <= 0:
        cfg = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["auto"])
        target_conf = cfg["conf"]
    if target_iou is None or target_iou <= 0:
        target_iou = 0.45

    rgb_img = decode_uploaded_image(file_bytes)
    h, w = rgb_img.shape[:2]

    # Run detection
    t0 = time.time()
    annotated_rgb, detections = det.detect(
        rgb_img,
        conf=target_conf,
        iou=target_iou,
        imgsz=None
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
            "preset": preset,
            "model_profile": current_profile_in_use or ACTIVE_MODEL_PROFILE,
            "model_imgsz": det.native_imgsz,
            "model_name": MODEL_PROFILES.get(current_profile_in_use or ACTIVE_MODEL_PROFILE, {}).get("name", "YOLOv8")
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
    Enforces 25 MB file size limit.
    """
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            )

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


# Serve built frontend single-page application if present
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if not FRONTEND_DIST.exists():
    FRONTEND_DIST = BACKEND_DIR / "static"

if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
