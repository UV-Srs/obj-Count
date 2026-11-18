import os
import sys
from pathlib import Path

# Robust root path resolution for both IDE analyzer and runtime from any cwd
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.insulation_detector import InsulationPipeDetector
except ImportError:
    # pyrefly: ignore[missing-import]
    from insulation_detector import InsulationPipeDetector

import cv2

model_path = str(BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.onnx")
detector = InsulationPipeDetector(model_path=model_path)

print("="*80)
print("            🚀 PRODUCTION MODEL VERIFICATION BENCHMARK REPORT")
print("="*80)

# 1. Negative Zero-False-Alarm Tests
neg_tests = [
    ('c:/AI/data/images/val/neg_user_construction.jpg', 'Construction site workers in helmets'),
    ('c:/AI/data/images/val/neg_user_car.jpg', 'Car with passengers & steering wheel'),
    ('c:/AI/data/images/val/neg_user_desk.jpg', 'Office desk & laptop'),
    ('c:/AI/data/images/val/neg_user_office_chairs.jpg', 'Office chairs & carpet'),
    ('c:/AI/data/images/val/neg_user_hand_toothbrush_val.jpg', 'Hand holding toothbrush'),
    ('c:/AI/data/images/val/neg_user_puja_bananas_fruits_val.jpg', 'Puja bananas & fruit baskets'),
    ('c:/AI/data/images/val/neg_user_desk_bottles_room_clutter_val.jpg', 'Room desk bottles & clutter'),
]

print("\n🛡️ 1. ZERO-DETECTION FALSE POSITIVE TESTS (Target: 0 Pipes Detected):")
total_fps = 0
for path_str, desc in neg_tests:
    p = Path(path_str)
    if not p.exists():
        p = Path(path_str.replace('/val/', '/train/'))
    _, det = detector.detect(str(p), conf=0.10, iou=0.45)
    cnt = len(det)
    total_fps += cnt
    status = "✅ PASS (0 detected)" if cnt == 0 else f"⚠️ {cnt} FALSE POSITIVES"
    print(f"  {desc:<40} | Detected: {cnt:>2} | {status}")

print(f"  --> Total False Positives: {total_fps}/7 (Zero Alarm Rate: {((7-total_fps)/7)*100:.1f}%)")

# 2. Positive Real Warehouse Carton Tests
carton_tests = [
    ('c:/AI/data/images/train/insuflex_carton_thick.jpg', 67, 'Insuflex Thick Foam Carton (67 Pipes)'),
    ('c:/AI/data/images/train/insuflex_carton_thin.jpg', 158, 'Insuflex Thin Foam Carton (158 Pipes)'),
    ('c:/AI/data/images/train/box_touching_grid_70.jpg', 70, 'Touching Grid Carton (70 Pipes)'),
]

print("\n📦 2. REAL PACKAGING CARTON ACCURACY TESTS:")
for path_str, gt, desc in carton_tests:
    p = Path(path_str)
    if p.exists():
        _, det = detector.detect(str(p), conf=0.10, iou=0.45)
        cnt = len(det)
        acc = max(0.0, 1.0 - abs(cnt - gt) / float(gt)) * 100.0
        print(f"  {desc:<40} | GT: {gt:>3} | Count: {cnt:>3} | Accuracy: {acc:6.2f}%")

print("\n" + "="*80)

