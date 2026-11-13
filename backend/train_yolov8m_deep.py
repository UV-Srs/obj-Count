"""
Local Deep Fine-Tuning Script for YOLOv8m (1024px)
Fine-tunes best_insulation_pipe_m_1024.pt on local CPU with frozen backbone to accelerate convergence.
"""

import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MODEL_PATH = str(BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.pt")
DATA_YAML = str(BACKEND_DIR / "data" / "data.yaml")

def run_deep_training(epochs: int = 25, batch: int = 2, imgsz: int = 1024, freeze: int = 10):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Dataset config not found at: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70)
    print("🚀 YOLOv8m (1024px) IN-DEPTH LOCAL FINE-TUNING")
    print("=" * 70)
    print(f"  Warm Checkpoint : {MODEL_PATH}")
    print(f"  Dataset Config  : {DATA_YAML}")
    print(f"  Resolution      : {imgsz}x{imgsz}")
    print(f"  Epochs          : {epochs}")
    print(f"  Batch Size      : {batch}")
    print(f"  Compute Device  : {device}")
    print(f"  Freeze Layers   : {freeze} (Backbone frozen for speed & feature stability)")
    print("=" * 70 + "\n")

    model = YOLO(MODEL_PATH)
    t0 = time.time()

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        patience=20,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        cos_lr=True,
        label_smoothing=0.05,
        box=7.5,
        cls=2.0,
        dfl=1.5,
        degrees=15.0,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,
        flipud=0.2,
        mosaic=1.0,
        mixup=0.15,
        erasing=0.2,
        close_mosaic=10,
        save=True,
        project="runs/train",
        name="yolov8m_1024_local_deep",
        exist_ok=True
    )

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\n✅ Local training finished in {elapsed_min:.1f} minutes!")

    # Validate
    best_weights = str(Path(results.save_dir) / "weights" / "best.pt")
    if not os.path.exists(best_weights):
        best_weights = str(PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / "yolov8m_1024_local_deep" / "weights" / "best.pt")
    if not os.path.exists(best_weights):
        best_weights = str(BACKEND_DIR / "runs" / "train" / "yolov8m_1024_local_deep" / "weights" / "best.pt")

    if os.path.exists(best_weights):
        print(f"Best weights saved at: {best_weights}")
        best_model = YOLO(best_weights)
        metrics = best_model.val(data=DATA_YAML, imgsz=imgsz, device=device)
        print("\n" + "=" * 50)
        print("🎯 VALIDATION METRICS:")
        print(f"   Precision : {metrics.box.p[0]*100:.2f}%")
        print(f"   Recall    : {metrics.box.r[0]*100:.2f}%")
        print(f"   mAP@50    : {metrics.box.map50*100:.2f}%")
        print(f"   mAP@50-95 : {metrics.box.map*100:.2f}%")
        print("=" * 50)

        # Export ONNX
        print("\n⚡ Exporting optimized ONNX at 1024px...")
        onnx_file = best_model.export(format="onnx", imgsz=imgsz, optimize=True, simplify=True)

        # Copy to production models folder
        prod_pt = BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.pt"
        prod_onnx = BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.onnx"
        import shutil
        shutil.copy(best_weights, str(prod_pt))
        shutil.copy(onnx_file, str(prod_onnx))
        print(f"✅ Deployed refined weights directly to {prod_pt} and {prod_onnx}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--freeze", type=int, default=10)
    args = parser.parse_args()

    run_deep_training(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, freeze=args.freeze)

