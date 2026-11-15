"""
Local Deep Fine-Tuning Script for YOLOv8m (1024px)
Fine-tunes best_insulation_pipe_m_1024.pt on local CPU with frozen backbone to maximize accuracy on purified dataset.
"""

import os
import sys
import time
import shutil
import glob
from pathlib import Path
import torch
from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MODEL_PATH = str(BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.pt")
DATA_YAML = str(PROJECT_ROOT / "data" / "data.yaml") if (PROJECT_ROOT / "data" / "data.yaml").exists() else str(BACKEND_DIR / "data" / "data.yaml")

def run_deep_training(epochs: int = 25, batch: int = 4, imgsz: int = 1024, freeze: int = 12):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Dataset config not found at: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70)
    print("🚀 YOLOv8m (1024px) DEEP MULTI-EPOCH LOCAL FINE-TUNING")
    print("=" * 70)
    print(f"  Warm Checkpoint : {MODEL_PATH}")
    print(f"  Dataset Config  : {DATA_YAML}")
    print(f"  Resolution      : {imgsz}x{imgsz}")
    print(f"  Target Epochs   : {epochs}")
    print(f"  Batch Size      : {batch}")
    print(f"  Compute Device  : {device}")
    print(f"  Frozen Layers   : {freeze} (Backbone frozen for CPU speed & stability)")
    print("=" * 70 + "\n")

    model = YOLO(MODEL_PATH)
    t0 = time.time()

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        patience=10,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.0001,             # Gentle fine-tuning rate
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=0,        # No warmup shock: preserve pretrained representations
        box=9.0,                # High box localization penalty for oval/squished pipes
        cls=1.0,
        dfl=1.5,
        degrees=8.0,            # Slight tilt invariance
        shear=4.0,              # Oval deformation invariance for packed cartons
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        close_mosaic=2,         # Clean convergence on real shapes
        save=True,
        save_period=2,
        project="runs/train",
        name="yolov8m_1024_local_deep",
        exist_ok=True
    )

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\n✅ Local training finished in {elapsed_min:.1f} minutes ({elapsed_min/60.0:.2f} hours)!")

    # Locate Best Weights
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        candidates = list(PROJECT_ROOT.glob("runs/**/yolov8m_1024_local_deep/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]

    if best_weights.exists():
        print(f"\n🎯 Best weights saved at: {best_weights}")
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=DATA_YAML, imgsz=imgsz, device=device)
        print("\n" + "=" * 50)
        print("🎯 VALIDATION METRICS:")
        print(f"   Precision : {metrics.box.p[0]*100:.2f}%")
        print(f"   Recall    : {metrics.box.r[0]*100:.2f}%")
        print(f"   mAP@50    : {metrics.box.map50*100:.2f}%")
        print(f"   mAP@50-95 : {metrics.box.map*100:.2f}%")
        print("=" * 50)

        # Export ONNX (1024px)
        print("\n⚡ Exporting optimized ONNX at 1024px resolution...")
        onnx_file = best_model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)

        # Copy to production models folder
        prod_pt = BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.pt"
        prod_onnx = BACKEND_DIR / "models" / "best_insulation_pipe_m_1024.onnx"
        shutil.copy(str(best_weights), str(prod_pt))
        shutil.copy(str(onnx_file), str(prod_onnx))
        print(f"✅ Successfully deployed refined weights directly to:\n   - {prod_pt}\n   - {prod_onnx}")

        # Post-training benchmark
        train_img_dir = BACKEND_DIR / "data" / "images" / "train"
        print("\n🧪 MULTI-SCENARIO BENCHMARK VERIFICATION:")
        candidates = [
            ("70-Pipe Grid Carton", train_img_dir / "box_touching_grid_70.jpg", 0.35, 70),
            ("29-Pipe Stack",       train_img_dir / "pipe_stack_dreamstime.jpg", 0.28, 29),
            ("INSUFLEX Thick Carton", train_img_dir / "insuflex_carton_thick.jpg", 0.28, 67),
            ("INSUFLEX Thin Carton",  train_img_dir / "insuflex_carton_thin.jpg", 0.28, 161),
        ]
        for title, fpath, conf_val, expected in candidates:
            if fpath.exists():
                res = best_model.predict(source=str(fpath), conf=conf_val, imgsz=1024, verbose=False)[0]
                det_count = len(res.boxes)
                print(f"   {title:25s} | Detected: {det_count:3d} / {expected:3d}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--freeze", type=int, default=12)
    args = parser.parse_args()

    run_deep_training(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, freeze=args.freeze)
