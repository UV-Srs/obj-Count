"""
YOLOv8 Insulation Pipe Detection Fine-Tuning CLI Script
With Real-Time Epoch Progress Monitor, ETA Tracker, and Loss Dashboard.
"""

import os
import time
import argparse
import torch
from ultralytics import YOLO

def extract_loss_val(container, key, default_idx=0):
    """Extract scalar loss value safely across ultralytics versions."""
    try:
        if isinstance(container, dict):
            val = container.get(key, 0.0)
            return float(val.item()) if hasattr(val, "item") else float(val)
        elif isinstance(container, (list, tuple)):
            if len(container) > default_idx:
                val = container[default_idx]
                return float(val.item()) if hasattr(val, "item") else float(val)
    except Exception:
        pass
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune YOLOv8 for Insulation Pipe Detection")
    parser.add_argument("--model", type=str, default="models/best_insulation_pipe.pt", help="Path to initial weights")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (640, 960, or 1280)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam"], help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--cls", type=float, default=2.0, help="Classification loss weight")
    parser.add_argument("--name", type=str, default="insulation_pipe_finetuned", help="Experiment name")
    parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    args = parser.parse_args()

    if not args.device:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n" + "=" * 78)
    print("🚀 YOLOv8 INSULATION PIPE DETECTION - REAL-TIME TRAINING MONITOR")
    print("=" * 78)
    print(f"  Base Checkpoint : {args.model}")
    print(f"  Dataset Config  : {args.data}")
    print(f"  Resolution      : {args.imgsz}x{args.imgsz}")
    print(f"  Total Epochs    : {args.epochs}")
    print(f"  Batch Size      : {args.batch}")
    print(f"  Compute Device  : {device}")
    print(f"  Freeze Layers   : {args.freeze} (Backbone preservation)")
    print(f"  Optimizer       : {args.optimizer} (cos_lr=True, lr0={args.lr0})")
    print("=" * 78 + "\n")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    model = YOLO(args.model)
    t_start = time.time()

    # Real-Time Callback for Live Epoch Progress
    def live_epoch_progress_callback(trainer):
        ep = trainer.epoch + 1
        tot = trainer.epochs
        if ep > tot:
            return
        pct = (ep / float(tot)) * 100
        now = time.time()
        elapsed = now - t_start
        avg_per_ep = elapsed / max(1, ep)
        remaining_eps = tot - ep
        eta_seconds = max(0, remaining_eps * avg_per_ep)

        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

        # Extract losses
        loss_dict = getattr(trainer, "loss_items", getattr(trainer, "tloss", {}))
        loss_box = extract_loss_val(loss_dict, "box_loss", 0)
        loss_cls = extract_loss_val(loss_dict, "cls_loss", 1)
        loss_dfl = extract_loss_val(loss_dict, "dfl_loss", 2)
        total_loss = loss_box + loss_cls + loss_dfl

        metrics = getattr(trainer, "metrics", {})
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0)) * 100
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0)) * 100
        prec = float(metrics.get("metrics/precision(B)", 0.0)) * 100
        rec = float(metrics.get("metrics/recall(B)", 0.0)) * 100

        # Beautiful live terminal dashboard
        bar_len = 25
        filled = int(bar_len * (ep / tot))
        bar = "█" * filled + "░" * (bar_len - filled)

        print("\n" + "┌" + "─" * 76 + "┐")
        print(f"│ ⚡ EPOCH {ep:>2}/{tot:<2}  [{bar}] {pct:>5.1f}% DONE  │  {remaining_eps:>2} Epochs Left │")
        print("├" + "─" * 76 + "┤")
        print(f"│ ⏱️  Elapsed: {elapsed_str}  │  ⏳ Est. Remaining (ETA): {eta_str}                           │")
        print(f"│ 📉 Box Loss: {loss_box:>6.3f}  │  Cls Loss: {loss_cls:>6.3f}  │  Total Loss: {total_loss:>6.3f}           │")
        print(f"│ 🎯 Precision: {prec:>5.1f}%  │  Recall: {rec:>5.1f}%  │  mAP50: {map50:>5.1f}%  │  mAP50-95: {map50_95:>5.1f}% │")
        print("└" + "─" * 76 + "┘\n")

    model.add_callback("on_fit_epoch_end", live_epoch_progress_callback)

    # Execute training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        freeze=args.freeze,
        cls=args.cls,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        label_smoothing=0.05,
        degrees=15.0,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,
        flipud=0.3,
        mosaic=1.0,
        mixup=0.15,
        erasing=0.3,
        close_mosaic=5,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=False
    )

    t_total = time.time() - t_start
    mins, secs = divmod(int(t_total), 60)

    print("\n" + "=" * 78)
    print(f"🎉 TRAINING COMPLETE in {mins}m {secs}s!")
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    print(f"  Best Checkpoint Saved: {best_weights}")

    target = os.path.join("models", f"{args.name}.pt")
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target)
        print(f"  Deployed to Frontend : {target}")
        try:
            onnx_model = YOLO(target)
            onnx_path = onnx_model.export(format="onnx", imgsz=args.imgsz)
            onnx_target = os.path.join("models", f"{args.name}.onnx")
            if os.path.exists(onnx_path) and onnx_path != onnx_target:
                shutil.copy(onnx_path, onnx_target)
            print(f"  Exported ONNX Model  : {onnx_target}")
        except Exception as ex:
            print(f"  ONNX export notice: {ex}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
