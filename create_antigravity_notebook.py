import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 YOLOv8 Medium (1024px) Deep Accuracy & Precision Fine-Tuning\n",
                "\n",
                "This notebook fine-tunes **`best_insulation_pipe_m_1024.pt`** directly inside the **Antigravity IDE**.\n",
                "\n",
                "---\n",
                "### 🎯 Key Enhancements:\n",
                "1. **Purified Master Dataset**: All 1,411 elongated side-view shaft distractors pruned.\n",
                "2. **Hard Negative Samples**: Worker face and warehouse shelving with 0 annotations to eliminate false positives.\n",
                "3. **Zero-Shock Fine-Tuning**: `warmup_epochs=0` and `lr0=0.0001` with `AdamW` to preserve pre-trained circular representations.\n",
                "4. **High-Resolution 1024px**: 2.56x higher spatial resolution for dense carton grids.\n",
                "5. **Auto-Deployment**: Automatically updates `backend/models/` and exports 1024px ONNX."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Environment & Hardware Verification\n",
                "import os, sys, time, shutil\n",
                "from pathlib import Path\n",
                "import torch\n",
                "import cv2\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from PIL import Image\n",
                "from ultralytics import YOLO\n",
                "\n",
                "print(f\"PyTorch Version : {torch.__version__}\")\n",
                "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
                "device = 0 if torch.cuda.is_available() else 'cpu'\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU Device      : {torch.cuda.get_device_name(0)}\")\n",
                "else:\n",
                "    print(\"Running on Local CPU (Backbone freezing enabled for fast throughput)\")\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Resolve Project Paths & Verify Dataset\n",
                "# Auto-detect root directory (c:\\AI)\n",
                "CURRENT_DIR = Path(os.getcwd())\n",
                "if (CURRENT_DIR / \"backend\").exists():\n",
                "    PROJECT_ROOT = CURRENT_DIR\n",
                "elif (CURRENT_DIR.parent / \"backend\").exists():\n",
                "    PROJECT_ROOT = CURRENT_DIR.parent\n",
                "else:\n",
                "    PROJECT_ROOT = Path(\"c:/AI\")\n",
                "\n",
                "BACKEND_DIR = PROJECT_ROOT / \"backend\"\n",
                "DATA_YAML = BACKEND_DIR / \"data\" / \"data.yaml\"\n",
                "MODEL_PATH = BACKEND_DIR / \"models\" / \"best_insulation_pipe_m_1024.pt\"\n",
                "\n",
                "print(f\"Project Root    : {PROJECT_ROOT}\")\n",
                "print(f\"Data Config     : {DATA_YAML} (Exists: {DATA_YAML.exists()})\")\n",
                "print(f\"Warm Model Path : {MODEL_PATH} (Exists: {MODEL_PATH.exists()})\")\n",
                "\n",
                "train_imgs = list((BACKEND_DIR / \"data\" / \"images\" / \"train\").glob(\"*.jpg\"))\n",
                "val_imgs = list((BACKEND_DIR / \"data\" / \"images\" / \"val\").glob(\"*.jpg\"))\n",
                "print(f\"Train Images    : {len(train_imgs)}\")\n",
                "print(f\"Val Images      : {len(val_imgs)}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: In-Depth Fine-Tuning Execution\n",
                "model = YOLO(str(MODEL_PATH))\n",
                "\n",
                "# Fine-tuning configuration:\n",
                "# If running on CPU, we freeze layers 0-11 so backprop only updates detection heads.\n",
                "# If running on GPU, freeze=0 to fine-tune all layers.\n",
                "freeze_layers = 0 if torch.cuda.is_available() else 12\n",
                "\n",
                "TRAIN_CONFIG = {\n",
                "    \"data\": str(DATA_YAML),\n",
                "    \"epochs\": 15,               # Adjust number of epochs as desired\n",
                "    \"patience\": 10,\n",
                "    \"imgsz\": 1024,              # Native 1024x1024 resolution\n",
                "    \"batch\": 4,                 # Optimal for CPU/GPU memory\n",
                "    \"device\": device,\n",
                "    \"workers\": 0,\n",
                "    \"freeze\": freeze_layers,    # Freezes backbone for fast CPU training\n",
                "    \"optimizer\": \"AdamW\",\n",
                "    \"lr0\": 0.0001,              # Low rate prevents weight shock\n",
                "    \"lrf\": 0.01,\n",
                "    \"cos_lr\": True,\n",
                "    \"warmup_epochs\": 0,         # Zero warmup: maintains pretrained circular features\n",
                "    \"box\": 7.5,                 # High box regression loss weight\n",
                "    \"cls\": 1.5,                 # Represses false positives\n",
                "    \"dfl\": 1.5,\n",
                "    \"degrees\": 10.0,            # Slight tilt invariance for real pipes\n",
                "    \"translate\": 0.1,\n",
                "    \"scale\": 0.5,\n",
                "    \"fliplr\": 0.5,\n",
                "    \"flipud\": 0.2,\n",
                "    \"mosaic\": 0.5,\n",
                "    \"close_mosaic\": 5,\n",
                "    \"save\": True,\n",
                "    \"save_period\": 5,\n",
                "    \"project\": \"runs/train\",\n",
                "    \"name\": \"yolov8m_antigravity_deep\",\n",
                "    \"exist_ok\": True\n",
                "}\n",
                "\n",
                "print(\"=\"*70)\n",
                "print(\"🚀 STARTING FINE-TUNING IN ANTIGRAVITY NOTEBOOK\")\n",
                "print(\"=\"*70)\n",
                "t0 = time.time()\n",
                "results = model.train(**TRAIN_CONFIG)\n",
                "elapsed = (time.time() - t0) / 60.0\n",
                "print(f\"\\n✅ Training completed in {elapsed:.1f} minutes!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: Validation & Convergence Curves\n",
                "save_dir = Path(results.save_dir)\n",
                "best_weights = save_dir / \"weights\" / \"best.pt\"\n",
                "if not best_weights.exists():\n",
                "    candidates = list(PROJECT_ROOT.glob(\"runs/**/yolov8m_antigravity_deep/weights/best.pt\"))\n",
                "    if candidates:\n",
                "        best_weights = candidates[-1]\n",
                "\n",
                "print(f\"🎯 Evaluating best model: {best_weights}\")\n",
                "best_model = YOLO(str(best_weights))\n",
                "metrics = best_model.val(data=str(DATA_YAML), imgsz=1024, device=device)\n",
                "\n",
                "p = metrics.box.p[0] * 100.0\n",
                "r = metrics.box.r[0] * 100.0\n",
                "map50 = metrics.box.map50 * 100.0\n",
                "map95 = metrics.box.map * 100.0\n",
                "f1 = 2 * (p * r) / (p + r + 1e-16)\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"🎯 FINAL VALIDATION METRICS:\")\n",
                "print(f\"   Precision (P) : {p:.2f}%\")\n",
                "print(f\"   Recall (R)    : {r:.2f}%\")\n",
                "print(f\"   mAP@50        : {map50:.2f}%\")\n",
                "print(f\"   mAP@50-95     : {map95:.2f}%\")\n",
                "print(f\"   F1-Score      : {f1:.2f}%\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "# Display training curves\n",
                "results_img = save_dir / \"results.png\"\n",
                "if results_img.exists():\n",
                "    plt.figure(figsize=(14, 8))\n",
                "    plt.imshow(Image.open(results_img))\n",
                "    plt.axis(\"off\")\n",
                "    plt.title(\"Training Loss Convergence & Metric Progression\", fontsize=14)\n",
                "    plt.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Multi-Scenario Benchmark Suite\n",
                "sample_dir = BACKEND_DIR / \"sample_test\"\n",
                "candidates = [\n",
                "    (\"70-Pipe Grid Carton\", sample_dir / \"box_touching_grid_70.jpg\", 0.35, 70),\n",
                "    (\"Multi-Layer Stack\", sample_dir / \"pipe_stack_dreamstime.jpg\", 0.28, 29),\n",
                "    (\"Warehouse Box Carton\", sample_dir / \"warehouse_box_uploaded.jpg\", 0.50, 58),\n",
                "    (\"6-Pipe Pyramid Bundle\", sample_dir / \"crop_6_pipes.jpg\", 0.28, 6),\n",
                "    (\"Loose Deformed Pipes\", sample_dir / \"box_deformed_pipes_clean.jpg\", 0.35, 48),\n",
                "]\n",
                "\n",
                "print(\"=\"*70)\n",
                "print(\"🧪 MULTI-SCENARIO BENCHMARK VERIFICATION\")\n",
                "print(\"=\"*70)\n",
                "for title, fpath, conf_val, expected in candidates:\n",
                "    if fpath.exists():\n",
                "        res = best_model.predict(source=str(fpath), conf=conf_val, imgsz=1024, verbose=False)[0]\n",
                "        det_count = len(res.boxes)\n",
                "        acc_pct = (1.0 - abs(det_count - expected) / float(expected)) * 100.0\n",
                "        print(f\"  {title:25s} | Detected: {det_count:3d} / {expected:3d} ({acc_pct:.1f}% accuracy)\")\n",
                "    else:\n",
                "        print(f\"  {title:25s} | File not found at {fpath}\")\n",
                "print(\"=\"*70)\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 6: Export to 1024px ONNX & Deploy to Backend\n",
                "print(\"⚡ Exporting refined model to 1024px ONNX...\")\n",
                "onnx_file = best_model.export(format=\"onnx\", imgsz=1024, simplify=True)\n",
                "\n",
                "prod_pt = BACKEND_DIR / \"models\" / \"best_insulation_pipe_m_1024.pt\"\n",
                "prod_onnx = BACKEND_DIR / \"models\" / \"best_insulation_pipe_m_1024.onnx\"\n",
                "\n",
                "shutil.copy(str(best_weights), str(prod_pt))\n",
                "shutil.copy(str(onnx_file), str(prod_onnx))\n",
                "\n",
                "print(\"\\n✅ Models Deployed Successfully to Production!\")\n",
                "print(f\"   PyTorch Checkpoint : {prod_pt} ({os.path.getsize(prod_pt)/1e6:.1f} MB)\")\n",
                "print(f\"   1024px ONNX Model  : {prod_onnx} ({os.path.getsize(prod_onnx)/1e6:.1f} MB)\")\n",
                "print(\"\\n🎉 Ready to test in UI: Launch run_all.bat to see your refined model live!\")\n"
            ]
        }
    ]
}

# Write to both root and backend
for dest in ["c:/AI/train_yolov8m_deep_accuracy.ipynb", "c:/AI/backend/train_yolov8m_deep_accuracy.ipynb"]:
    with open(dest, "w", encoding="utf-8") as fp:
        json.dump(notebook, fp, indent=2)
    print(f"✅ Generated Antigravity Notebook at: {dest}")

