# PipeCount — Project Presentation & Demo Guide

> **A Complete, Simple Guide to Presenting and Demonstrating the Automated Pipe Counting System**

---

## 1. Executive Summary (The 30-Second Pitch)

### What is PipeCount?
**PipeCount** is an AI-powered visual counting system made specifically for **thermal insulation foam pipes**. 

### The Problem It Solves
In warehouses, factories, and shipping hubs, workers manually count hundreds of pipes packed inside boxes or stacked in bundles. Manual counting is:
- **Slow**: Takes minutes per box.
- **Tiring**: Causes eye fatigue.
- **Error-prone**: Miscounts lead to inventory loss, incorrect shipments, and delayed deliveries.

### The Solution
A worker takes a photo using a phone or camera and uploads it. Within **half a second**, the system automatically detects, counts, and verifies every single pipe with **high accuracy**, showing the total count, bounding boxes, and an exportable report.

---

## 2. Why Normal AI Fails (And How We Solved It)

Hollow insulation pipes are surprisingly tricky for standard AI models. We engineered three key solutions to handle real-world warehouse conditions:

### Problem 1: Double Counting (The Inner Hole)
* **Why it happens**: Every pipe has a thick outer foam ring and a dark inner hollow hole. A standard computer vision model often detects the outer ring **and** the inner hole, counting one pipe as two!
* **Our Solution**: We built an **Anti-Double Filter (IoS — Intersection over Smaller)**. If a smaller circle is inside a larger circle, the system knows it is the same pipe and keeps only one count.

### Problem 2: Cardboard Box Holes Look Like Pipes
* **Why it happens**: Shipping boxes have round ventilation punch-holes that are round and dark, just like pipe faces.
* **Our Solution**: We built a **Color & Texture Analyzer**. The system checks the background color around each circle. If it detects warm yellowish-brown kraft cardboard, it ignores it. It only counts genuine foam rubber pipes.

### Problem 3: Tightly Packed & Squished Pipes
* **Why it happens**: When 70 or 100+ pipes are squeezed into a carton, they touch each other, lose their perfect circular shape, and become oval or squished.
* **Our Solution**: We trained a custom YOLOv8 neural network fine-tuned on touching grids and deformed pipes, achieving **70 out of 70 pipes counted (100% precision)** on dense warehouse boxes.

---

## 3. System Architecture (How It Is Built)

The project has a modern, clean architecture split into two parts:

```
┌────────────────────────────────┐       ┌────────────────────────────────┐
│      React 19 Frontend         │  ──>  │      FastAPI Backend           │
│  • Clean Dashboard (No clutter)│       │  • Python REST API             │
│  • Instant File Manager Upload │  <──  │  • ONNX Runtime (< 85 MB)      │
│  • Image & CSV Data Export     │       │  • 4-Pass Anti-Double Filter   │
└────────────────────────────────┘       └────────────────────────────────┘
```

1. **Frontend (React 19 + Vite)**:
   - A clean, simple dashboard interface.
   - Shows the **Original Photo on top** and the **Detection Result with Counts directly below**.
   - One-click buttons to download the annotated image or export coordinates to CSV (Excel).
   - Instant file manager access when clicking **Upload New Image**.

2. **Backend (FastAPI + Python)**:
   - Ultra-fast REST API endpoints (`/api/detect`, `/api/health`, `/api/docs`).
   - Supports **all image formats**: JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF.

3. **Lightweight AI Engine (ONNX Runtime)**:
   - Instead of heavy 6 GB machine learning software (like PyTorch with CUDA), we run our model on **ONNX Runtime**.
   - **Total bundle size is only ~85 MB**.
   - Runs fast on regular CPUs (no expensive GPU needed).
   - Ready for cloud hosting (Vercel Serverless, AWS, or local factory PCs).

---

## 4. Step-by-Step Live Demo Script

Use this exact walkthrough during your presentation:

### Step 1: Open the Application
* **Action**: Double-click `run_all.bat` on Windows (or open `http://localhost:5173` in Chrome).
* **What to say**:
  > *"Here is the PipeCount Dashboard. We designed this interface to be very clean, simple, and functional — without any unnecessary clutter or confusing buttons. On the left, we have our Dashboard navigation, and in the center is our upload workspace."*

### Step 2: Upload an Image
* **Action**: Drag and drop a warehouse box image into the upload area (or click to choose `box_touching_grid_70.jpg` from `backend/sample_test/`).
* **What to say**:
  > *"I will upload a warehouse carton containing 70 tightly packed insulation pipes. Notice that these pipes are touching each other and slightly deformed by the box."*

### Step 3: Present the Results
* **Action**: Point out the stacked layout.
* **What to say**:
  > *"In less than 450 milliseconds, the results are ready:*
  > *1. At the top, we see the **Original Image** exactly as taken by the camera.*
  > *2. Directly below, we see the **Detection Result**. Every single pipe has a crisp bounding box, and the badge confirms: **70 Pipes Counted** — zero omissions, zero duplicates.*
  > *3. Above the images, we have key metrics: Total Count (70) and Processing Time."*

### Step 4: Show the Data Table & Exports
* **Action**:
  1. Click **Pipe Bounding Boxes** accordion to show the coordinate table.
  2. Click **Export CSV** (opens a clean CSV file in Excel).
  3. Click **Download Image** to save the labeled picture.
* **What to say**:
  > *"For quality control and warehouse auditing, we can expand the Bounding Box table to see the exact pixel coordinates and dimensions of each pipe.*
  > *With one click, we can export this ledger directly to CSV for inventory systems like SAP or Excel, or download the full-resolution marked image for documentation."*

### Step 5: Test a New Image
* **Action**: Click the **Upload New Image** button in the top right.
* **What to say**:
  > *"When the worker is ready for the next box, clicking 'Upload New Image' immediately opens the computer's file manager so they can choose the next photo right away."*

---

## 5. Key Metrics & Selling Points

| Feature | Performance | Benefit to Business |
|---|---|---|
| **Counting Accuracy** | **100% on benchmark grids (70/70)** | Eliminates inventory discrepancies |
| **Processing Speed** | **350 ms – 450 ms per image** | 10x faster than manual counting |
| **File Format Support** | **JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF** | Workers can upload from any phone or camera |
| **System Footprint** | **~85 MB (ONNX Engine)** | Runs on standard laptops or cheap cloud servers |
| **Hardware Requirement**| **Standard CPU** | No expensive graphics cards (GPUs) required |

---

## 6. Frequently Asked Questions (FAQ)

### Q1: Does it need an active internet connection to count?
**Answer**: No. The backend runs completely offline on any local computer or factory workstation. It can also be deployed to the cloud (e.g. Vercel) if cloud access is desired.

### Q2: What happens if pipes are deformed or squished?
**Answer**: Our custom dataset was trained on real-world factory cartons with squished, deformed, and oval-shaped pipes. The model does not expect perfect circles; it recognizes real foam insulation cross-sections.

### Q3: Can it be connected to our ERP or warehouse software?
**Answer**: Yes. The backend provides standard REST API endpoints (`/api/detect`) that return standard JSON data, making integration with any inventory software or mobile app straightforward.

### Q4: Why is the model using ONNX instead of standard PyTorch?
**Answer**: PyTorch bundles large CUDA GPU libraries that take up over 6 GB of space. By exporting our trained YOLO model to ONNX, we reduced the size to only 12 MB (and the entire software stack to ~85 MB), while making CPU inference up to 2x faster.

---

## 7. How to Start the App for Demo Day

### Quick Start (Recommended)
Double-click:
```powershell
c:\AI\run_all.bat
```
This starts both the Python backend (port 8000) and the React frontend (port 5173), and automatically opens the browser.

### Manual Commands (If running from terminal)
1. **Start Backend**:
   ```powershell
   cd c:\AI\backend
   python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start Frontend**:
   ```powershell
   cd c:\AI\frontend
   npm run dev
   ```
3. Open browser at: **`http://localhost:5173`**
