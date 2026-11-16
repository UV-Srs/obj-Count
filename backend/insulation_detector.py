"""
Self-Contained Insulation Pipe Detection Engine (Anti-Double & Anti-Distractor)
Integrates YOLOv8 deep learning model (ONNX Runtime lightweight or PyTorch) with:
1. Anti-Double Concentric Suppression (no inner bore duplicate counts)
2. Cardboard packaging & watermark distractor rejection
3. Axial cylinder body suppression
4. Spatial isolation outlier filtering

Usage in any Python project:
----------------------------
from insulation_detector import InsulationPipeDetector

detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_clean.onnx")
annotated_img, detections = detector.detect("my_pipes.jpg")
print(f"Total pipes counted: {len(detections)}")
"""

import os
import math
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Union, Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    Image.register_extension("JPEG", ".gfif")
    Image.register_extension("JPEG", ".jfif")
except Exception:
    pass


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Standard Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class InsulationPipeDetector:
    def __init__(self, model_path: str = "models/best_insulation_pipe_clean.onnx"):
        """Load YOLO model checkpoint (prefers ONNX for ultra-lightweight execution, falls back to PyTorch)."""
        self.model_path = model_path
        self.use_onnx = False
        self.session = None
        self.torch_model = None
        self.native_imgsz = 640

        # Check if ONNX model is available or preferred
        onnx_candidate = None
        if model_path.endswith(".onnx") and os.path.exists(model_path):
            onnx_candidate = model_path
        elif model_path.endswith(".pt"):
            alt_onnx = model_path[:-3] + ".onnx"
            if os.path.exists(alt_onnx):
                onnx_candidate = alt_onnx

        if onnx_candidate and ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_candidate, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            try:
                shape = self.session.get_inputs()[0].shape
                if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
                    self.native_imgsz = int(shape[2])
            except Exception:
                pass
            self.use_onnx = True
        elif YOLO is not None and os.path.exists(model_path):
            self.torch_model = YOLO(model_path)
            if "1024" in os.path.basename(model_path) or "yolov8m" in os.path.basename(model_path):
                self.native_imgsz = 1024
            self.use_onnx = False
        elif os.path.exists(model_path) and model_path.endswith(".onnx") and ort is not None:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            try:
                shape = self.session.get_inputs()[0].shape
                if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
                    self.native_imgsz = int(shape[2])
            except Exception:
                pass
            self.use_onnx = True
        else:
            raise RuntimeError(
                f"Cannot load detector: model_path='{model_path}', "
                f"onnx_candidate='{onnx_candidate}' (ort_loaded={ort is not None}), "
                f"ultralytics_loaded={YOLO is not None}"
            )

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf: float = 0.18,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        max_det: int = 1500,
        box_color: Tuple[int, int, int] = (0, 200, 255),
        line_width: int = 2,
        draw_badge: bool = False,
        reject_square_tubes: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run pipe detection with integrated anti-double bore suppression and distractor filtering.

        Args:
            image_input: File path string or RGB numpy array.
            conf: Confidence threshold (default 0.25).
            iou: IoU threshold for model NMS (default 0.45).
            imgsz: Resolution for inference (default None -> auto-detected from model).
            box_color: BGR/RGB tuple for drawn boxes (default vibrant amber-gold).
            line_width: Thickness of bounding boxes.

        Returns:
            Tuple of:
                annotated_image: Image with clean boxes and total count badge.
                detections: List of filtered pipe dicts with coordinates and confidence.
        """
        if imgsz is None:
            imgsz = getattr(self, "native_imgsz", 640)
        # Load image if file path, PIL Image, or numpy array
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    pil_img = Image.open(image_input).convert("RGB")
                    rgb = np.array(pil_img)
                except Exception as e:
                    raise FileNotFoundError(f"Could not load image from '{image_input}': {e}")
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
        else:
            rgb = image_input.copy()

        h, w = rgb.shape[:2]

        raw_boxes = []
        if self.use_onnx:
            padded, r, (dw, dh) = letterbox(rgb, (imgsz, imgsz))
            tensor = padded.astype(np.float32) / 255.0
            tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

            outputs = self.session.run(None, {self.input_name: tensor})[0]
            preds = outputs[0].T  # Shape: [8400, 5] (cx, cy, w, h, conf)

            boxes = preds[:, :4]
            confs = preds[:, 4]

            mask = confs >= conf
            boxes = boxes[mask]
            confs = confs[mask]

            if len(confs) > 0:
                cx = (boxes[:, 0] - dw) / r
                cy = (boxes[:, 1] - dh) / r
                bw = boxes[:, 2] / r
                bh = boxes[:, 3] / r

                x1 = np.clip(cx - bw / 2, 0, w)
                y1 = np.clip(cy - bh / 2, 0, h)
                x2 = np.clip(cx + bw / 2, 0, w)
                y2 = np.clip(cy + bh / 2, 0, h)
                xyxy = np.column_stack([x1, y1, x2, y2])

                keep_indices = nms(xyxy, confs, iou)
                for idx in keep_indices:
                    raw_boxes.append({
                        "x1": int(xyxy[idx, 0]),
                        "y1": int(xyxy[idx, 1]),
                        "x2": int(xyxy[idx, 2]),
                        "y2": int(xyxy[idx, 3]),
                        "conf": float(confs[idx]),
                        "w": int(xyxy[idx, 2] - xyxy[idx, 0]),
                        "h": int(xyxy[idx, 3] - xyxy[idx, 1])
                    })
        else:
            results = self.torch_model.predict(
                source=rgb,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                verbose=False
            )
            for r in results:
                for b in r.boxes:
                    c = float(b.conf[0])
                    if c < conf:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    raw_boxes.append({
                        "x1": max(0, x1),
                        "y1": max(0, y1),
                        "x2": min(w, x2),
                        "y2": min(h, y2),
                        "conf": c,
                        "w": x2 - x1,
                        "h": y2 - y1
                    })

        # Built-in Anti-Double & Distractor Suppression Pipeline
        filtered = self._suppress_duplicates_and_distractors(
            raw_boxes, rgb, reject_square_tubes=reject_square_tubes
        )

        # Draw clean output: pipe bounding boxes (kept clean and unobstructed)
        annotated = rgb.copy()
        for b in filtered:
            cv2.rectangle(
                annotated,
                (b["x1"], b["y1"]),
                (b["x2"], b["y2"]),
                box_color,
                line_width
            )

        # Optional in-image count badge (default False: keeps image unobstructed)
        if draw_badge and len(filtered) > 0:
            badge_text = f"Pipes: {len(filtered)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            badge_scale = max(0.65, min(w, h) / 750.0)
            thick = max(2, int(badge_scale * 2.2))
            (btw, bth), _ = cv2.getTextSize(badge_text, font, badge_scale, thick)
            pad = 8
            # Semi-dark badge with crisp border
            cv2.rectangle(annotated, (12, 12), (12 + btw + pad * 2, 12 + bth + pad * 2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (12, 12), (12 + btw + pad * 2, 12 + bth + pad * 2), box_color, 2)
            cv2.putText(annotated, badge_text, (12 + pad, 12 + bth + pad - 2), font, badge_scale, (255, 255, 255), thick, cv2.LINE_AA)

        return annotated, filtered

    def _suppress_duplicates_and_distractors(
        self,
        boxes: List[Dict[str, Any]],
        image_rgb: np.ndarray,
        reject_square_tubes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Pure geometric post-processing engine:
        1. Aspect Ratio & Dimension Sanity Filter (removes non-pipe slivers/stripes).
        2. Concentric Inner Bore Suppression (IoS) - suppresses inner hole duplicates.
        3. Axial Body Duplicate Suppression - suppresses long side cylinder bodies overlapping circular faces.
        """
        if not boxes:
            return []

        h_img, w_img = image_rgb.shape[:2]

        # Pass 0: Aspect ratio & dimension sanity filter
        valid_boxes = []
        for b in boxes:
            bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
            box_w = bx2 - bx1
            box_h = by2 - by1
            if box_w < 6 or box_h < 6:
                continue

            aspect = box_h / max(1.0, box_w)
            # Pipe cross sections (even squashed or oval) have aspect ratio between 0.35 and 2.85
            if aspect > 2.85 or aspect < 0.35:
                continue

            valid_boxes.append(b)

        if not valid_boxes:
            return []

        # Pass 1: Concentric Inner Bore Suppression (IoS)
        # Sort by area descending so outer tube is checked against inner bore
        sorted_boxes = sorted(valid_boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
        suppressed = set()
        n = len(sorted_boxes)

        for i in range(n):
            if i in suppressed:
                continue
            b1 = sorted_boxes[i]
            w1 = max(1.0, b1["x2"] - b1["x1"])
            h1 = max(1.0, b1["y2"] - b1["y1"])
            c1x = (b1["x1"] + b1["x2"]) / 2.0
            c1y = (b1["y1"] + b1["y2"]) / 2.0
            max_dim1 = max(w1, h1)

            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                b2 = sorted_boxes[j]
                w2 = max(1.0, b2["x2"] - b2["x1"])
                h2 = max(1.0, b2["y2"] - b2["y1"])
                area2 = w2 * h2
                c2x = (b2["x1"] + b2["x2"]) / 2.0
                c2y = (b2["y1"] + b2["y2"]) / 2.0

                ix1 = max(b1["x1"], b2["x1"])
                iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"])
                iy2 = min(b1["y2"], b2["y2"])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                ios = inter / area2
                dist = math.hypot(c1x - c2x, c1y - c2y)

                # If smaller box is substantially inside larger box and centers are close -> inner bore
                if (ios >= 0.45 and dist < max_dim1 * 0.48) or ios >= 0.70:
                    suppressed.add(j)

        remaining = [b for idx, b in enumerate(sorted_boxes) if idx not in suppressed]

        # Pass 2: Axial Cylinder Body Suppression (suppresses elongated body when circular face exists)
        if len(remaining) > 1:
            circular_faces = []
            for b in remaining:
                bw = max(1.0, b["x2"] - b["x1"])
                bh = max(1.0, b["y2"] - b["y1"])
                ar = bh / bw
                if 0.55 <= ar <= 1.80:
                    circular_faces.append(b)

            if len(circular_faces) >= 2:
                body_filtered = []
                for b in remaining:
                    bw = max(1.0, b["x2"] - b["x1"])
                    bh = max(1.0, b["y2"] - b["y1"])
                    ar = bh / bw
                    if ar > 2.0 or ar < 0.45:
                        is_duplicate_body = False
                        for cf in circular_faces:
                            cfw = max(1.0, cf["x2"] - cf["x1"])
                            cfh = max(1.0, cf["y2"] - cf["y1"])
                            x_ratio = max(0.0, min(b["x2"], cf["x2"]) - max(b["x1"], cf["x1"])) / min(bw, cfw)
                            y_ratio = max(0.0, min(b["y2"], cf["y2"]) - max(b["y1"], cf["y1"])) / min(bh, cfh)
                            if (x_ratio > 0.45 and y_ratio > 0.30) or (y_ratio > 0.45 and x_ratio > 0.30):
                                is_duplicate_body = True
                                break
                        if not is_duplicate_body:
                            body_filtered.append(b)
                    else:
                        body_filtered.append(b)
                remaining = body_filtered

        # Pass 3: Spatial Outlier & Background Distractor Rejection
        # Rejects isolated false detections:
        # 1. Background worker faces & heads
        # 2. Worker safety shoes on floor & distant floor trash
        # 3. Flap holes and cardboard voids on brown shipping boxes
        if len(remaining) >= 6:
            centers = np.array([[(b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0] for b in remaining])
            dims = np.array([max(b["w"], b["h"]) for b in remaining])
            areas = np.array([b["w"] * b["h"] for b in remaining])
            median_dim = float(np.median(dims))
            median_area = float(np.median(areas))

            n_rem = len(remaining)
            min_dists = []
            for i in range(n_rem):
                dists = [math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) for j in range(n_rem) if i != j]
                min_dists.append(min(dists) if dists else 0.0)
            median_nn_dist = float(np.median(min_dists)) if min_dists else 1.0

            cluster_filtered = []
            for i, b in enumerate(remaining):
                nn = min_dists[i]
                ratio_nn = nn / max(1.0, median_nn_dist)
                ratio_dim = nn / max(1.0, median_dim)
                area_ratio = (b["w"] * b["h"]) / max(1.0, median_area)

                # Distractor 1: Worker safety shoes on floor or far background objects
                is_floor_shoe_distractor = (ratio_nn > 3.6 and (ratio_dim > 1.6 or area_ratio > 1.6))

                # Distractor 2: Background Worker Head & Face Elimination
                is_worker_face_distractor = False
                if ratio_nn > 1.8 and b["h"] > 16 and b["w"] > 16:
                    crop = image_rgb[b["y1"]:b["y2"], b["x1"]:b["x2"]]
                    if crop.size > 0:
                        ch, cw = crop.shape[:2]
                        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                        center_g = gray[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4].mean()
                        outer_g = np.concatenate([
                            gray[:ch // 4, :].ravel(),
                            gray[3 * ch // 4:, :].ravel(),
                            gray[:, :cw // 4].ravel(),
                            gray[:, 3 * cw // 4:].ravel()
                        ]).mean()
                        # Real pipes have dark foam outer walls. Worker heads have hair/face darker than bright background warehouse wall
                        if outer_g > center_g + 12:
                            ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
                            skin = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 175) & (ycrcb[:, :, 2] >= 80) & (ycrcb[:, :, 2] <= 130)
                            if np.mean(skin) > 0.25 and crop.mean(axis=(0, 1))[0] > crop.mean(axis=(0, 1))[2] + 2.0:
                                is_worker_face_distractor = True

                # Distractor 3: Flap Punch-Holes on Kraft Cardboard
                is_cardboard_flap_distractor = False
                if ratio_nn > 2.5 and b["y1"] > 0 and b["x1"] > 0:
                    crop = image_rgb[b["y1"]:b["y2"], b["x1"]:b["x2"]]
                    if crop.size > 0:
                        c_rgb = crop.mean(axis=(0, 1))
                        # Brown cardboard hue (R > 85, warm red > blue)
                        if c_rgb[0] > 85 and (c_rgb[0] - c_rgb[2]) > 25 and (c_rgb[1] - c_rgb[2]) > 10:
                            is_cardboard_flap_distractor = True

                if not (is_floor_shoe_distractor or is_worker_face_distractor or is_cardboard_flap_distractor):
                    cluster_filtered.append(b)

            remaining = cluster_filtered

        # Pass 4: Square Hollow Section (SHS) & 4-Vertex Polygon Rejection
        # Genuine pipes have circular/oval cross-sections (approxPolyDP produces >= 6 vertices).
        # Square and rectangular structural steel tubes have 4 flat sides and 90-degree corners (approxPolyDP gives exactly 4 vertices).
        if reject_square_tubes and len(remaining) > 0:
            square_filtered = []
            for b in remaining:
                bw = b["x2"] - b["x1"]
                bh = b["y2"] - b["y1"]
                if bw >= 16 and bh >= 16 and b["y2"] <= h_img and b["x2"] <= w_img:
                    crop = image_rgb[b["y1"]:b["y2"], b["x1"]:b["x2"]]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                        edges = cv2.Canny(blurred, 35, 120)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        is_square_tube = False
                        if contours:
                            c_largest = max(contours, key=cv2.contourArea)
                            c_area = cv2.contourArea(c_largest)
                            c_peri = cv2.arcLength(c_largest, True)
                            if c_peri > 14 and c_area > 20:
                                circ = (4.0 * math.pi * c_area) / (c_peri * c_peri)
                                approx = cv2.approxPolyDP(c_largest, 0.035 * c_peri, True)
                                if len(approx) == 4 and circ < 0.76:
                                    is_square_tube = True
                        if not is_square_tube:
                            square_filtered.append(b)
                        continue
                square_filtered.append(b)
            remaining = square_filtered

        return remaining
