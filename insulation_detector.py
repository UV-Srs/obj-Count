"""
Self-Contained Insulation Pipe Detection Engine (Anti-Double & Anti-Distractor)
Integrates YOLOv8 deep learning model with:
1. Anti-Double Concentric Suppression (no inner bore duplicate counts)
2. Cardboard packaging & watermark distractor rejection
3. Axial cylinder body suppression
4. Spatial isolation outlier filtering

Usage in any Python project:
----------------------------
from insulation_detector import InsulationPipeDetector

detector = InsulationPipeDetector(model_path="models/best_insulation_pipe_clean.pt")
annotated_img, detections = detector.detect("my_pipes.jpg")
print(f"Total pipes counted: {len(detections)}")
"""

import math
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Union

try:
    Image.register_extension("JPEG", ".gfif")
    Image.register_extension("JPEG", ".jfif")
except Exception:
    pass


class InsulationPipeDetector:
    def __init__(self, model_path: str = "models/best_insulation_pipe_clean.pt"):
        """Load YOLO model checkpoint."""
        self.model = YOLO(model_path)

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf: float = 0.45,
        iou: float = 0.45,
        imgsz: int = 640,
        box_color: Tuple[int, int, int] = (0, 200, 255),
        line_width: int = 2
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run pipe detection with integrated anti-double bore suppression and distractor filtering.

        Args:
            image_input: File path string or RGB numpy array.
            conf: Confidence threshold (default 0.45).
            iou: IoU threshold for model NMS (default 0.45).
            imgsz: Resolution for inference (default 640).
            box_color: BGR/RGB tuple for drawn boxes (default vibrant amber-gold).
            line_width: Thickness of bounding boxes.

        Returns:
            Tuple of:
                annotated_image: Image with clean boxes and total count badge.
                detections: List of filtered pipe dicts with coordinates and confidence.
        """
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

        # Model inference
        results = self.model.predict(
            source=rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

        raw_boxes = []
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
        filtered = self._suppress_duplicates_and_distractors(raw_boxes, rgb)

        # Draw clean output: boxes + top-left count badge (NO label clutter)
        annotated = rgb.copy()
        for b in filtered:
            cv2.rectangle(
                annotated,
                (b["x1"], b["y1"]),
                (b["x2"], b["y2"]),
                box_color,
                line_width
            )

        if len(filtered) > 0:
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
        image_rgb: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Integrated 4-pass filter engine."""
        if not boxes:
            return []

        h_img, w_img = image_rgb.shape[:2]

        # Pass 0: Cardboard & Distractor Rejection
        valid_boxes = []
        for b in boxes:
            bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
            if (bx2 - bx1) < 6 or (by2 - by1) < 6:
                valid_boxes.append(b)
                continue

            crop = image_rgb[by1:by2, bx1:bx2]
            top = crop[:max(2, crop.shape[0] // 5), :]
            bot = crop[-max(2, crop.shape[0] // 5):, :]
            left = crop[:, :max(2, crop.shape[1] // 5)]
            right = crop[:, -max(2, crop.shape[1] // 5):]
            outer = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
            mean_c = np.mean(outer, axis=0)
            r_val, g_val, b_val = mean_c[0], mean_c[1], mean_c[2]
            outer_gray = float(np.mean(mean_c))

            cx1, cy1 = max(0, (bx2 - bx1) // 3), max(0, (by2 - by1) // 3)
            cx2, cy2 = min((bx2 - bx1), (bx2 - bx1) - cx1), min((by2 - by1), (by2 - by1) - cy1)
            center_crop = crop[cy1:cy2, cx1:cx2]
            center_gray = float(np.mean(center_crop)) if center_crop.size > 0 else outer_gray

            # 1. Kraft cardboard brown check
            is_cardboard = (
                (r_val > 135 and (r_val - b_val) > 35 and (g_val - b_val) > 15) or
                (r_val / max(1.0, b_val) > 1.45 and g_val / max(1.0, b_val) > 1.18 and r_val > 70)
            )

            # 2. Dark shadow void check
            is_shadow_void = (outer_gray < 30 and center_gray < 25)

            # 3. Watermark check
            is_watermark = (
                (center_gray > outer_gray + 10 and outer_gray < 35 and ((bx2 - bx1) < 55 or (by2 - by1) < 55)) or
                (by2 < h_img * 0.22 and (bx2 - bx1) < 50 and (by2 - by1) < 50 and center_gray > outer_gray + 6)
            )

            # 4. Worker skin tone check (must have bright outer gray, no dark hollow bore)
            is_skin = False
            if crop.ndim == 3 and crop.shape[2] == 3 and outer_gray > 85 and center_gray > 38:
                crop_ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
                cr = crop_ycrcb[:, :, 1]
                cb = crop_ycrcb[:, :, 2]
                skin_mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
                is_skin = (np.mean(skin_mask) > 0.35 and ((by2 - by1) / max(1, bx2 - bx1) > 1.15))

            if not (is_cardboard or is_shadow_void or is_watermark or is_skin):
                valid_boxes.append(b)

        # Size outlier rejection
        if len(valid_boxes) >= 6:
            widths = [(b["x2"] - b["x1"]) for b in valid_boxes]
            med_w = float(np.median(widths))
            if med_w >= 18:
                valid_boxes = [b for b in valid_boxes if (b["x2"] - b["x1"]) >= med_w * 0.42]

        # Pass 1: Concentric Inner Bore Suppression (IoS)
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

                if (ios >= 0.50 and dist < max_dim1 * 0.45) or ios >= 0.75:
                    suppressed.add(j)

        remaining = [b for idx, b in enumerate(sorted_boxes) if idx not in suppressed]

        # Pass 2: Axial Cylinder Body Suppression
        if len(remaining) > 1:
            circular_faces = []
            for b in remaining:
                bw = max(1.0, b["x2"] - b["x1"])
                bh = max(1.0, b["y2"] - b["y1"])
                ar = bh / bw
                # Accommodate both circular faces and deformed/squished oval pipe cross-sections
                if 0.52 <= ar <= 1.82:
                    circular_faces.append(b)

            if len(circular_faces) >= 2:
                body_filtered = []
                for b in remaining:
                    bw = max(1.0, b["x2"] - b["x1"])
                    bh = max(1.0, b["y2"] - b["y1"])
                    ar = bh / bw
                    # True longitudinal cylinder bodies have high elongation (shafts)
                    if ar > 1.85 or ar < 0.50:
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

        # Pass 3: Spatial Isolation Outlier Filter
        if len(remaining) >= 8:
            widths = [(b["x2"] - b["x1"]) for b in remaining]
            med_w = float(np.median(widths))
            clustered = []
            for i, b in enumerate(remaining):
                cx = (b["x1"] + b["x2"]) / 2.0
                cy = (b["y1"] + b["y2"]) / 2.0
                min_dist = float('inf')
                for j, other in enumerate(remaining):
                    if i == j:
                        continue
                    ocx = (other["x1"] + other["x2"]) / 2.0
                    ocy = (other["y1"] + other["y2"]) / 2.0
                    d = math.hypot(cx - ocx, cy - ocy)
                    if d < min_dist:
                        min_dist = d
                if min_dist <= med_w * 2.8:
                    clustered.append(b)
            remaining = clustered

        return remaining

