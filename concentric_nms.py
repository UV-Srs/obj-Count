"""
Advanced Concentric & Axial Non-Maximum Suppression (Anti-Double Engine)
Specifically engineered for annular/hollow cylinders (insulation foam pipes).

Solves three fundamental object detection challenges:
1. Double detection of concentric circles: Outer foam circumference (big radius) vs inner bore (small radius).
2. Double detection of pipe faces vs elongated cylinder bodies: End cross-section face vs side barrel body.
3. Cardboard packaging distractors: Ventilation punch holes & flaps on brown cardboard boxes misclassified as pipes.
"""

import math
import cv2
import numpy as np
from typing import List, Union, Dict, Any


def suppress_concentric_boxes(
    boxes: List[Any],
    ios_threshold: float = 0.50,
    center_dist_ratio: float = 0.45,
    suppress_axial_body: bool = True,
    image_np: Any = None,
    is_bgr: bool = False
) -> List[Any]:
    """
    Suppresses inner concentric boxes (small radius / bore), elongated side-body duplicates,
    and cardboard packaging false alarms.

    Args:
        boxes: List of bounding boxes, either dicts or lists/tuples.
        ios_threshold: Intersection-over-Smaller threshold (default 0.50).
        center_dist_ratio: Max allowable distance between centers relative to larger box size.
        suppress_axial_body: If True, suppresses elongated cylinder bodies when circular end faces exist.
        image_np: Optional image array (numpy) to verify cellular foam texture vs brown kraft cardboard.
        is_bgr: Set to True if image_np is in BGR format (e.g. cv2.imread). Default False (RGB as in PIL/Streamlit).

    Returns:
        Deduplicated list of boxes with false alarms removed.
    """
    if not boxes:
        return []

    is_dict = isinstance(boxes[0], dict)

    def get_coords(b):
        if is_dict:
            c = float(b.get("conf", b.get("Confidence", 1.0)))
            return float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]), c
        return float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]) if len(b) > 4 else 1.0

    # Sort boxes by area descending (largest outer boundaries first)
    sorted_boxes = sorted(
        boxes,
        key=lambda b: (get_coords(b)[2] - get_coords(b)[0]) * (get_coords(b)[3] - get_coords(b)[1]),
        reverse=True
    )

    # Pass 0: Cardboard Packaging Distractor Suppression
    if image_np is not None:
        h_img, w_img = image_np.shape[:2]
        valid_boxes = []
        for b in sorted_boxes:
            x1, y1, x2, y2, _ = get_coords(b)
            bx1, by1 = max(0, int(x1)), max(0, int(y1))
            bx2, by2 = min(w_img, int(x2)), min(h_img, int(y2))
            if (bx2 - bx1) < 6 or (by2 - by1) < 6:
                valid_boxes.append(b)
                continue
            crop = image_np[by1:by2, bx1:bx2]
            top = crop[:max(2, crop.shape[0] // 5), :]
            bot = crop[-max(2, crop.shape[0] // 5):, :]
            left = crop[:, :max(2, crop.shape[1] // 5)]
            right = crop[:, -max(2, crop.shape[1] // 5):]
            outer = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
            mean_c = np.mean(outer, axis=0)
            if is_bgr:
                b_val, g_val, r_val = mean_c[0], mean_c[1], mean_c[2]
            else:
                r_val, g_val, b_val = mean_c[0], mean_c[1], mean_c[2]
            outer_gray = float(np.mean(mean_c))

            # Center crop (middle 35%)
            cx1, cy1 = max(0, (bx2 - bx1) // 3), max(0, (by2 - by1) // 3)
            cx2, cy2 = min((bx2 - bx1), (bx2 - bx1) - cx1), min((by2 - by1), (by2 - by1) - cy1)
            center_crop = crop[cy1:cy2, cx1:cx2]
            center_gray = float(np.mean(center_crop)) if center_crop.size > 0 else outer_gray

            # 1. Kraft cardboard brown test: warm yellow/brown texture with significant R-B contrast
            is_cardboard = (
                (r_val > 135 and (r_val - b_val) > 35 and (g_val - b_val) > 15) or
                (r_val / max(1.0, b_val) > 1.45 and g_val / max(1.0, b_val) > 1.18 and r_val > 70)
            )

            # 2. Dark shadow void test: pitch-black background cavities with no foam wall (outer < 30 and center < 25)
            is_shadow_void = (outer_gray < 30 and center_gray < 25)

            # 3. Inverted watermark / glare test: white watermark on dark background
            is_watermark = (
                (center_gray > outer_gray + 10 and outer_gray < 35 and ((bx2 - bx1) < 55 or (by2 - by1) < 55)) or
                (by2 < h_img * 0.22 and (bx2 - bx1) < 50 and (by2 - by1) < 50 and center_gray > outer_gray + 6)
            )

            # 4. Human skin tone test (person face/head in background: must have bright outer gray, no dark hollow bore)
            is_skin = False
            if crop.ndim == 3 and crop.shape[2] == 3 and outer_gray > 85 and center_gray > 38:
                crop_ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb if is_bgr else cv2.COLOR_RGB2YCrCb)
                cr = crop_ycrcb[:, :, 1]
                cb = crop_ycrcb[:, :, 2]
                skin_mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
                is_skin = (np.mean(skin_mask) > 0.35 and ((by2 - by1) / max(1, bx2 - bx1) > 1.15))

            if not (is_cardboard or is_shadow_void or is_watermark or is_skin):
                valid_boxes.append(b)
        sorted_boxes = valid_boxes

        # 5. Outlier Watermark Stamp Rejection:
        # If there are multiple detected pipes and a candidate box is less than 42% of the median pipe size, it's a watermark logo or speck
        if len(sorted_boxes) >= 6:
            widths = [(get_coords(b)[2] - get_coords(b)[0]) for b in sorted_boxes]
            med_w = float(np.median(widths))
            if med_w >= 18:
                sorted_boxes = [b for b in sorted_boxes if (get_coords(b)[2] - get_coords(b)[0]) >= med_w * 0.42]

    n = len(sorted_boxes)
    suppressed = set()

    # Pass 1: Concentric Inner Bore Suppression (Intersection-over-Smaller / IoS)
    for i in range(n):
        if i in suppressed:
            continue
        x1_l, y1_l, x2_l, y2_l, conf_l = get_coords(sorted_boxes[i])
        w_l = max(1.0, x2_l - x1_l)
        h_l = max(1.0, y2_l - y1_l)
        area_l = w_l * h_l
        c_lx = (x1_l + x2_l) / 2.0
        c_ly = (y1_l + y2_l) / 2.0
        max_dim = max(w_l, h_l)

        for j in range(i + 1, n):
            if j in suppressed:
                continue

            x1_s, y1_s, x2_s, y2_s, conf_s = get_coords(sorted_boxes[j])
            w_s = max(1.0, x2_s - x1_s)
            h_s = max(1.0, y2_s - y1_s)
            area_s = w_s * h_s
            c_sx = (x1_s + x2_s) / 2.0
            c_sy = (y1_s + y2_s) / 2.0

            # Calculate intersection
            ix1 = max(x1_l, x1_s)
            iy1 = max(y1_l, y1_s)
            ix2 = min(x2_l, x2_s)
            iy2 = min(y2_l, y2_s)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area = iw * ih

            # IoS = fraction of smaller box inside larger box
            ios = inter_area / area_s
            dist = math.hypot(c_lx - c_sx, c_ly - c_sy)

            # Suppress inner bore duplicates
            if (ios >= ios_threshold and dist < max_dim * center_dist_ratio) or ios >= 0.75:
                suppressed.add(j)

    remaining = [b for idx, b in enumerate(sorted_boxes) if idx not in suppressed]

    # Pass 2: Axial Cylinder Body Suppression
    if suppress_axial_body and len(remaining) > 1:
        circular_faces = []
        for b in remaining:
            x1, y1, x2, y2, _ = get_coords(b)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            ar = bh / bw
            # Accommodate both circular faces and deformed/squished oval pipe cross-sections
            if 0.52 <= ar <= 1.82:
                circular_faces.append(b)

        if len(circular_faces) >= 2:
            body_filtered = []
            for b in remaining:
                x1, y1, x2, y2, _ = get_coords(b)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                ar = bh / bw
                # True longitudinal cylinder bodies have high elongation (shafts)
                is_elongated = (ar > 1.85 or ar < 0.50)

                if is_elongated:
                    is_duplicate_body = False
                    for cf in circular_faces:
                        cx1, cy1, cx2, cy2, _ = get_coords(cf)
                        cfw = max(1.0, cx2 - cx1)
                        cfh = max(1.0, cy2 - cy1)

                        x_overlap = max(0.0, min(x2, cx2) - max(x1, cx1))
                        x_ratio = x_overlap / min(bw, cfw)

                        y_overlap = max(0.0, min(y2, cy2) - max(y1, cy1))
                        y_ratio = y_overlap / min(bh, cfh)

                        # MUST overlap substantially in BOTH dimensions to be a duplicate cylinder body of the same pipe.
                        # Touching or neighboring pipes in the same row/column share only one dimension and must NOT be suppressed.
                        if (x_ratio > 0.45 and y_ratio > 0.30) or (y_ratio > 0.45 and x_ratio > 0.30):
                            is_duplicate_body = True
                            break

                    if not is_duplicate_body:
                        body_filtered.append(b)
                else:
                    body_filtered.append(b)
            remaining = body_filtered

    # Pass 3: Spatial Isolation Outlier Filter
    # In bundled or carton-packed scenes, pipes form contiguous clusters.
    # Rejects isolated false alarms (stray background boxes, fixtures, people) > 2.8x median diameter from any neighbor.
    if len(remaining) >= 8:
        widths = [(get_coords(b)[2] - get_coords(b)[0]) for b in remaining]
        med_w = float(np.median(widths))
        clustered = []
        for i, b in enumerate(remaining):
            cx1, cy1, cx2, cy2, _ = get_coords(b)
            cx = (cx1 + cx2) / 2.0
            cy = (cy1 + cy2) / 2.0
            min_dist = float('inf')
            for j, other in enumerate(remaining):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2, _ = get_coords(other)
                ocx = (ox1 + ox2) / 2.0
                ocy = (oy1 + oy2) / 2.0
                d = math.hypot(cx - ocx, cy - ocy)
                if d < min_dist:
                    min_dist = d
            if min_dist <= med_w * 2.8:
                clustered.append(b)
        remaining = clustered

    return remaining
