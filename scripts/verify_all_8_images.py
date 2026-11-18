import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import cv2
from backend.insulation_detector import InsulationPipeDetector

det = InsulationPipeDetector(str(root_dir / 'backend/models/best_insulation_pipe_m_1024.onnx'))

all_cases = [
    ('3 New Images (Cardboard Cutouts / Vent Holes)', [
        ('New 1 (Carton vent hole)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1795000828497_14bccc21.jpg'),
        ('New 2 (BG carton holes & flap)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1795000889905_a77e601c.jpg'),
        ('New 3 (Handle slot)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1795000909830_cba111c2.jpg')
    ]),
    ('5 Earlier Real-World Benchmark Images (Zero Regression Check)', [
        ('Img 1 (Warehouse Box)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1794997103096_9445673e.jpg'),
        ('Img 2 (Trolley Cart)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1794997131829_9248df18.jpg'),
        ('Img 3 (Grid Touching)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1794997168211_7acb04ac.jpg'),
        ('Img 4 (Thick Carton)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1794997191158_ebcb0e5f.jpg'),
        ('Img 5 (Vertical Bundle)', r'C:/Users/SRS1/.gemini/antigravity/brain/151f212a-e3d2-4d9f-a0f0-7f57f44d9d68/.user_uploaded/media_1794997226234_72e3c0aa.jpg')
    ])
]

out_dir = root_dir / "backend/sample_test/verified_results"
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE VERIFICATION: 8 TOTAL REAL-WORLD SCENES")
print("=" * 70)

for group_name, cases in all_cases:
    print(f"\n--- {group_name} ---")
    for name, p in cases:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb, dets = det.detect(rgb)
        
        # Check specific cardboard hole regions:
        if 'New 1' in name:
            # Check right side hole at x > 340, y ~ 230-270
            right_hole = [d for d in dets if d['x1'] > 330 and 200 < d['y1'] < 300]
            print(f"[{name}] Total: {len(dets)} pipes | Right flap hole detected? {len(right_hole) > 0}")
        elif 'New 2' in name:
            # Check upper right bg holes at x > 220, y < 160
            bg_holes = [d for d in dets if d['x1'] > 220 and d['y2'] < 160]
            print(f"[{name}] Total: {len(dets)} pipes | Background box holes detected? {len(bg_holes) > 0}")
        elif 'New 3' in name:
            # Check bottom handle slot at y > 280
            slot = [d for d in dets if d['y1'] > 280]
            print(f"[{name}] Total: {len(dets)} pipes | Handle slot detected? {len(slot) > 0}")
        else:
            print(f"[{name}] Count: {len(dets)} pipes")
            
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        save_path = out_dir / f"{name[:5].replace(' ', '_').lower()}_verified.jpg"
        cv2.imwrite(str(save_path), annotated_bgr)

print("\n" + "=" * 70)
print("ALL 8 REAL-WORLD TEST CASES EVALUATED SUCCESSFULLY.")
print("=" * 70)
