import os
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
try:
    Image.register_extension("JPEG", ".gfif")
    Image.register_extension("JPEG", ".jfif")
except Exception:
    pass
import streamlit as st
from ultralytics import YOLO
from concentric_nms import suppress_concentric_boxes

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Insulation Pipe Detection Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #00B4D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .metric-box {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Model Caching
# ---------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_test")
DATA_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data.yaml")

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    """Load and cache the YOLO model."""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def get_available_models():
    """Scan models directory for available .pt weights."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    return sorted(models)

def annotate_image(
    image_np,
    results,
    conf_threshold,
    show_boxes=True,
    show_labels=False,
    show_conf=False,
    show_numbers=False,
    show_count_badge=True,
    line_width=2,
    suppress_concentric=True
):
    """Draw bounding boxes and metadata on an image using OpenCV."""
    annotated = image_np.copy()
    boxes_info = []
    h, w = annotated.shape[:2]

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            boxes_info.append({
                "Class": cls_name,
                "Confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "Width (px)": box_width,
                "Height (px)": box_height,
                "Area (px²)": area,
                "crop": image_np[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None
            })

    if suppress_concentric:
        boxes_info = suppress_concentric_boxes(boxes_info, image_np=image_np)

    if show_boxes:
        for idx, b in enumerate(boxes_info):
            bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
            b_conf = b["Confidence"]
            b_cls = b["Class"]
            box_color = (0, 165, 255)  # Orange-Amber
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, line_width)

            label_parts = []
            if show_numbers:
                label_parts.append(f"#{idx + 1}")
            if show_labels:
                label_parts.append(b_cls)
            if show_conf:
                label_parts.append(f"{b_conf * 100:.1f}%")
            label_text = " | ".join(label_parts)

            if label_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.35, line_width * 0.18)
                thickness = max(1, int(line_width * 0.6))
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                banner_y1 = max(0, by1 - th - baseline - 4)
                banner_y2 = by1
                cv2.rectangle(annotated, (bx1, banner_y1), (bx1 + tw + 6, banner_y2), box_color, -1)
                cv2.putText(annotated, label_text, (bx1 + 3, by1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if show_count_badge and len(boxes_info) > 0:
            badge_text = f"Pipes: {len(boxes_info)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            badge_scale = max(0.65, min(w, h) / 750.0)
            badge_thick = max(2, int(badge_scale * 2.2))
            (btw, bth), bbase = cv2.getTextSize(badge_text, font, badge_scale, badge_thick)

            pad = 8
            # Semi-dark badge with crisp amber border
            cv2.rectangle(annotated, (12, 12), (12 + btw + pad * 2, 12 + bth + pad * 2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (12, 12), (12 + btw + pad * 2, 12 + bth + pad * 2), (0, 165, 255), 2)
            cv2.putText(annotated, badge_text, (12 + pad, 12 + bth + pad - 2), font, badge_scale, (255, 255, 255), badge_thick, cv2.LINE_AA)

    return annotated, boxes_info

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", width="stretch")
    st.title("⚙️ Model Settings")

    available_models = get_available_models()
    if available_models:
        default_idx = 0
        if "best_insulation_pipe_clean.pt" in available_models:
            default_idx = available_models.index("best_insulation_pipe_clean.pt")
        elif "best_insulation_pipe_finetuned.pt" in available_models:
            default_idx = available_models.index("best_insulation_pipe_finetuned.pt")

        selected_model_name = st.selectbox(
            "Select YOLO Weights (.pt)",
            available_models,
            index=default_idx,
            help="Models loaded from the /models directory. 'best_insulation_pipe_clean.pt' provides highest precision without inner-hole duplicates."
        )
        selected_model_path = os.path.join(MODELS_DIR, selected_model_name)
    else:
        st.error("No model files found in `models/` folder.")
        selected_model_path = None

    st.markdown("---")
    st.subheader("🎯 Inference Presets & Parameters")

    preset_choice = st.radio(
        "Detection Mode Preset",
        [
            "🎯 Standard Bundles & Multi-Layer Stacks (Conf: 0.45)",
            "📦 Dense Touching Pipes Grid (70 Pipes) (Conf: 0.35)",
            "📦 Deformed / Loose Oval Pipes in Box (Conf: 0.35)",
            "📦 Warehouse Carton / Box Packaging (Conf: 0.58)",
            "⚙️ Custom Adjust"
        ],
        index=0,
        help="Preset automatically configures optimal thresholds for pipe bundles vs dense shipping boxes."
    )

    if "Touching" in preset_choice or "Grid" in preset_choice or "Deformed" in preset_choice or "Oval" in preset_choice:
        default_conf = 0.35
    elif "Warehouse" in preset_choice or "Carton" in preset_choice:
        default_conf = 0.58
    elif "Standard" in preset_choice:
        default_conf = 0.45
    else:
        default_conf = 0.45

    conf_slider = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=default_conf,
        step=0.05,
        help="Recommended 0.48 for warehouse carton boxes, or 0.45 for standard multi-layer stacks."
    )

    iou_slider = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression."
    )

    img_size = st.select_slider(
        "Inference Size (imgsz)",
        options=[320, 480, 640, 960, 1280],
        value=640,
        help="Higher resolution (e.g. 960/1280) detects smaller, distant pipes better."
    )

    st.markdown("---")
    st.subheader("🎨 Display Options")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        show_boxes = st.checkbox("Boxes", value=True, help="Draw bounding boxes around detected pipes")
        show_count_badge = st.checkbox("Count on Image", value=True, help="Display pipe count badge directly on the image")
    with col_d2:
        show_numbers = st.checkbox("Pipe Numbers (#1, #2)", value=False, help="Number each detected pipe sequentially")
        show_labels = st.checkbox("Class Name ('pipe')", value=False, help="Show class name text banner on each box")

    col_d3, col_d4 = st.columns(2)
    with col_d3:
        show_conf = st.checkbox("Confidence %", value=False, help="Show confidence percentage on each box")
    with col_d4:
        show_crops = st.checkbox("Show Crops Gallery", value=False, help="Display cropped gallery of detected pipes")

    line_thickness = st.slider("Box Line Thickness", 1, 6, 2)

# Load selected model
model = None
model_load_err = None
if selected_model_path and os.path.exists(selected_model_path):
    with st.spinner("Loading YOLO model..."):
        model, model_load_err = load_yolo_model(selected_model_path)

if model_load_err:
    st.sidebar.error(f"Error loading model: {model_load_err}")
elif model is not None:
    st.sidebar.success(f"Active Model: `{os.path.basename(selected_model_path)}`")
    with st.sidebar.expander("Model Details", expanded=False):
        st.write("**Classes:**", model.names)
        st.write("**Task:**", model.task)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">⚡ Insulation Pipe Detection Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Deep Learning Inspection & Training for Thermal Insulation Pipes</div>', unsafe_allow_html=True)

tab_image, tab_video, tab_train, tab_guide = st.tabs([
    "📸 Photo Detection",
    "🎬 Video Detection",
    "🏋️ Live Model Training & Dashboard",
    "📈 Accuracy & Architecture Guide"
])

# =========================================================
# TAB 1: Photo Detection
# =========================================================
with tab_image:
    st.markdown("### 📤 Upload or Select an Image to Inspect")

    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_image = st.file_uploader(
            "Choose a pipe image (All formats: JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF, etc.)",
            type=[
                "jpg", "jpeg", "jfif", "gfif", "png", "webp",
                "bmp", "dib", "tiff", "tif", "gif", "avif",
                "ico", "ppm", "pgm", "pbm", "pnm"
            ],
            key="img_uploader"
        )
    with col_up2:
        st.write("")
        sample_choice = st.selectbox(
            "🧪 Load Demo Sample",
            [
                "-- None --",
                "🏢 Multi-Layer Pipe Stack (Anti-Watermark/Void)",
                "📦 Warehouse Box (Silver Foil Pipes)",
                "📦 Loose Deformed / Oval Pipes in Carton",
                "📦 Touching Pipes Grid (70 Pipes)",
                "🔴 Red Insulation Pipe Bundle",
                "📦 Standing Pipes + Cardboard Box",
                "📦 170-Pipe Box Packaging",
                "🔺 6-Pipe Pyramid Stack",
                "📏 Single Pipe"
            ],
            index=0
        )

    img_to_process = None
    source_name = "uploaded_image.jpg"

    if uploaded_image is not None:
        try:
            pil_image = Image.open(uploaded_image).convert("RGB")
            img_to_process = np.array(pil_image)
            source_name = uploaded_image.name
        except Exception as e:
            st.error(f"⚠️ Could not decode image '{uploaded_image.name}': {e}. Please ensure it is a valid image file.")
            img_to_process = None
    elif sample_choice != "-- None --":
        sample_path = None
        if "Multi-Layer" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "pipe_stack_dreamstime.jpg")
            source_name = "multi_layer_pipe_stack_sample.jpg"
        elif "Warehouse Box" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "warehouse_box_uploaded.jpg")
            source_name = "warehouse_box_sample.jpg"
        elif "Loose Deformed" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "box_deformed_pipes_clean.jpg")
            source_name = "loose_deformed_pipes_sample.jpg"
        elif "Touching" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "box_touching_grid_70.jpg")
            source_name = "touching_pipes_grid_sample.jpg"
        elif "Red" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "bundle_red_pipes.jpg")
            source_name = "red_pipe_bundle_sample.jpg"
        elif "Standing" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "clean_uploaded_box_pipes.jpg")
            source_name = "standing_pipes_cardboard_box_sample.jpg"
        elif "170-Pipe" in sample_choice:
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images", "train", "pipe_box_full.jpg")
            source_name = "cardboard_box_packaging_sample.jpg"
        elif "6-Pipe" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "crop_6_pipes.jpg")
            source_name = "6_pipes_pyramid_sample.jpg"
        elif "Single" in sample_choice:
            sample_path = os.path.join(SAMPLE_DIR, "pipe_horizontal_sample.jpg")
            source_name = "single_horizontal_pipe_sample.jpg"

        if sample_path and os.path.exists(sample_path):
            img_bgr = cv2.imread(sample_path)
            img_to_process = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            st.warning(f"Sample image not found on disk at {sample_path}.")

    if img_to_process is not None:
        if model is None:
            st.error("Model is not loaded. Please verify the model file in the sidebar.")
        else:
            t_start = time.perf_counter()
            results = model.predict(
                source=img_to_process,
                conf=conf_slider,
                iou=iou_slider,
                imgsz=img_size,
                verbose=False
            )
            t_infer = (time.perf_counter() - t_start) * 1000

            annotated_img, detections = annotate_image(
                img_to_process,
                results,
                conf_threshold=conf_slider,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_conf=show_conf,
                show_numbers=show_numbers,
                show_count_badge=show_count_badge,
                line_width=line_thickness,
                suppress_concentric=True
            )

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Pipes Detected", len(detections))
            with m2:
                max_conf = max([d["Confidence"] for d in detections]) if detections else 0.0
                st.metric("Highest Confidence", f"{max_conf * 100:.1f}%")
            with m3:
                avg_conf = (sum([d["Confidence"] for d in detections]) / len(detections)) if detections else 0.0
                st.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")
            with m4:
                st.metric("Inference Time", f"{t_infer:.1f} ms")

            st.markdown("---")

            # Side-by-Side Comparison
            col_orig, col_pred = st.columns(2)
            with col_orig:
                st.markdown("#### 📷 Original Image")
                st.image(img_to_process, width="stretch")
            with col_pred:
                st.markdown(f"#### 🎯 Detection Result ({len(detections)} Pipes)")
                st.image(annotated_img, width="stretch")

            # Download & Export
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".jpg", annotated_bgr)
                st.download_button(
                    label="💾 Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name=f"detected_{source_name}",
                    mime="image/jpeg",
                    width="stretch"
                )
            with res_col2:
                if detections:
                    df_det = pd.DataFrame([{
                        "Pipe #": idx + 1,
                        "Class": d["Class"],
                        "Confidence (%)": f"{d['Confidence']*100:.2f}",
                        "X1": d["x1"], "Y1": d["y1"], "X2": d["x2"], "Y2": d["y2"],
                        "Width": d["Width (px)"], "Height": d["Height (px)"],
                        "Area (px²)": d["Area (px²)"]
                    } for idx, d in enumerate(detections)])
                    csv_data = df_det.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Export Detections (CSV)",
                        data=csv_data,
                        file_name=f"detections_{source_name}.csv",
                        mime="text/csv",
                        width="stretch"
                    )

            if detections:
                st.markdown("### 📋 Detected Pipe Details")
                st.dataframe(df_det, width="stretch")

                if show_crops:
                    st.markdown("### 🔍 Cropped Pipe Inspect Gallery")
                    crop_cols = st.columns(min(len(detections), 4))
                    for idx, d in enumerate(detections):
                        if d["crop"] is not None and d["crop"].size > 0:
                            col_idx = idx % min(len(detections), 4)
                            with crop_cols[col_idx]:
                                st.image(
                                    d["crop"],
                                    caption=f"Pipe #{idx+1} ({d['Class']}: {d['Confidence']*100:.1f}%)",
                                    width="stretch"
                                )
            else:
                st.info("ℹ️ No insulation pipes detected above the current confidence threshold. Try lowering the threshold slider in the sidebar.")

# =========================================================
# TAB 2: Video Detection
# =========================================================
with tab_video:
    st.markdown("### 🎥 Upload or Test a Pipe Inspection Video")

    vcol1, vcol2 = st.columns([3, 1])
    with vcol1:
        uploaded_video = st.file_uploader(
            "Upload a video file (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader"
        )
    with vcol2:
        st.write("")
        st.write("")
        use_demo_vid = st.button("🧪 Load Demo Video", width="stretch")

    video_source_path = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_source_path = tfile.name
    elif use_demo_vid:
        demo_path = os.path.join(SAMPLE_DIR, "pipe_demo_video.mp4")
        if os.path.exists(demo_path):
            video_source_path = demo_path
        else:
            st.warning("Demo video not found on disk.")

    if video_source_path is not None:
        st.write("---")
        st.markdown("#### ⚙️ Video Processing Settings")
        v_opt1, v_opt2 = st.columns(2)
        with v_opt1:
            frame_stride = st.selectbox(
                "Frame Sampling (Process every N frames)",
                options=[1, 2, 3, 5],
                index=0,
                help="1 = Every frame. 2 or 3 = Faster processing."
            )
        with v_opt2:
            max_frames = st.number_input(
                "Max Frames to Process (0 = Entire video)",
                min_value=0,
                max_value=10000,
                value=150,
                step=50,
                help="Set a limit for quick previews."
            )

        start_btn = st.button("🚀 Start Video Detection", type="primary", width="stretch")

        if start_btn:
            if model is None:
                st.error("Model is not loaded. Please select a valid weights file.")
            else:
                cap = cv2.VideoCapture(video_source_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frames_to_run = total_frames if max_frames == 0 else min(total_frames, max_frames)

                out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_temp.name, fourcc, fps / frame_stride, (width, height))

                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_placeholder = st.empty()

                pipe_counts_per_frame = []
                frame_idx = 0
                processed_count = 0
                t_vid_start = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_idx >= frames_to_run:
                        break

                    if frame_idx % frame_stride == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = model.predict(
                            source=frame_rgb,
                            conf=conf_slider,
                            iou=iou_slider,
                            imgsz=img_size,
                            verbose=False
                        )
                        annotated_rgb, detections = annotate_image(
                            frame_rgb,
                            res,
                            conf_threshold=conf_slider,
                            show_boxes=show_boxes,
                            show_labels=show_labels,
                            show_conf=show_conf,
                            show_numbers=show_numbers,
                            show_count_badge=show_count_badge,
                            line_width=line_thickness,
                            suppress_concentric=True
                        )
                        pipe_counts_per_frame.append(len(detections))
                        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                        out_writer.write(annotated_bgr)
                        processed_count += 1

                        if processed_count % 5 == 0 or processed_count == 1:
                            preview_placeholder.image(
                                annotated_rgb,
                                caption=f"Processing Frame {frame_idx + 1}/{frames_to_run} | Pipes Detected: {len(detections)}",
                                width="stretch"
                            )

                    frame_idx += 1
                    progress_bar.progress(min(1.0, frame_idx / max(1, frames_to_run)))
                    status_text.text(f"Processed {frame_idx}/{frames_to_run} frames ({processed_count} saved)...")

                cap.release()
                out_writer.release()
                t_total_vid = time.time() - t_vid_start

                status_text.success(f"✅ Video processing finished in {t_total_vid:.2f}s! ({processed_count} frames processed)")

                # Video Analytics
                st.markdown("### 📊 Video Detection Analytics")
                vm1, vm2, vm3 = st.columns(3)
                with vm1:
                    st.metric("Total Frames Processed", processed_count)
                with vm2:
                    st.metric("Max Pipes in Single Frame", max(pipe_counts_per_frame) if pipe_counts_per_frame else 0)
                with vm3:
                    avg_p = sum(pipe_counts_per_frame) / len(pipe_counts_per_frame) if pipe_counts_per_frame else 0
                    st.metric("Avg Pipes per Frame", f"{avg_p:.2f}")

                if pipe_counts_per_frame:
                    df_chart = pd.DataFrame({
                        "Frame": list(range(len(pipe_counts_per_frame))),
                        "Detected Pipes": pipe_counts_per_frame
                    }).set_index("Frame")
                    st.line_chart(df_chart)

                with open(out_temp.name, "rb") as vf:
                    vid_bytes = vf.read()
                st.download_button(
                    label="💾 Download Processed Video (MP4)",
                    data=vid_bytes,
                    file_name="detected_insulation_pipe_video.mp4",
                    mime="video/mp4",
                    type="primary",
                    width="stretch"
                )

# =========================================================
# TAB 3: Live Model Training & Dashboard
# =========================================================
with tab_train:
    st.markdown("### 🏋️ Live Epoch Training Progress & Metrics Dashboard")
    st.markdown("Train or fine-tune your model right here with live epoch-by-epoch progress, time remaining (ETA), and real-time loss tracking.")

    # Training Setup Form
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        train_base_model = st.selectbox(
            "Starting Checkpoint",
            available_models if available_models else ["best_insulation_pipe.pt"],
            index=0,
            key="train_base_select"
        )
        train_epochs = st.slider("Total Epochs", min_value=5, max_value=100, value=20, step=5)
    with col_t2:
        train_batch = st.selectbox("Batch Size", options=[2, 4, 8, 16], index=1)
        train_imgsz = st.selectbox("Image Resolution", options=[640, 960, 1280], index=0)
    with col_t3:
        train_lr = st.select_slider("Learning Rate", options=[0.00005, 0.0001, 0.0005, 0.001], value=0.0001)
        freeze_backbone = st.checkbox("Freeze Backbone (Preserve base features)", value=True)

    exp_name = st.text_input("New Model Version Name", value="insulation_pipe_v3")
    start_train_btn = st.button("🚀 Start Live Fine-Tuning", type="primary", width="stretch")

    if start_train_btn:
        model_src = os.path.join(MODELS_DIR, train_base_model)
        if not os.path.exists(model_src):
            st.error(f"Base model not found at {model_src}")
        elif not os.path.exists(DATA_YAML_PATH):
            st.error(f"Dataset config not found at {DATA_YAML_PATH}")
        else:
            # Prepare UI Containers for Real-Time Updates
            progress_container = st.empty()
            status_banner = st.empty()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            card_epoch = col_m1.empty()
            card_time = col_m2.empty()
            card_loss = col_m3.empty()
            card_map = col_m4.empty()

            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            log_placeholder = st.empty()

            # State trackers
            epoch_history = []
            t_train_start = time.time()
            epoch_times = []

            # Custom Real-Time Callback
            def on_fit_epoch_end_callback(trainer):
                ep = trainer.epoch + 1
                tot = trainer.epochs
                if ep > tot:
                    return
                pct = ep / float(tot)

                # Time stats
                now = time.time()
                elapsed = now - t_train_start
                epoch_times.append(now)
                avg_ep_time = elapsed / max(1, ep)
                remaining_epochs = tot - ep
                eta_seconds = max(0, remaining_epochs * avg_ep_time)

                # Losses
                loss_dict = getattr(trainer, "loss_items", getattr(trainer, "tloss", {}))
                def safe_get_loss(k, idx=0):
                    if isinstance(loss_dict, dict):
                        v = loss_dict.get(k, 0.0)
                        return float(v.item()) if hasattr(v, "item") else float(v)
                    elif isinstance(loss_dict, (list, tuple)) and len(loss_dict) > idx:
                        v = loss_dict[idx]
                        return float(v.item()) if hasattr(v, "item") else float(v)
                    return 0.0

                loss_box = safe_get_loss("box_loss", 0)
                loss_cls = safe_get_loss("cls_loss", 1)
                loss_dfl = safe_get_loss("dfl_loss", 2)
                total_loss = loss_box + loss_cls + loss_dfl

                metrics = getattr(trainer, 'metrics', {})
                map50 = float(metrics.get('metrics/mAP50(B)', 0.0))
                map50_95 = float(metrics.get('metrics/mAP50-95(B)', 0.0))

                epoch_history.append({
                    "Epoch": ep,
                    "Total Loss": round(total_loss, 4),
                    "Box Loss": round(loss_box, 4),
                    "Cls Loss": round(loss_cls, 4),
                    "DFL Loss": round(loss_dfl, 4),
                    "mAP50 (%)": round(map50 * 100, 2),
                    "mAP50-95 (%)": round(map50_95 * 100, 2),
                    "Elapsed (s)": round(elapsed, 1),
                    "ETA (s)": round(eta_seconds, 1)
                })

                # Format times
                elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

                # Update UI in real-time!
                progress_container.progress(min(1.0, pct))
                status_banner.info(f"⚡ **Epoch {ep} / {tot}** ({pct*100:.1f}% Completed) — **{remaining_epochs} Epochs Left** (ETA: {eta_str})")

                card_epoch.metric("Epoch Progress", f"{ep} / {tot}", f"{pct*100:.0f}% Done")
                card_time.metric("Time & ETA", f"{elapsed_str}", f"-{eta_str} remaining")
                card_loss.metric("Box Loss", f"{loss_box:.3f}", f"Total: {total_loss:.2f}")
                card_map.metric("mAP50", f"{map50*100:.1f}%", f"mAP: {map50_95*100:.1f}%")

                # Live loss chart
                df_hist = pd.DataFrame(epoch_history)
                chart_placeholder.line_chart(df_hist.set_index("Epoch")[["Box Loss", "Cls Loss", "Total Loss"]])

                # Live table (last 5 epochs)
                table_placeholder.dataframe(df_hist.tail(5), width="stretch")

            # Initialize Model & Attach Callback
            status_banner.info("🔄 Initializing YOLO model and dataset loaders...")
            train_model = YOLO(model_src)
            train_model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

            # Execute Training
            save_proj = os.path.join(tempfile.gettempdir(), "yolo_live_train")
            train_results = train_model.train(
                data=DATA_YAML_PATH,
                epochs=train_epochs,
                batch=train_batch,
                imgsz=train_imgsz,
                device="cpu",
                freeze=10 if freeze_backbone else None,
                optimizer="AdamW",
                lr0=train_lr,
                lrf=0.01,
                cos_lr=True,
                degrees=15.0,
                fliplr=0.5,
                project=save_proj,
                name=exp_name,
                exist_ok=True,
                verbose=False
            )

            # Copy Best Weights to models/ directory
            best_weights = os.path.join(save_proj, exp_name, "weights", "best.pt")
            new_model_filename = f"{exp_name}_best.pt"
            dst_weights = os.path.join(MODELS_DIR, new_model_filename)

            if os.path.exists(best_weights):
                import shutil
                shutil.copy(best_weights, dst_weights)
                progress_container.progress(1.0)
                status_banner.success(f"🎉 **Training Finished Successfully!** Best model saved as `{new_model_filename}` in `models/`.")
                st.balloons()
                st.info(f"👉 You can now select `{new_model_filename}` in the sidebar to test it immediately!")

# =========================================================
# TAB 4: Accuracy & Architecture Guide
# =========================================================
with tab_guide:
    st.markdown("""
    ## 🎯 How to Maximize YOLOv8 Detection Accuracy for Insulation Pipes

    To take your insulation pipe detection model to production-grade accuracy (>90-95% mAP), follow these 6 proven deep-learning strategies implemented in the provided **`train_insulation_pipe.ipynb`** notebook:

    ---

    ### 1. 📐 Higher Input Resolution (`imgsz=960` or `1280`)
    - **Problem:** Insulation pipes in facilities are often long, thin, or captured at a distance. At default `640x640`, thin pipe boundaries get blurred into downsampled feature maps.
    - **Solution:** Fine-tune at **960x960** or **1280x1280**. This dramatically boosts small object recall and localization precision.

    ### 2. 🏗️ Model Architecture Upgrade (YOLOv8 Nano ➔ Small / Medium)
    - **Problem:** `YOLOv8n` has only **3.1M parameters**. It is designed for microcontrollers/edge devices and quickly reaches capacity limits on complex backgrounds.
    - **Solution:** Upgrade backbone to `YOLOv8s` (11.2M parameters) or `YOLOv8m` (25.9M parameters). This immediately delivers a **+5% to +12% mAP increase** while remaining real-time (>60 FPS).

    ### 3. 🔄 Domain-Specific Augmentation for Industrial Pipes
    - **Cylindrical & Rotational Invariance:** Pipes can run in any direction (horizontal, vertical, diagonal). Set `degrees=20.0`, `fliplr=0.5`, `flipud=0.3`.
    - **Scale Jittering:** Pipes appear both close-up and far away. Set `scale=0.6`.
    - **Occlusion Handling:** Real pipes are partially obscured by valves, brackets, or other pipes. Use `erasing=0.3` (Cutout) and `mixup=0.15`.
    - **Material & Lighting Variations:** Jacketing materials (aluminum foil, galvanized sheet, rubber foam, PVC) reflect light differently. Use `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`.

    ### 4. 📉 Cosine Annealing Learning Rate & Warmup
    - Enable `cos_lr=True`. Unlike standard step decay, Cosine Annealing smoothly descends towards `lrf=0.01`, allowing the weights to settle into sharper global minima without oscillating out.

    ### 5. 🚫 Hard Negative Mining (Background Images)
    - Include 5% to 10% **background images** with **empty label files** (walls, bare metal uninsulated pipes, electrical conduits, empty boiler rooms).
    - This drastically suppresses false positives!

    ### 6. 🧪 Test-Time Augmentation (TTA)
    - At inference, passing `augment=True` evaluates multi-scale and flipped versions of the input image, boosting validation mAP by +1.5% to +3% automatically.

    ---

    ### 🚀 Training Notebook Ready: `train_insulation_pipe.ipynb`
    Open the Jupyter Notebook in your project folder (`c:\\AI\\train_insulation_pipe.ipynb`) or upload it to Google Colab / Kaggle with a free GPU to run the full fine-tuning pipeline!
    """)
