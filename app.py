"""
app.py - Streamlit Web Application for Parking Space Occupancy Detection.
Upload a parking lot image and get real-time occupancy analysis via YOLOv8.
"""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.detection_utils import (
    draw_results,
    get_summary,
    load_model,
    run_detection,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/best.pt")


# ---------------------------------------------------------------------------
# Page config  (must be FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parking Detector",
    page_icon="🅿️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Load YOLO model once per session and cache it."""
    return load_model(str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Occupancy progress bar (custom CSS)
# ---------------------------------------------------------------------------
def render_occupancy_bar(rate: float) -> None:
    """Render a coloured CSS progress bar for occupancy rate."""
    rate = max(0.0, min(100.0, rate))

    if rate < 50:
        color = "#4CAF50"   # green
    elif rate < 80:
        color = "#FF9800"   # orange
    else:
        color = "#F44336"   # red

    st.markdown(
        f"""
        <div style="
            width:100%; background:#e0e0e0; border-radius:8px;
            height:30px; overflow:hidden; margin:8px 0 20px 0;
            border:1px solid #bbb;">
          <div style="
              width:{rate:.1f}%; background:{color}; height:100%;
              display:flex; align-items:center; justify-content:center;
              color:white; font-weight:bold; font-size:14px;
              font-family:sans-serif; border-radius:8px;
              transition:width .4s ease;">
            {rate:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> float:
    with st.sidebar:
        st.title("🅿️ Parking Detector")
        st.markdown(
            "Upload a parking lot image and the system will classify "
            "each detected space as **free**, **occupied**, or "
            "**partially occupied** using a fine-tuned YOLOv8 model."
        )
        st.markdown("---")

        conf = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Detections below this score are ignored.",
        )

        st.markdown("---")
        st.markdown("### Color Legend")
        st.markdown(
            """
            <div style="font-size:15px; font-family:sans-serif; line-height:2;">
              🟢 &nbsp;<b>Free</b><br>
              🔴 &nbsp;<b>Occupied</b><br>
              🟡 &nbsp;<b>Partially Occupied</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if MODEL_PATH.exists():
            st.success("Model: Loaded ✅")
        else:
            st.error("Model: Not Found ❌")
            st.info("Run `python train.py` to train and save the model.")

    return conf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    conf_threshold = render_sidebar()

    st.title("🅿️ Parking Space Occupancy Detector")
    st.markdown(
        "Upload a photo of a parking lot and the system will detect and "
        "classify each parking space using a YOLOv8 model trained on your dataset."
    )

    # ── Model loading ──────────────────────────────────────────────────────
    model       = None
    model_ok    = False
    class_names = []

    if not MODEL_PATH.exists():
        st.error(
            f"Model not found at `{MODEL_PATH}`.  "
            "Run **`python train.py`** first to train and save the model."
        )
    else:
        try:
            with st.spinner("Loading model …"):
                model = load_detection_model()
            # Get actual class names from the loaded model
            class_names = list(model.names.values())
            model_ok    = True
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")

    # ── File uploader ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose a parking lot image",
        type=["jpg", "jpeg", "png"],
        disabled=not model_ok,
        help="Upload a JPG or PNG aerial/ground-level photo of a parking lot.",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return

    if not model_ok or model is None:
        st.warning("Model is not available. Please train the model first.")
        return

    # ── Inference ──────────────────────────────────────────────────────────
    pil_image = Image.open(uploaded).convert("RGB")
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Analyzing parking lot …"):
        try:
            detections = run_detection(model, bgr_image, conf_threshold)
        except Exception as exc:
            st.error(f"Inference error: {exc}")
            return

    if not detections:
        st.warning(
            "No parking spaces detected. "
            "Try lowering the **Confidence Threshold** in the sidebar "
            "or upload a clearer image."
        )

    # ── Annotate ───────────────────────────────────────────────────────────
    # draw_results called with 2 args — class_names derived inside from detections
    annotated_bgr = draw_results(bgr_image, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # ── Two-column image display ───────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("#### Original Image")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.markdown("#### Detection Results")
        st.image(annotated_pil, use_container_width=True)

    # ── Summary section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🅿️ Parking Lot Summary")

    # Pass actual class_names so get_summary uses dynamic matching
    summary        = get_summary(detections, class_names)
    total          = summary["total"]
    counts         = summary["counts"]
    rate           = summary["occupancy_rate"]

    free_count     = counts.get("free", 0)
    occupied_count = counts.get("occupied", 0)
    partial_count  = counts.get("partially_occupied", 0)
    not_free       = occupied_count + partial_count

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Spaces",    total)
    mc2.metric("Free Spaces",     free_count)
    mc3.metric("Occupied Spaces", not_free)
    mc4.metric("Occupancy Rate",  f"{rate:.1f}%")

    # Detailed breakdown if partially_occupied spaces detected
    if partial_count > 0:
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Fully Occupied",     occupied_count)
        bc2.metric("Partially Occupied", partial_count)
        bc3.metric("Occupancy Rate",     f"{rate:.1f}%")

    # Progress bar
    render_occupancy_bar(rate)

    # ── Download button ────────────────────────────────────────────────────
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Annotated Image",
        data=buf.getvalue(),
        file_name="parking_detection_result.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
