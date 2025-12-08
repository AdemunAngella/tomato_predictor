# app_streamlit.py# ---------------------------
# IMPORTS
# ---------------------------
import os
import io
import uuid
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Tomato Quality Predictor",
    page_icon="static/favicon.png" if os.path.exists("static/favicon.png") else "🍅",
    layout="wide"
)

# ---------------------------
# STATIC / UPLOAD DIRECTORIES
# ---------------------------
STATIC_DIR = "static"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# MODEL PATH
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tomato_predictor.keras")

# ---------------------------
# LOGO / ICONS
# ---------------------------
LOGO = os.path.join(STATIC_DIR, "logo.png")
FRESH_ICON = os.path.join(STATIC_DIR, "fresh.png")
ROTTEN_ICON = os.path.join(STATIC_DIR, "rotten.png")

# ---------------------------
# IMAGE SETTINGS
# ---------------------------
IMG_DISPLAY_SIZE = (570, 400)
IMG_MODEL_SIZE = (224, 224)

# ---------------------------
# LOAD MODEL (cached)
# ---------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(path)

MODEL = load_model()

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {date, label, confidence}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

# Save uploaded image
def save_upload(pil_img: Image.Image, prefix="img"):
    fn = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(UPLOAD_DIR, fn)
    pil_img.save(path)
    return path

# Make center-cropped preview image
def make_preview_image(pil_img: Image.Image, size=IMG_DISPLAY_SIZE):
    img = ImageOps.fit(pil_img, size, Image.Resampling.LANCZOS, centering=(0.5,0.5))
    return img

# Prepare image for model
def prepare_for_model(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_MODEL_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0  # scale 0-1
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr

# ---------------------------
# CSS (Styling)
# ---------------------------
BASE_CSS = """
<style>
body {
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
    color: #000;
    background-color: #f5f6f7;
}
.st-emotion-cache-tn0cau {
    gap: 0;
}

/* Header / Navigation */
.st-emotion-cache-1permvm {
    background: #fff;
    padding: 5px 120px;
    box-shadow: 0 .125rem .25rem rgba(0, 0, 0, 0.075);
    position: absolute;
    top: 0;
    right: 0px;
    align-items: center;
    z-index: 999990;
    gap: 0;
}
.st-emotion-cache-uwwqev {
    justify-content: left !important;
}
.st-emotion-cache-1ilw1s0 {
    width: calc(5% - 1rem);
    flex: 1 1 calc(5% - 1rem);
}
.st-emotion-cache-1ju2zhm {
    width: calc(95% - 1rem);
    flex: 1 1 calc(95% - 1rem);
}
.brand-title { 
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
    font-weight: 700; 
    font-size: 19px;
    margin-top: -10px !important;
}
.st-emotion-cache-pk3c77 {
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
.brand-sub {
    font-size: 17px;
    color: #4a5568;
    margin: 5px 0 15px;
    font-weight: 300;
}

/* Drop area */
.drop-area {
  border: 2px dashed #1e90ff;
  border-radius: 10px;
  padding: 22px;
  text-align:center;
  background: #f8fbff;
  color:#374151;
}

/* Red button */
.stButton > button {
    background-color: #dc2626 !important;
    color: #fff !important;
    border: none !important;
    border-radius:5px !important;
}
.stButton > button:hover {
    background-color: #b91c1c !important;
}

/* Preview / Result */
.result-card { 
    text-align: center;
    padding-top: 6px;
}
.muted {
    color:#6b7280;
    font-size:18px;
    font-weight: 500;
}

/* Footer */
.footer { 
    background: rgb(33 37 41); 
    font-size: 15px; 
    color: #fff; 
    padding: 13px; 
    text-align: center; 
    margin-top: 37px; 
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ---------------------------
# NAVIGATION HEADER
# ---------------------------
col1, col2 = st.columns([0.15, 0.85], gap="small")

with col1:
    if os.path.exists(LOGO):
        st.image(LOGO, width=50)

with col2:
    st.markdown('<div class="brand-title">Tomato Quality Predictor</div>', unsafe_allow_html=True)

st.markdown("", unsafe_allow_html=True)

# ---------------------------
# MAIN LAYOUT
# ---------------------------
left, right = st.columns([0.5, 0.5], gap="large")

# LEFT PANEL
with left:
    st.markdown("""
        <div class="brand-sub">
            Upload a photo of your tomato, and our AI model will instantly 
            predict whether it is fresh or rotten.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Upload or Take a Photo")
    st.markdown('<div class="drop-area">Drag & drop an image here</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    camera_file = st.camera_input("", label_visibility="collapsed")

    pil_image = None

    if uploaded is not None:
        try:
            pil_image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            save_upload(pil_image, prefix="upl")
        except Exception:
            st.error("Could not read uploaded image. Try a different file.")
    elif camera_file is not None:
        try:
            pil_image = Image.open(io.BytesIO(camera_file.getvalue())).convert("RGB")
            save_upload(pil_image, prefix="cam")
        except Exception:
            st.error("Could not read camera image.")

    if pil_image is not None:
        preview = make_preview_image(pil_image, IMG_DISPLAY_SIZE)
        st.image(preview, caption="Preview", width=IMG_DISPLAY_SIZE[0])
    else:
        st.info("No preview yet. Upload or take a photo to continue.")

    analyze_clicked = st.button("Analyze", use_container_width=True)

    if analyze_clicked:
        if pil_image is None:
            st.error("Please upload or take a photo first.")
        else:
            with st.spinner("Analyzing image..."):
                arr = prepare_for_model(pil_image)
                pred = MODEL.predict(arr)[0][0]

                if pred > 0.5:
                    label = "Fresh Tomato"
                    confidence = float(pred)
                    icon_path = FRESH_ICON
                else:
                    label = "Rotten Tomato"
                    confidence = float(1 - pred)
                    icon_path = ROTTEN_ICON

                st.session_state.history.append({
                    "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": label,
                    "confidence": round(confidence*100,2)
                })

                # st.markdown("<h4 style='text-align:center; margin-top:20px;'>Results:</h4>", unsafe_allow_html=True)
                # if os.path.exists(icon_path):
                #     st.image(icon_path, width=42)

                color = "#28a745" if "Fresh" in label else "#dc2626"
                st.markdown(f"<h4 style='text-align:center; color:{color}; margin-bottom:-10px; margin-top: -75px;'>{label}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; color:#6b7280; margin-top:-50px;'>Confidence: {round(confidence,2)}%</p>", unsafe_allow_html=True)

# RIGHT PANEL
with right:
    st.markdown("""
        <div class="brand-sub">
            Track your predictions and view the history of results — all from a user-friendly interface.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Prediction History")
    if len(st.session_state.history)==0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Cumulative counts
        fresh_cum = []
        rotten_cum = []
        f = r = 0
        for _, row in df.iterrows():
            if "Fresh" in row["label"]: f+=1
            else: r+=1
            fresh_cum.append(f)
            rotten_cum.append(r)

        # Plot cumulative predictions
        fig, ax = plt.subplots(figsize=(6,2.2), dpi=100)
        ax.plot(fresh_cum, label="Fresh", linewidth=2.5, color="#22c55e")
        ax.plot(rotten_cum, label="Rotten", linewidth=2.5, color="#ef4444")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(fresh_cum)))
        ax.set_xticklabels(range(1,len(fresh_cum)+1))
        ax.set_yticks(range(0,max(max(fresh_cum),max(rotten_cum))+1))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig, use_container_width=True)

        # Recent predictions table
        st.markdown("##### Recent Predictions")
        df_display = df.rename(columns={"date":"Time (UTC)","label":"Result","confidence":"Confidence (%)"}).iloc[::-1].reset_index(drop=True)
        df_display.index = range(1,len(df_display)+1)
        df_display.index.name = "No."
        st.table(df_display.head(20))

        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()

# ---------------------------
# FOOTER
# ---------------------------
year = datetime.now().year
st.markdown(f'<div class="footer">© {year} Tomato Quality Predictor. All Rights Reserved.</div>', unsafe_allow_html=True)
