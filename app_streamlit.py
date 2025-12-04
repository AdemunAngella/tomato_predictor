# app_streamlit.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os, uuid, io

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="Tomato Quality Predictor",
    page_icon="static/favicon.png" if os.path.exists("static/favicon.png") else "🍅",
    layout="wide"
)

STATIC_DIR = "static"
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

LOGO = os.path.join(STATIC_DIR, "logo.png")
FRESH_ICON = os.path.join(STATIC_DIR, "fresh.png")
ROTTEN_ICON = os.path.join(STATIC_DIR, "rotten.png")

MODEL_PATH = os.path.join("models", "tomato_grading_model.h5")
IMG_DISPLAY_SIZE = (526, 250)   # preview display size (width, height)
IMG_MODEL_SIZE = (224, 224)     # model input size

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
# SESSION STATE init
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {time, label, confidence}

# ---------------------------
# CSS (cards, dashed box)
# ---------------------------
BASE_CSS = """
<style>
body {
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
    color: #000;
    background-color: #f5f6f7;
}
.st-emotion-cache-4rsbii{ 
    background: #fafbfb;
}

/* Header area */
.st-emotion-cache-zy6yx3 {
    padding: 1rem 7.5rem 0rem;
}
.st-emotion-cache-1permvm {
    background: #fff;
    padding: 6.5px 110px;
    box-shadow: 0 .125rem .25rem rgba(0, 0, 0, 0.075);
    position: absolute;
    top: 0;
    right: 0px;
    align-items: center;
    z-index: 999990;
}
.st-emotion-cache-qvj5rf {
    width: calc(5% - 1rem);
    flex: 1 1 calc(5% - 1rem);
}
.st-emotion-cache-1bmqxsm {
    width: calc(95% - 1rem);
    flex: 1 1 calc(95% - 1rem);
}
.st-emotion-cache-3uj0rx {
    margin-bottom: 0rem;
}
.brand-title { 
    font-weight:700; 
    font-size:19px; 
    margin:0; 
}
.brand-sub {
    font-size: 19px;
    color: #4a5568;
    margin-bottom: 20px 0;
    font-weight: 300;
}
/* dashed drop area */
.drop-area {
  border: 2px dashed #1e90ff;
  border-radius: 10px;
  padding: 22px;
  text-align:center;
  background: #f8fbff;
  color:#374151;
}

/* Red button */
.stButton button[kind="primary"], 
.stButton button {
    background-color: #dc2626 !important;
    color: #fff !important;
    border: none !important;
}
.stButton button:hover {
    background-color: #b91c1c !important;
}

.muted { 
    color:#6b7280; font-size:18px; font-weight: 500;
}

.result-card { text-align:center; padding-top:6px; }
.st-emotion-cache-12tkkxz {
    background: #fff;
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
}
.st-emotion-cache-1sb501b {
    color: #000 !important;
    font-weight: 500;
    font-size: 18px;
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;     
}
/* Target Streamlit buttons */
div.stButton > button {
    border-radius: 5px !important;
}
/* Center all Streamlit images (including your icon) */
.st-emotion-cache-uwwqev {
    display: flex;
    justify-content: center;
}
/* footer */
.footer { 
    background: rgb(33 37 41); 
    font-size: 15px; 
    color:#fff; 
    padding:13px; 
    text-align:center; 
    margin-top:37px; 
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------------------
# small helper: center-crop & resize for preview
# ---------------------------
def make_preview_image(pil_img: Image.Image, size=IMG_DISPLAY_SIZE):
    img = ImageOps.fit(pil_img, size, Image.Resampling.LANCZOS, centering=(0.5,0.5))
    return img

# ---------------------------
# helper: prepare for model
# ---------------------------
def prepare_for_model(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    img = img.resize(IMG_MODEL_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# ---------------------------
# helper: save image to uploads
# ---------------------------
def save_upload(pil_img: Image.Image, prefix="img"):
    fn = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(UPLOAD_DIR, fn)
    pil_img.save(path)
    return path

# ---------------------------
# HEADER
# ---------------------------
col1, col2 = st.columns([0.15, 0.85])

with col1:
    if os.path.exists(LOGO):
        st.image(LOGO, width=45)

with col2:
    st.markdown('<div class="brand-title">Tomato Quality Predictor</div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------
# MAIN LAYOUT
# ---------------------------
left, right = st.columns([0.5, 0.5], gap="large")

# LEFT
with left:
    st.markdown(
        """
        <div class="brand-sub">
            Upload a photo of your tomato, and our AI model will instantly 
            predict whether it is fresh or rotten.
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("##### Upload or Take a Photo")
    st.markdown('<div class="drop-area">Drag & drop an image here</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    pil_image = None
    if uploaded is not None:
        try:
            pil_image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            save_upload(pil_image, prefix="upl")
        except Exception:
            st.error("Could not read uploaded image. Try a different file.")

    if pil_image is not None:
        preview = make_preview_image(pil_image, IMG_DISPLAY_SIZE)
        st.image(preview, caption="Preview", width=IMG_DISPLAY_SIZE[0])
    else:
        st.info("No preview yet. Upload a photo to continue.")

    analyze_clicked = st.button("Analyze", use_container_width=True)

    if analyze_clicked:
        if pil_image is None:
            st.error("Please upload an image first.")
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
                    "confidence": round(confidence * 100, 2)
                })
                
                # Results heading
                st.markdown(
                    "<h4 style='text-align:center; margin-top:20px;'>Results:</h4>",
                    unsafe_allow_html=True
                )

                if os.path.exists(icon_path):
                    st.image(icon_path, width=42)

                color = "#28a745" if "Fresh" in label else "#dc2626"

                # Centered label
                st.markdown(
                    f"<h4 style='text-align:center; color:{color}; margin-bottom:-20px;'>{label}</h4>",
                    unsafe_allow_html=True
                )

                # Centered confidence
                st.markdown(
                    f"<p style='text-align:center; color:#6b7280; margin-top:0;'>Confidence: {round(confidence * 100, 2)}%</p>",
                    unsafe_allow_html=True
                )

# RIGHT
with right:
    st.markdown(
        """
        <div class="brand-sub">
            Track your predictions and view the history of results — all from a user-friendly interface.
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("##### Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Cumulative count
        fresh_cum = []
        rotten_cum = []
        f = r = 0

        for _, row in df.iterrows():
            if "Fresh" in row["label"]:
                f += 1
            else:
                r += 1
            fresh_cum.append(f)
            rotten_cum.append(r)

        # PLOT (Green = Fresh, Red = Rotten)
        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)

        ax.plot(fresh_cum, label="Fresh", linewidth=2.5, color="#22c55e")   # Green
        ax.plot(rotten_cum, label="Rotten", linewidth=2.5, color="#ef4444") # Red

        # Force numbers on axes
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Count")

        ax.set_xticks(range(len(fresh_cum)))
        ax.set_xticklabels(range(1, len(fresh_cum) + 1))  # start from 1

        ax.set_yticks(range(0, max(max(fresh_cum), max(rotten_cum)) + 1))

        ax.legend()

        # Set legend color to match line colors
        # for text, line in zip(legend.get_texts(), [fresh_line, rotten_line]):
        #     text.set_color(line.get_color())

        ax.grid(True, linestyle="--", alpha=0.4)

        st.pyplot(fig, use_container_width=True)

    st.markdown("##### Recent Predictions")

    if len(st.session_state.history) != 0:

        df_display = pd.DataFrame(st.session_state.history)

        # Rename columns
        df_display = df_display.rename(columns={
            "date": "Time (UTC)",
            "label": "Result",
            "confidence": "Confidence (%)"
        })

        # Reverse order: most recent first
        df_display = df_display.iloc[::-1].reset_index(drop=True)

        # Set index to start from 1 and name it "No."
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = "No."

        # Show only last 20
        st.table(df_display.head(20))

        if st.button("Clear History"):
            st.session_state.history = []
            st.experimental_rerun()

# Footer
year = datetime.now().year
st.markdown(f'<div class="footer">© {year} Tomato Quality Predictor. All Rights Reserved.</div>', unsafe_allow_html=True)
