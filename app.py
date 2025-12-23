import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0
import tempfile, shutil, base64, random

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="CrickVision - Cricket Shot Classifier", layout="wide")

# -----------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8f5e9 0%, #ffffff 80%);
        font-family: 'Poppins', sans-serif;
        color: #1b1b1b;
        overflow-x: hidden;
        padding: 0;
    }
    html, body { margin:0; padding:0; }

    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] {
        visibility: hidden;
        display: none;
    }
    a.anchor, a.anchor > svg {display: none !important; visibility: hidden !important;}

    .navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 72px;
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 80px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.18);
        z-index: 999;
    }
    .navbar .brand { font-size:26px; font-weight:700; color:white; }
    .navbar .links a { color:white; text-decoration:none; margin-left:30px; font-weight:500; }
    .navbar .links a:hover { color:#c8e6c9; }

    .main {
        margin-top: 72px;
        padding: 40px 60px 100px;
        max-width: 1300px;
        margin-left: auto;
        margin-right: auto;
        box-sizing: border-box;
    }

    h1,h2,h3 { text-align:center; color:#1b5e20; font-weight:700; }

    div.stButton > button {
        background:#388e3c; color:white; border-radius:8px; padding:0.7em 1.8em; font-weight:600;
    }
    section[data-testid="stFileUploader"] {
        border:2px dashed #388e3c; border-radius:12px; padding:1em; background: rgba(56,142,60,0.05);
    }

    .footer { text-align:center; color:#444; font-size:14px; margin-top:60px; font-weight:500; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# NAVBAR
# -----------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="brand">🏏 CrickVision</div>
    <div class="links">
        <a href="?page=predict">Predict</a>
        <a href="?page=compare">Compare</a>
        <a href="https://github.com/avsharma01" target="_blank">GitHub</a>
        <a href="#">By Anant Vaibhav</a>
    </div>
</div>
<div class="main">
""", unsafe_allow_html=True)

# -----------------------------------------------------
# CLASS DISPLAY (ALL 10 SHOTS)
# -----------------------------------------------------
st.markdown("""
<div style='margin-top: 80px; text-align: center;'>
    <h4 style='color:#1b5e20; font-weight:700;'>Featureing shot types</h4>
    <div style='display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin-top:18px;'>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Cover</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Defense</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Flick</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Hook</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Late Cut</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Lofted</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Pull</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Square Cut</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Straight</div>
        <div style='background:#e8f5e9; border:2px solid #388e3c; border-radius:10px; padding:10px 18px; font-weight:600; color:#1b5e20;'>Sweep</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# VIDEO DISPLAY FIX (Base64 Embed)
# -----------------------------------------------------
def show_video(video_path):
    try:
        with open(video_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        video_html = f"""
            <video width="720" height="480" controls>
                <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying video: {e}")

# -----------------------------------------------------
# MODEL + HELPERS
# -----------------------------------------------------
classes = {'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 'late_cut': 4,
           'lofted': 5, 'pull': 6, 'square_cut': 7, 'straight': 8, 'sweep': 9}

def load_model(weights_path):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    model = models.Sequential([
        layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.GRU(256, return_sequences=True),
        layers.GRU(128),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.load_weights(weights_path)
    return model

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame.numpy()

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=1):
    result = []
    src = cv2.VideoCapture(str(video_path))
    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = src.read()
    if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
    else:
        result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmpfile:
        shutil.copyfileobj(uploaded_file, tmpfile)
        return tmpfile.name

# -----------------------------------------------------
# Commentary Sentences
# -----------------------------------------------------
def classy_sentence(label, conf):
    shots = {
        "cover": [
            "What a lovely cover drive!",
            "That’s a textbook cover — pure class.",
            "Effortless timing through the covers!",
            "A graceful cover drive — poetry in motion."
        ],
        "pull": [
            "The batsman played a crisp pull shot!",
            "Short and punished — brilliant pull!",
            "That’s a commanding pull, pure authority.",
            "He rocks back and pulls it beautifully!"
        ],
        "hook": [
            "A brave hook — pure timing!",
            "Took on the short ball — masterful hook!",
            "What courage! A fearless hook shot.",
            "That hook was all about confidence."
        ],
        "lofted": [
            "A majestic lofted shot straight down the ground!",
            "That’s gone miles — stunning lofted hit!",
            "He lifts it high and handsome!",
            "Beautiful lofted stroke with effortless power."
        ],
        "square_cut": [
            "A sharp square cut — brilliant placement!",
            "He carves it square with precision.",
            "Perfect timing on that square cut!",
            "That’s a classic square cut past point!"
        ],
        "defense": [
            "Solid defense — textbook technique.",
            "He meets it with the full face of the bat.",
            "Rock-solid defense, unshakable balance.",
            "Compact technique, blocking it with confidence."
        ],
        "straight": [
            "Beautiful straight drive down the pitch!",
            "Straight as an arrow — wonderful drive!",
            "That’s elegance defined — a perfect straight drive.",
            "Pure timing! Straight down the ground."
        ],
        "sweep": [
            "A graceful sweep shot played with control!",
            "Gets down low and sweeps it nicely!",
            "That’s a fine sweep, placed perfectly.",
            "Brilliant sweep — reading the length early."
        ],
        "late_cut": [
            "Delicate late cut, perfectly executed!",
            "Guides it past the keeper — exquisite touch.",
            "Beautifully late on the shot — wonderful placement.",
            "That’s finesse — a perfect late cut."
        ],
        "flick": [
            "A stylish flick through mid-wicket!",
            "Elegant wrist work — brilliant flick shot.",
            "That’s trademark wristy elegance!",
            "He just flicks it — pure timing!"
        ]
    }

    desc = random.choice(shots.get(label, [f"A nice {label} shot."]))
    return f"{desc} (Confidence: {conf:.1f}%)"

# -----------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------
@st.cache_resource
def _load_model_singleton(weights_path="model.h5"):
    return load_model(weights_path)

# Load model once silently
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = True
    model = _load_model_singleton()
else:
    model = _load_model_singleton()

# -----------------------------------------------------
# Router
# -----------------------------------------------------
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["predict"])[0]

# -----------------------------------------------------
# Predict Page
# -----------------------------------------------------
if page == "predict":
    st.title("🏏 Shot Prediction")
    st.write("Upload a video and watch CrickVision describe the shot like a commentator.")

    video_file = st.file_uploader("Upload your cricket shot video", type=["mp4", "avi", "mov"], key="predict")
    if video_file:
        video_path = save_uploaded_file(video_file)
        show_video(video_path)

        frames = frames_from_video_file(video_path, 30)

        with st.spinner("Loading model..."):
            pass  # just display message, don't reload anything

        with st.spinner("Analysing shot..."):
            pred = model.predict(np.expand_dims(frames, axis=0))

        label_idx = np.argmax(pred, axis=1)[0]
        label = list(classes.keys())[list(classes.values()).index(label_idx)]
        conf = pred[0][label_idx] * 100
        st.success(classy_sentence(label, conf))

# -----------------------------------------------------
# Compare Page
# -----------------------------------------------------
elif page == "compare":
    st.title("⚖️ Compare Two Shots")
    st.write("Upload two shots and see how similar their motion patterns are.")

    col1, col2 = st.columns(2)
    class1 = conf1 = class2 = conf2 = None

    with col1:
        video1 = st.file_uploader("First video", type=["mp4", "avi", "mov"], key="vid1")
        if video1:
            video1_path = save_uploaded_file(video1)
            show_video(video1_path)
            frames1 = frames_from_video_file(video1_path, 30)
            with st.spinner("Analysing first video..."):
                pred1 = model.predict(np.expand_dims(frames1, axis=0))
            idx1 = np.argmax(pred1, axis=1)[0]
            class1 = list(classes.keys())[list(classes.values()).index(idx1)]
            conf1 = pred1[0][idx1] * 100
            st.success(classy_sentence(class1, conf1))

    with col2:
        video2 = st.file_uploader("Second video", type=["mp4", "avi", "mov"], key="vid2")
        if video2:
            video2_path = save_uploaded_file(video2)
            show_video(video2_path)
            frames2 = frames_from_video_file(video2_path, 30)
            with st.spinner("Analysing second video..."):
                pred2 = model.predict(np.expand_dims(frames2, axis=0))
            idx2 = np.argmax(pred2, axis=1)[0]
            class2 = list(classes.keys())[list(classes.values()).index(idx2)]
            conf2 = pred2[0][idx2] * 100
            st.success(classy_sentence(class2, conf2))

    if st.button("Compare Videos"):
        if video1 and video2 and class1 == class2:
            feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
            f1 = feature_model.predict(np.expand_dims(frames1, axis=0))
            f2 = feature_model.predict(np.expand_dims(frames2, axis=0))
            dot = np.dot(f1, f2.T)
            sim = (dot / (np.linalg.norm(f1) * np.linalg.norm(f2)))[0][0] * 100
            st.success(f"Both are **{class1}** shots! Similarity: {sim:.2f}%")
        elif class1 and class2 and class1 != class2:
            st.warning("Different shot types detected — similarity skipped.")
        else:
            st.warning("Upload both videos to compare.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown('<div class="footer">:)</div>', unsafe_allow_html=True)
