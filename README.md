Video Analytics for Cricket
Purpose

Classify cricket shots from video input using a trained deep learning model and a Streamlit interface.

Structure

src/app: UI layer.

src/core: model loading, inference, preprocessing, postprocessing.

src/utils: video and file I/O.

src/config: paths and constants.

models: stored weights.

data/TestSamples: sample videos.

notebooks: experimentation artifacts.

tests: verification scripts.

app.py: entry point.

Pipeline

Load video.

Extract and normalize frames.

Run model inference.

Map logits to class labels.

Return structured output to UI.

Requirements

Python 3.10+
TensorFlow
OpenCV
Streamlit
NumPy

Usage

Install dependencies:
pip install -r requirements.txt

Start interface:
streamlit run app.py

Model

Stored at models/model.h5. Expects preprocessed frames of the defined input shape.

Notes

Notebook contains experimental logic. Source directory holds production logic.