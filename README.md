# 🏏 CrickVision — AI-Powered Cricket Shot Analysis

<p align="center">
  <b>Deep Learning • Computer Vision • Video Analytics • ML Deployment</b>
</p>

CrickVision is a **deep learning–based video analysis system** that classifies cricket shots directly from video inputs.  
It combines **computer vision** and **temporal sequence modeling** to understand both the *visual appearance* and *motion dynamics* of cricket strokes.

The project is deployed as an **interactive web application**, demonstrating an **end-to-end ML pipeline** — from model architecture to cloud deployment.

---

## 🌐 Live Demo
🔗 **Web App:** https://crickvision-docker.onrender.com/  
🔗 **GitHub Repo:** https://github.com/avsharma01/video-analysis-for-cricket

> ⚠️ *Note:* On free cloud tiers, prediction may fail for very long videos due to memory limits.  
> Use short clips (5–10 seconds) for best performance.

---

## ✨ Key Highlights
- 🎥 Video-based cricket shot classification  
- 🏏 Supports **10 different cricket shot types**  
- 🧠 CNN + RNN architecture for spatial & temporal learning  
- 📊 Confidence-based predictions with commentary-style output  
- 🌐 Interactive Streamlit web interface  
- 📦 Dockerized and cloud deployed  

---

## 🏏 Supported Shot Types
- Cover  
- Defense  
- Flick  
- Hook  
- Late Cut  
- Lofted  
- Pull  
- Square Cut  
- Straight  
- Sweep  

---

## 🧠 Model Architecture
- **CNN Backbone:** EfficientNetB0 (pretrained on ImageNet)  
- **Temporal Modeling:** GRU layers for sequence learning  
- **Input:** Extracted video frames  
- **Output:** Softmax probabilities over 10 shot classes  

The model processes frames sequentially, enabling **motion-aware action recognition**, which is critical for sports video analysis.

---

## 🛠 Tech Stack
- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Web Framework:** Streamlit  
- **Deployment:** Docker, Cloud Hosting  

---

## 📁 Project Structure
video-analysis-for-cricket/
│
├── app.py # Streamlit web application
├── model.h5 # Trained deep learning model
├── reqthing.txt # Project dependencies
├── Dockerfile # Docker configuration
├── README.md # Project documentation
└── .gitignore


---

## ⚙️ How It Works
1. User uploads a cricket shot video  
2. Video frames are extracted and preprocessed  
3. EfficientNet extracts spatial features from frames  
4. GRU layers model temporal motion patterns  
5. Model predicts the shot type with confidence  
6. Result is displayed with commentary-style description  

---

## ▶️ Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/avsharma01/video-analysis-for-cricket.git
cd video-analysis-for-cricket

---

2️⃣ Create & Activate Virtual Environment (Python 3.10)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r reqthing.txt

4️⃣ Run the Application
streamlit run app.py


Open in browser:

http://localhost:8501

🐳 Run Using Docker
docker build -t crickvision .
docker run -p 8501:8501 crickvision

📌 Learning Outcomes

Video-based deep learning using CNN-RNN architectures

Efficient frame extraction and preprocessing

Building interactive ML web applications

Dockerizing and deploying ML models

Handling real-world deployment constraints

👤 Author

Anant Vaibhav
B.Tech Computer Science (Artificial Intelligence & Machine Learning)

🔗 GitHub: https://github.com/avsharma01

⭐ Acknowledgements

TensorFlow & Keras

Streamlit

OpenCV

<p align="center"> ⭐ If you found this project interesting, consider giving it a star! </p> ```+
