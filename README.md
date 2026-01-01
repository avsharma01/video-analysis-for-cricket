🏏 CrickVision – AI-Powered Cricket Shot Analysis

CrickVision is a deep learning–based cricket shot analysis system that classifies cricket shots directly from video inputs.
It uses a combination of computer vision and sequence modeling to understand both spatial and temporal patterns in cricket strokes.

The project is deployed as an interactive web application and demonstrates an end-to-end ML workflow — from model design to cloud deployment.

🚀 Live Demo
🔗 Web App: https://crickvision-docker.onrender.com/

🧠 Key Features
🎥 Video-based cricket shot classification
🏏 Supports 10 different cricket shots
🧠 CNN + RNN architecture for spatial & temporal learning
📊 Confidence-based predictions with commentary-style output
🌐 Interactive Streamlit web interface
📦 Dockerized and deployed on cloud

🏏 Supported Shot Types
-Cover
-Defense
-Flick
-Hook
-Late Cut
-Lofted
-Pull
-Square Cut
-Straight
-Sweep

🏗️ Model Architecture
-CNN Backbone: EfficientNetB0 (pretrained on ImageNet)
-Temporal Modeling: GRU layers for sequence learning
-Input: Extracted video frames
-Output: Softmax probabilities over 10 shot classes
-The model processes video frames sequentially to capture motion dynamics, making it suitable for video-based action recognition.

🛠 Tech Stack
-Programming Language: Python
-Deep Learning: TensorFlow / Keras
-Computer Vision: OpenCV
-Web Framework: Streamlit
-Deployment: Docker, Cloud Hosting

📁 Project Structure
video-analysis-for-cricket/
│
├── app.py              # Streamlit web application
├── model.h5            # Trained deep learning model
├── reqthing.txt        # Project dependencies
├── Dockerfile          # Docker configuration
├── README.md           # Project documentation
└── runtime.txt         # Python runtime specification

⚙️ How It Works
-User uploads a cricket shot video
-Video frames are extracted and preprocessed
-Frames are passed through EfficientNet for feature extraction
-GRU layers model temporal motion patterns
-Model predicts the shot type with confidence score
-Result is displayed with commentary-style description

▶️ Run Locally
1️⃣ Clone the Repository
git clone https://github.com/avsharma01/video-analysis-for-cricket.git
cd video-analysis-for-cricket

2️⃣ Create & Activate Virtual Environment (Python 3.10)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r reqthing.txt

4️⃣ Run the App
streamlit run app.py


App will open at:
http://localhost:8501

🐳 Run Using Docker
docker build -t crickvision .
docker run -p 8501:8501 crickvision

📌 Learning Outcomes
-Video-based deep learning using CNN-RNN architectures
-Efficient frame extraction and preprocessing
-Building interactive ML web apps
-Dockerizing and deploying ML models
-Bridging the gap between model training and real-world usage

👤 Author

Anant Vaibhav
B.Tech Computer Science (Artificial Intelligence & Machine Learning)

🔗 GitHub: https://github.com/avsharma01

⭐ Acknowledgements
-TensorFlow & Keras
-Streamlit
-OpenCV

