# 🎭 Face Emotion Recognition using CNN

A Deep Learning-based Face Emotion Recognition system built using **Convolutional Neural Networks (CNN)**.  
The model detects human facial emotions from images or webcam input in real-time.

This project uses OpenCV for face detection and TensorFlow for emotion classification.

---

## 📌 Detected Emotions

The model classifies facial expressions into the following 7 categories:

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😄 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

---

## 🚀 Project Overview

This project implements a Convolutional Neural Network (CNN) to recognize facial emotions.

Workflow:

1. Capture image (Webcam / Upload)
2. Detect face using OpenCV
3. Preprocess image (resize, normalize, grayscale if required)
4. Pass image to trained CNN model
5. Display predicted emotion label on the image

The system is designed to be modular, scalable, and portfolio-ready.

---

## 🛠️ Tech Stack

- TensorFlow == 2.20.0  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```
### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install tensorflow==2.20.0 opencv-python numpy pandas matplotlib scipy
```

OR
```bash
pip install -r requirements.txt
```

## ▶️ Running the Application
### Run Web Application
```bash
python train.py
python test.py
python live_emotions.py
```

