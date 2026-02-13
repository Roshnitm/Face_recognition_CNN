import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from src.utils import preprocess_face, emotion_labels

model = load_model("models/emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = preprocess_face(face)

        pred = model.predict(face, verbose=0)
        label = emotion_labels[np.argmax(pred)]

        # choose color per emotion 😄
        color = (0,255,0) if label=="Happy" else (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

    return frame