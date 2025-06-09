import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

st.set_page_config(page_title="Live Webcam Analytics", layout="centered")
st.title("🎯 Real-Time Face Detection & Alignment (Streamlit Cloud Compatible)")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Face detection using cvlib
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            img_h, img_w, _ = img.shape
            alignment = "Centered ✅" if abs(center_x - img_w // 2) < img_w * 0.1 else "Not Centered ❌"
            cv2.putText(img, alignment, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if "✅" in alignment else (0, 0, 255), 2)
            
        for face in faces:
            startX, startY, endX, endY = face
            center_x = (startX + endX) // 2
            center_y = (startY + endY) // 2

            # Draw rectangle around face
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)

            # Check alignment
            img_h, img_w, _ = img.shape
            alignment = "Centered ✅" if abs(center_x - img_w // 2) < img_w * 0.1 else "Not Centered ❌"
            cv2.putText(img, alignment, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if "✅" in alignment else (0, 0, 255), 2)

        return img

webrtc_streamer(key="face-detect", video_transformer_factory=VideoTransformer)
