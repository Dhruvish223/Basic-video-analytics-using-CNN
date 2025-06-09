import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np

# Set Streamlit config
st.set_page_config(page_title="Video Analytics using CNN", layout="centered")

# STUN config for WebRTC handshake
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("📹 Real-Time Video Analytics with CNN")

st.markdown(
    """
This app uses your browser’s webcam to perform real-time video analysis using a simulated CNN model.
If running on Streamlit Cloud, make sure your browser allows webcam access.
"""
)

# --- CNN-like processing placeholder ---
def fake_cnn_processor(image):
    """Simulate a CNN processing pipeline (e.g., edge detection)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.Canny(gray, 100, 200)  # Just to simulate CNN-like output
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Return 3-channel image


# --- Frame callback ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    processed_img = fake_cnn_processor(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# --- Webcam streamer ---
webrtc_streamer(
    key="video-analytics",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
