import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(layout="wide")
st.title("📹 Live Webcam Analytics Dashboard")

# Initialize mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=False)

# Function to analyze gaze direction
def is_looking_center(landmarks, image_w):
    left_eye = landmarks[33]  # Approx left eye
    right_eye = landmarks[263]  # Approx right eye
    eye_center_x = (left_eye.x + right_eye.x) / 2
    return 0.4 < eye_center_x < 0.6

# Function to check if face is centered
def is_face_centered(landmarks, image_w):
    nose = landmarks[1]
    return 0.4 < nose.x < 0.6

# Function to draw face landmarks
def draw_landmarks(image, landmarks, connections, color=(0, 255, 0)):
    h, w = image.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 1, color, -1)

# Main loop
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb)
        results_pose = pose.process(rgb)

        h, w, _ = frame.shape
        status_text = ""

        if results_face.multi_face_landmarks:
            landmarks = results_face.multi_face_landmarks[0].landmark
            draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Gaze detection
            looking = is_looking_center(landmarks, w)
            gaze_status = "Looking at Camera ✅" if looking else "Not Looking ❌"
            color = (0, 255, 0) if looking else (0, 0, 255)
            cv2.putText(frame, gaze_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Center alignment
            centered = is_face_centered(landmarks, w)
            alignment_status = "Centered ✅" if centered else "Not Centered ❌"
            color2 = (0, 255, 0) if centered else (0, 0, 255)
            cv2.putText(frame, alignment_status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color2, 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
