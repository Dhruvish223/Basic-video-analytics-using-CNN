import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
import math

# Page configuration
st.set_page_config(
    page_title="Live Video Analytics Dashboard",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe_models():
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_detection, face_mesh, pose, mp_drawing

# Global variables for analytics
class VideoAnalytics:
    def __init__(self):
        self.face_detected = False
        self.face_position = (0, 0)
        self.looking_at_camera = False
        self.posture_centered = False
        self.head_pose = {"yaw": 0, "pitch": 0, "roll": 0}
        self.shoulder_angle = 0
        self.gaze_direction = "Unknown"
        self.face_bbox = None
        
    def calculate_head_pose(self, landmarks, image_shape):
        """Calculate head pose angles"""
        h, w = image_shape[:2]
        
        # Key facial landmarks for head pose estimation
        nose_tip = landmarks[1]
        chin = landmarks[175]
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Convert normalized coordinates to pixel coordinates
        nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))
        chin = (int(chin.x * w), int(chin.y * h))
        left_eye = (int(left_eye_corner.x * w), int(left_eye_corner.y * h))
        right_eye = (int(right_eye_corner.x * w), int(right_eye_corner.y * h))
        left_mouth = (int(left_mouth.x * w), int(left_mouth.y * h))
        right_mouth = (int(right_mouth.x * w), int(right_mouth.y * h))
        
        # Calculate yaw (left-right head turn)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        face_center_x = w / 2
        yaw = (eye_center_x - face_center_x) / (w / 2) * 45  # Normalize to degrees
        
        # Calculate pitch (up-down head tilt)
        nose_chin_dist = abs(nose_tip[1] - chin[1])
        pitch = (nose_tip[1] - h/2) / (h/2) * 30  # Normalize to degrees
        
        # Calculate roll (head tilt left-right)
        eye_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = math.degrees(eye_angle)
        
        self.head_pose = {"yaw": yaw, "pitch": pitch, "roll": roll}
        
        # Determine if looking at camera (within threshold)
        if abs(yaw) < 15 and abs(pitch) < 15:
            self.looking_at_camera = True
            self.gaze_direction = "Looking at camera"
        elif yaw > 15:
            self.looking_at_camera = False
            self.gaze_direction = "Looking right"
        elif yaw < -15:
            self.looking_at_camera = False
            self.gaze_direction = "Looking left"
        elif pitch > 15:
            self.looking_at_camera = False
            self.gaze_direction = "Looking down"
        elif pitch < -15:
            self.looking_at_camera = False
            self.gaze_direction = "Looking up"
        else:
            self.looking_at_camera = False
            self.gaze_direction = "Looking away"
    
    def calculate_posture(self, pose_landmarks, image_shape):
        """Calculate posture alignment"""
        if not pose_landmarks:
            return
            
        h, w = image_shape[:2]
        
        # Get shoulder landmarks
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        
        # Convert to pixel coordinates
        left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        
        # Calculate shoulder center
        shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) / 2
        frame_center_x = w / 2
        
        # Calculate shoulder angle
        shoulder_angle = math.atan2(
            right_shoulder_px[1] - left_shoulder_px[1],
            right_shoulder_px[0] - left_shoulder_px[0]
        )
        self.shoulder_angle = math.degrees(shoulder_angle)
        
        # Check if centered (within 10% of frame width)
        center_threshold = w * 0.1
        self.posture_centered = abs(shoulder_center_x - frame_center_x) < center_threshold

# Initialize analytics
analytics = VideoAnalytics()

# Video processor class
class VideoProcessor:
    def __init__(self):
        self.face_detection, self.face_mesh, self.pose, self.mp_drawing = load_mediapipe_models()
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_results = self.face_detection.process(rgb_frame)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        # Process face detection
        if face_results.detections:
            analytics.face_detected = True
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            
            # Calculate face position
            face_x = int((bbox.xmin + bbox.width/2) * w)
            face_y = int((bbox.ymin + bbox.height/2) * h)
            analytics.face_position = (face_x, face_y)
            
            # Draw face bounding box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            analytics.face_bbox = (x, y, width, height)
        else:
            analytics.face_detected = False
            analytics.face_bbox = None
        
        # Process face mesh for head pose
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                analytics.calculate_head_pose(face_landmarks.landmark, frame.shape)
        
        # Process pose for posture analysis
        if pose_results.pose_landmarks:
            analytics.calculate_posture(pose_results.pose_landmarks, frame.shape)
            
            # Draw pose landmarks (shoulders only for cleaner visualization)
            landmarks = pose_results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            
            cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)
            cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 2)
        
        return frame

# WebRTC callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame
    processor = VideoProcessor()
    processed_img = processor.process_frame(img)
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Streamlit UI
def main():
    st.title("🎥 Live Video Analytics Dashboard")
    st.markdown("Real-time face detection, gaze tracking, and posture analysis for interviews and presentations")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Analytics thresholds
        st.subheader("Detection Thresholds")
        gaze_threshold = st.slider("Gaze Detection Sensitivity", 5, 30, 15)
        posture_threshold = st.slider("Posture Centering Threshold (%)", 5, 20, 10)
        
        # Display options
        st.subheader("Display Options")
        show_landmarks = st.checkbox("Show Face Landmarks", value=True)
        show_pose = st.checkbox("Show Pose Points", value=True)
        
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        
        # WebRTC configuration for better connectivity
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Start video stream
        webrtc_ctx = webrtc_streamer(
            key="video-analytics",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📊 Real-time Analytics")
        
        # Create placeholders for real-time updates
        face_status = st.empty()
        gaze_status = st.empty()
        posture_status = st.empty()
        metrics_container = st.empty()
        
        # Real-time updates
        if webrtc_ctx.state.playing:
            while True:
                time.sleep(0.1)  # Update every 100ms
                
                with face_status.container():
                    if analytics.face_detected:
                        st.success("✅ Face Detected")
                        if analytics.face_bbox:
                            x, y, w, h = analytics.face_bbox
                            st.caption(f"Position: ({x + w//2}, {y + h//2})")
                    else:
                        st.error("❌ No Face Detected")
                
                with gaze_status.container():
                    if analytics.looking_at_camera:
                        st.success(f"👁️ {analytics.gaze_direction}")
                    else:
                        st.warning(f"👁️ {analytics.gaze_direction}")
                    
                    # Head pose details
                    if analytics.head_pose["yaw"] != 0:
                        st.caption(f"Yaw: {analytics.head_pose['yaw']:.1f}°")
                        st.caption(f"Pitch: {analytics.head_pose['pitch']:.1f}°")
                
                with posture_status.container():
                    if analytics.posture_centered:
                        st.success("🎯 Posture: Centered")
                    else:
                        st.warning("🎯 Posture: Off-center")
                    
                    if analytics.shoulder_angle != 0:
                        st.caption(f"Shoulder angle: {analytics.shoulder_angle:.1f}°")
                
                with metrics_container.container():
                    st.subheader("📈 Session Metrics")
                    
                    # Create metrics
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        face_score = 100 if analytics.face_detected else 0
                        st.metric("Face Detection", f"{face_score}%")
                        
                        gaze_score = 100 if analytics.looking_at_camera else 0
                        st.metric("Eye Contact", f"{gaze_score}%")
                    
                    with col_b:
                        posture_score = 100 if analytics.posture_centered else 0
                        st.metric("Posture", f"{posture_score}%")
                        
                        overall_score = (face_score + gaze_score + posture_score) / 3
                        st.metric("Overall Score", f"{overall_score:.1f}%")
                
                # Break if stream stops
                if not webrtc_ctx.state.playing:
                    break
    
    # Instructions
    with st.expander("📋 How to Use"):
        st.markdown("""
        ### Instructions:
        1. **Allow camera access** when prompted by your browser
        2. **Position yourself** in front of the camera at arm's length
        3. **Monitor the analytics** in real-time on the right panel
        
        ### Analytics Explained:
        - **Face Detection**: Shows if your face is clearly visible
        - **Eye Contact**: Indicates if you're looking directly at the camera
        - **Posture**: Checks if you're centered and aligned properly
        
        ### Tips for Best Results:
        - Ensure good lighting on your face
        - Keep your head and shoulders in frame
        - Maintain steady positioning
        - Look directly at the camera lens, not the screen
        """)
    
    # Technical info
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Technology Stack:
        - **Computer Vision**: MediaPipe for face detection and pose estimation
        - **Real-time Processing**: WebRTC for low-latency video streaming
        - **Analytics**: Custom algorithms for gaze and posture analysis
        
        ### Performance Optimizations:
        - Optimized MediaPipe model settings for speed
        - Efficient frame processing pipeline
        - Minimal UI updates to reduce latency
        """)

if __name__ == "__main__":
    main()
