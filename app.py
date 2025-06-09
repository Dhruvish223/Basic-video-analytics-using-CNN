# app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.ANALYSIS_RESULTS = {
            "face_detected": False,
            "face_position": (0, 0),
            "looking_at_camera": False,
            "posture_aligned": False
        }
    
    def get_face_metrics(self, face_landmarks, frame_width, frame_height):
        # Extract key landmarks
        nose_tip = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        left_iris = face_landmarks.landmark[468]
        right_iris = face_landmarks.landmark[473]

        # Calculate face position
        face_x = int(nose_tip.x * frame_width)
        face_y = int(nose_tip.y * frame_height)
        
        # Calculate eye direction
        left_eye_center = ((left_eye.x + left_iris.x)/2, (left_eye.y + left_iris.y)/2)
        right_eye_center = ((right_eye.x + right_iris.x)/2, (right_eye.y + right_iris.y)/2)
        
        # Determine if looking at camera
        left_dist = abs(left_eye_center[0] - left_iris.x)
        right_dist = abs(right_eye_center[0] - right_iris.x)
        looking = (left_dist < 0.01) and (right_dist < 0.01)
        
        return (face_x, face_y), looking
    
    def check_posture_alignment(self, pose_landmarks, frame_width, frame_height):
        if not pose_landmarks:
            return False
            
        # Get relevant landmarks
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
        # Calculate shoulder midpoint
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Check alignment
        x_diff = abs(shoulder_mid_x - nose.x) * frame_width
        y_diff = abs(shoulder_mid_y - nose.y) * frame_height
        
        # Thresholds for alignment (adjust as needed)
        return (x_diff < 0.1 * frame_width) and (y_diff < 0.1 * frame_height)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reset analysis results
        self.ANALYSIS_RESULTS = {
            "face_detected": False,
            "face_position": (0, 0),
            "looking_at_camera": False,
            "posture_aligned": False
        }
        
        # Process face and pose
        face_results = self.face_mesh.process(img_rgb)
        pose_results = self.pose.process(img_rgb)
        
        img.flags.writeable = True
        h, w, _ = img.shape
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Check posture alignment
            self.ANALYSIS_RESULTS["posture_aligned"] = self.check_posture_alignment(
                pose_results.pose_landmarks, w, h)
        
        # Process face results
        if face_results.multi_face_landmarks:
            self.ANALYSIS_RESULTS["face_detected"] = True
            
            for face_landmarks in face_results.multi_face_landmarks:
                # Get face metrics
                position, looking = self.get_face_metrics(face_landmarks, w, h)
                self.ANALYSIS_RESULTS["face_position"] = position
                self.ANALYSIS_RESULTS["looking_at_camera"] = looking
                
                # Draw face landmarks
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                # Draw iris landmarks
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        
        # Draw center alignment guides
        cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(img, (0, h//2), (w, h//2), (0, 255, 0), 1)
        cv2.rectangle(img, (w//3, h//3), (2*w//3, 2*h//3), (0, 255, 0), 1)
        
        # Display analysis results on frame
        cv2.putText(img, f"Face Position: {self.ANALYSIS_RESULTS['face_position']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        eye_status = "Looking at camera" if self.ANALYSIS_RESULTS["looking_at_camera"] else "Not looking"
        cv2.putText(img, f"Eye Contact: {eye_status}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.ANALYSIS_RESULTS["looking_at_camera"] else (0, 0, 255), 2)
        
        posture_status = "Good Posture" if self.ANALYSIS_RESULTS["posture_aligned"] else "Adjust Posture"
        cv2.putText(img, f"Posture: {posture_status}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.ANALYSIS_RESULTS["posture_aligned"] else (0, 0, 255), 2)
        
        return img

def main():
    st.set_page_config(page_title="Real-Time Webcam Analytics", layout="wide")
    st.title("Real-Time Webcam Analytics Dashboard")
    st.markdown("""
    This dashboard provides real-time analysis of:
    - 👤 Face detection and position
    - 👀 Eye contact detection (whether you're looking at the camera)
    - 🧍 Posture alignment (centered position)
    """)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Webcam Feed")
        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            async_transform=True,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_transformer:
            st.session_state.analytics = ctx.video_transformer.ANALYSIS_RESULTS
    
    with col2:
        st.header("Real-time Analytics")
        placeholder = st.empty()
        
        if 'analytics' in st.session_state:
            analytics = st.session_state.analytics
            
            if analytics['face_detected']:
                st.success("✅ Face Detected")
                st.metric("Face Position", str(analytics['face_position']))
                
                if analytics['looking_at_camera']:
                    st.success("👀 Looking at Camera")
                else:
                    st.error("👁️ Not Looking at Camera")
                
                if analytics['posture_aligned']:
                    st.success("🧍 Good Posture")
                else:
                    st.error("⚠️ Adjust Posture")
            else:
                st.warning("🔍 No Face Detected")
                
        st.markdown("---")
        st.subheader("Analysis Guide")
        st.markdown("""
        - **Face Position**: Shows coordinates of your face center
        - **Eye Contact**: Green when looking directly at camera
        - **Posture**: Green when shoulders and head are aligned
        - **Center Lines**: Helps you position yourself in frame
        """)
        st.info("For best results: Ensure good lighting and position your face within the center rectangle")

if __name__ == "__main__":
    main()
