import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Live Webcam Analytics",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe_models():
    """Load MediaPipe models with caching for better performance"""
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_detection, face_mesh, pose, mp_drawing, mp_drawing_styles

def calculate_gaze_direction(landmarks, img_w, img_h):
    """Calculate if user is looking at camera based on eye landmarks"""
    try:
        # Key landmarks for gaze detection
        left_eye_left = landmarks[33]
        left_eye_right = landmarks[133]
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        
        right_eye_left = landmarks[362]
        right_eye_right = landmarks[263]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        # Calculate eye centers
        left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
        left_eye_center_y = (left_eye_top.y + left_eye_bottom.y) / 2
        
        right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2
        right_eye_center_y = (right_eye_top.y + right_eye_bottom.y) / 2
        
        # Calculate relative positions of pupils within eyes
        eye_center_avg_x = (left_eye_center_x + right_eye_center_x) / 2
        eye_center_avg_y = (left_eye_center_y + right_eye_center_y) / 2
        
        # Check if looking at camera (eyes relatively centered)
        gaze_threshold = 0.02
        looking_at_camera = (
            abs(eye_center_avg_x - 0.5) < gaze_threshold and
            abs(eye_center_avg_y - 0.5) < gaze_threshold
        )
        
        return looking_at_camera, eye_center_avg_x, eye_center_avg_y
        
    except Exception:
        return False, 0.5, 0.5

def analyze_posture(pose_landmarks, img_w, img_h):
    """Analyze user posture alignment"""
    try:
        # Key landmarks for posture analysis
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        
        # Calculate shoulder alignment
        shoulder_diff_y = abs(left_shoulder.y - right_shoulder.y)
        shoulder_tilt = shoulder_diff_y < 0.05  # Threshold for level shoulders
        
        # Calculate center alignment
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_aligned = abs(shoulder_center_x - 0.5) < 0.1  # Threshold for center alignment
        
        # Calculate head position relative to shoulders
        head_aligned = abs(nose.x - shoulder_center_x) < 0.08
        
        overall_posture_good = shoulder_tilt and center_aligned and head_aligned
        
        return {
            'shoulder_level': shoulder_tilt,
            'center_aligned': center_aligned,
            'head_aligned': head_aligned,
            'overall_good': overall_posture_good,
            'shoulder_center_x': shoulder_center_x,
            'head_x': nose.x
        }
        
    except Exception:
        return {
            'shoulder_level': False,
            'center_aligned': False,
            'head_aligned': False,
            'overall_good': False,
            'shoulder_center_x': 0.5,
            'head_x': 0.5
        }

def process_frame(frame, face_detection, face_mesh, pose, mp_drawing, mp_drawing_styles):
    """Process a single frame and return analytics"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    # Face detection
    face_results = face_detection.process(rgb_frame)
    face_mesh_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    analytics = {
        'face_detected': False,
        'face_position': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
        'looking_at_camera': False,
        'gaze_position': {'x': 0.5, 'y': 0.5},
        'posture': {
            'shoulder_level': False,
            'center_aligned': False,
            'head_aligned': False,
            'overall_good': False,
            'shoulder_center_x': 0.5,
            'head_x': 0.5
        }
    }
    
    # Process face detection
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            analytics['face_detected'] = True
            analytics['face_position'] = {
                'x': bbox.xmin,
                'y': bbox.ymin,
                'width': bbox.width,
                'height': bbox.height
            }
            
            # Draw face bounding box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Process gaze detection
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            looking_at_camera, gaze_x, gaze_y = calculate_gaze_direction(
                face_landmarks.landmark, w, h
            )
            analytics['looking_at_camera'] = looking_at_camera
            analytics['gaze_position'] = {'x': gaze_x, 'y': gaze_y}
            
            # Draw gaze indicator
            gaze_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            gaze_text = "Looking at Camera" if looking_at_camera else "Not Looking at Camera"
            cv2.putText(frame, gaze_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
    
    # Process posture analysis
    if pose_results.pose_landmarks:
        posture_analysis = analyze_posture(pose_results.pose_landmarks, w, h)
        analytics['posture'] = posture_analysis
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw posture status
        posture_color = (0, 255, 0) if posture_analysis['overall_good'] else (0, 165, 255)
        posture_text = "Good Posture" if posture_analysis['overall_good'] else "Poor Posture"
        cv2.putText(frame, posture_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        
        # Draw alignment indicators
        if not posture_analysis['center_aligned']:
            cv2.putText(frame, "Move to Center", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if not posture_analysis['shoulder_level']:
            cv2.putText(frame, "Level Shoulders", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return frame, analytics

def main():
    st.title("🎥 Live Webcam Analytics Dashboard")
    st.markdown("Real-time face detection, gaze tracking, and posture analysis")
    
    # Load MediaPipe models
    face_detection, face_mesh, pose, mp_drawing, mp_drawing_styles = load_mediapipe_models()
    
    # Sidebar controls
    st.sidebar.header("📊 Controls")
    enable_recording = st.sidebar.checkbox("Enable Analytics", value=True)
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        
        # Webcam input
        camera_input = st.camera_input("Take a picture or start video", key="camera")
        
        if camera_input is not None and enable_recording:
            # Convert uploaded image to OpenCV format
            image = Image.open(camera_input)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame
            processed_frame, analytics = process_frame(
                frame_bgr, face_detection, face_mesh, pose, mp_drawing, mp_drawing_styles
            )
            
            # Convert back to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(processed_frame_rgb, caption="Processed Video Feed", use_column_width=True)
            
            # Display analytics in sidebar
            with col2:
                st.subheader("📊 Real-time Analytics")
                
                # Face Detection Status
                st.markdown("### 👤 Face Detection")
                if analytics['face_detected']:
                    st.success("✅ Face Detected")
                    face_pos = analytics['face_position']
                    st.write(f"**Position:** X: {face_pos['x']:.2f}, Y: {face_pos['y']:.2f}")
                    st.write(f"**Size:** W: {face_pos['width']:.2f}, H: {face_pos['height']:.2f}")
                else:
                    st.error("❌ No Face Detected")
                
                # Gaze Detection Status
                st.markdown("### 👁️ Gaze Detection")
                if analytics['looking_at_camera']:
                    st.success("✅ Looking at Camera")
                else:
                    st.warning("⚠️ Not Looking at Camera")
                
                gaze_pos = analytics['gaze_position']
                st.write(f"**Gaze Position:** X: {gaze_pos['x']:.2f}, Y: {gaze_pos['y']:.2f}")
                
                # Posture Analysis
                st.markdown("### 🧍 Posture Analysis")
                posture = analytics['posture']
                
                if posture['overall_good']:
                    st.success("✅ Good Posture")
                else:
                    st.warning("⚠️ Poor Posture")
                
                # Detailed posture breakdown
                st.write("**Alignment Details:**")
                st.write(f"• Shoulders Level: {'✅' if posture['shoulder_level'] else '❌'}")
                st.write(f"• Center Aligned: {'✅' if posture['center_aligned'] else '❌'}")
                st.write(f"• Head Aligned: {'✅' if posture['head_aligned'] else '❌'}")
                
                # Progress bars for alignment
                st.markdown("### 📈 Alignment Metrics")
                
                center_score = 1.0 - abs(posture.get('shoulder_center_x', 0.5) - 0.5) * 2
                st.progress(max(0, min(1, center_score)), text="Center Alignment")
                
                head_alignment_score = 1.0 - abs(posture.get('head_x', 0.5) - posture.get('shoulder_center_x', 0.5)) * 10
                st.progress(max(0, min(1, head_alignment_score)), text="Head Alignment")
        
        else:
            with col2:
                st.info("📷 Enable analytics and take a photo to start analysis")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions:
    1. **Click the camera button** to capture your image
    2. **Enable Analytics** in the sidebar to start real-time analysis
    3. **Position yourself** so your face is clearly visible
    4. **Look directly at the camera** for optimal gaze detection
    5. **Sit up straight** and center yourself for good posture analysis
    
    ### 📊 Analytics Explained:
    - **Face Detection**: Identifies and tracks your face position
    - **Gaze Detection**: Determines if you're looking directly at the camera
    - **Posture Analysis**: Evaluates your sitting position and alignment
    """)

if __name__ == "__main__":
    main()
