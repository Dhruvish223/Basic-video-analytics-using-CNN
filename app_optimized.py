import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time
import threading
from collections import deque

# Configure Streamlit page
st.set_page_config(
    page_title="Live Webcam Analytics - Optimized",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for optimization
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=5)
if 'analytics_cache' not in st.session_state:
    st.session_state.analytics_cache = {}
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False

# Optimized MediaPipe initialization
@st.cache_resource
def init_lightweight_models():
    """Initialize lightweight MediaPipe models for better performance"""
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Use lightweight models for better performance
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for close-range, 1 for full-range
        min_detection_confidence=0.6
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # 0 for lightweight, 2 for heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    
    return face_detection, pose, mp_drawing

def simple_gaze_estimation(face_landmarks, img_w, img_h):
    """Simplified gaze estimation for better performance"""
    try:
        # Use fewer landmarks for faster processing
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        left_eye = face_landmarks.landmark[33]  # Left eye corner
        right_eye = face_landmarks.landmark[263]  # Right eye corner
        
        # Calculate face center
        face_center_x = (left_eye.x + right_eye.x) / 2
        face_center_y = (left_eye.y + right_eye.y) / 2
        
        # Simple heuristic: if nose is roughly centered between eyes
        nose_center_offset = abs(nose_tip.x - face_center_x)
        looking_forward = nose_center_offset < 0.03
        
        return looking_forward, face_center_x, face_center_y
        
    except:
        return False, 0.5, 0.5

def quick_posture_check(pose_landmarks):
    """Quick posture analysis with minimal calculations"""
    try:
        landmarks = pose_landmarks.landmark
        
        # Key points
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
        
        # Quick checks
        shoulder_level = abs(left_shoulder.y - right_shoulder.y) < 0.08
        shoulder_center = (left_shoulder.x + right_shoulder.x) / 2
        center_aligned = abs(shoulder_center - 0.5) < 0.15
        head_straight = abs(nose.x - shoulder_center) < 0.12
        
        return {
            'good_posture': shoulder_level and center_aligned and head_straight,
            'shoulder_level': shoulder_level,
            'center_aligned': center_aligned,
            'head_straight': head_straight,
            'alignment_score': max(0, 1 - abs(shoulder_center - 0.5) * 3)
        }
    except:
        return {
            'good_posture': False,
            'shoulder_level': False,
            'center_aligned': False,
            'head_straight': False,
            'alignment_score': 0.5
        }

def process_frame_lightweight(frame, face_detection, pose, mp_drawing):
    """Lightweight frame processing for real-time performance"""
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 240))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    h, w = frame.shape[:2]
    small_h, small_w = small_frame.shape[:2]
    
    analytics = {
        'timestamp': time.time(),
        'face_detected': False,
        'face_confidence': 0,
        'face_center': {'x': 0.5, 'y': 0.5},
        'looking_at_camera': False,
        'posture_good': False,
        'posture_score': 0.5,
        'recommendations': []
    }
    
    # Process with small frame
    face_results = face_detection.process(rgb_small)
    
    if face_results.detections:
        detection = face_results.detections[0]  # Use first detection only
        bbox = detection.location_data.relative_bounding_box
        confidence = detection.score[0]
        
        analytics['face_detected'] = True
        analytics['face_confidence'] = confidence
        analytics['face_center'] = {
            'x': bbox.xmin + bbox.width/2,
            'y': bbox.ymin + bbox.height/2
        }
        
        # Scale up bbox for display
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Draw on original frame
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f'Face ({confidence:.2f})', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Simple gaze check
        center_x, center_y = analytics['face_center']['x'], analytics['face_center']['y']
        looking_at_camera = (0.3 < center_x < 0.7) and (0.3 < center_y < 0.7)
        analytics['looking_at_camera'] = looking_at_camera
        
        gaze_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
        gaze_text = "👁️ Looking" if looking_at_camera else "👁️ Look Here"
        cv2.putText(frame, gaze_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
    
    # Quick pose check
    pose_results = pose.process(rgb_small)
    if pose_results.pose_landmarks:
        posture = quick_posture_check(pose_results.pose_landmarks)
        analytics['posture_good'] = posture['good_posture']
        analytics['posture_score'] = posture['alignment_score']
        
        # Draw minimal pose indicators
        posture_color = (0, 255, 0) if posture['good_posture'] else (255, 165, 0)
        posture_text = "✅ Good Posture" if posture['good_posture'] else "⚠️ Adjust Posture"
        cv2.putText(frame, posture_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
        
        # Add recommendations
        if not posture['center_aligned']:
            analytics['recommendations'].append("Move to center")
            cv2.putText(frame, "← → Center yourself", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        if not posture['shoulder_level']:
            analytics['recommendations'].append("Level shoulders")
            cv2.putText(frame, "📐 Level shoulders", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    return frame, analytics

def main():
    st.title("⚡ Optimized Webcam Analytics")
    st.markdown("*Low-latency real-time analysis for better performance*")
    
    # Initialize models
    face_detection, pose, mp_drawing = init_lightweight_models()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    enable_analytics = st.sidebar.checkbox("Enable Real-time Analytics", value=True)
    show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)
    processing_frequency = st.sidebar.slider("Processing Speed", 1, 10, 5, 
                                           help="Higher = faster processing, more CPU usage")
    
    # Status indicators
    st.sidebar.markdown("### 📊 System Status")
    status_placeholder = st.sidebar.empty()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        
        # Camera input with key for refresh
        camera_input = st.camera_input("📸 Capture Image for Analysis", key="webcam_feed")
        
        if camera_input is not None and enable_analytics:
            # Process image
            start_time = time.time()
            
            image = Image.open(camera_input)
            frame = np.array(image)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame
            processed_frame, analytics = process_frame_lightweight(
                frame_bgr, face_detection, pose, mp_drawing
            )
            
            processing_time = time.time() - start_time
            
            # Display processed frame
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(processed_frame_rgb, caption="Analyzed Feed", use_column_width=True)
            
            # Update status
            with status_placeholder.container():
                st.success(f"✅ Active - {processing_time:.2f}s")
                st.metric("Processing Time", f"{processing_time:.3f}s")
                if analytics['face_detected']:
                    st.metric("Face Confidence", f"{analytics['face_confidence']:.2f}")
            
            # Real-time analytics panel
            with col2:
                st.subheader("📊 Live Analytics")
                
                # Quick status cards
                if analytics['face_detected']:
                    st.success("👤 Face Detected")
                else:
                    st.error("❌ No Face Found")
                
                if analytics['looking_at_camera']:
                    st.success("👁️ Good Eye Contact")
                else:
                    st.warning("👁️ Look at Camera")
                
                if analytics['posture_good']:
                    st.success("🧍 Good Posture")
                else:
                    st.warning("🧍 Adjust Posture")
                
                # Detailed metrics
                if show_detailed_metrics:
                    st.markdown("### 📈 Metrics")
                    
                    # Posture score
                    posture_score = analytics.get('posture_score', 0.5)
                    st.progress(posture_score, text=f"Alignment Score: {posture_score:.0%}")
                    
                    # Face position
                    if analytics['face_detected']:
                        face_pos = analytics['face_center']
                        st.write(f"**Face Position:** X: {face_pos['x']:.2f}, Y: {face_pos['y']:.2f}")
                    
                    # Recommendations
                    if analytics.get('recommendations'):
                        st.markdown("### 💡 Suggestions")
                        for rec in analytics['recommendations']:
                            st.write(f"• {rec}")
                
                # Performance info
                st.markdown("### ⚡ Performance")
                fps_estimate = 1 / max(processing_time, 0.001)
                st.metric("Est. FPS", f"{fps_estimate:.1f}")
                
    else:
        # Show instructions when camera is off
        with col1:
            st.info("📷 Click the camera button above to start analysis")
            
        with status_placeholder.container():
            st.info("💤 Standby")
    
    # Tips and instructions
    st.markdown("---")
    with st.expander("📋 Usage Tips", expanded=False):
        st.markdown("""
        **For Best Performance:**
        - Ensure good lighting on your face
        - Sit about 2-3 feet from the camera
        - Keep your head and shoulders in frame
        - Look directly at the camera for gaze detection
        
        **Posture Guidelines:**
        - Keep shoulders level and relaxed
        - Center yourself in the frame
        - Maintain straight head alignment
        - Sit up straight with good back support
        
        **Troubleshooting:**
        - If analysis is slow, reduce processing speed in sidebar
        - Refresh the page if camera doesn't work
        - Ensure browser has camera permissions
        """)

if __name__ == "__main__":
    main()
