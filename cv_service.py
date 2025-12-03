import cv2
import numpy as np
import tempfile
import os

def analyze_pit_stop_video(video_file):
    """
    Computer Vision Pit Stop Timer
    Uses Motion Detection (Frame Differencing) to calculate stationary time.
    """
    # 1. Upload and Read Video
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_frame = None
    stationary_frames = 0
    is_stopped = False
    
    # Metrics
    pit_stop_duration = 0.0
    
    print("--- Starting Computer Vision Analysis ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Preprocessing (Grayscale + Blur)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # 3. Compute Difference (Motion Detection)
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Count white pixels (motion magnitude)
        motion_score = np.sum(thresh)
        
        # 4. Pit Stop Logic
        # Threshold: If motion < 5000 pixels, assume car is stopped
        if motion_score < 5000:
            stationary_frames += 1
            is_stopped = True
        else:
            if is_stopped and stationary_frames > (fps * 1.5): 
                pit_stop_duration = stationary_frames / fps
                break 
            
            stationary_frames = 0
            is_stopped = False
            
        prev_frame = gray

    cap.release()
    os.remove(video_path) 
    
    return pit_stop_duration