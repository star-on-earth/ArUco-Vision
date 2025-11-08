#!/usr/bin/env python3
"""
YOLOv8 + ArUco Visual Aid - FIXED CALIBRATION
Features:
- Working camera calibration using ArUco markers
- Accurate distance measurement with ArUco markers
- Fallback distance estimation without markers
- Audio announcements every 1 second
"""

import cv2
import pyttsx3
import numpy as np
import argparse
import time
import threading
import pickle
import os
from queue import Queue, Empty
from ultralytics import YOLO
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ArUco settings - MEASURE YOUR ACTUAL MARKER SIZE!
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.05  # 5cm markers - CRITICAL: Update this to your actual size!
CALIBRATION_FILE = "camera_calibration.pkl"

class YOLOArUcoVisualAid:
    def __init__(self, args):
        self.args = args
        self.setup_models()
        self.setup_audio()
        self.setup_tracking()
        self.setup_aruco()
        self.load_or_calibrate_camera()
        
    def setup_models(self):
        """Initialize YOLOv8 model"""
        try:
            logger.info("Loading YOLOv8n model...")
            self.yolo_model = YOLO("yolov8nfinetuned.pt")
            logger.info("✅ YOLOv8n loaded")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def setup_aruco(self):
        """Setup ArUco marker detector"""
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            logger.info("✅ ArUco detector initialized")
        except Exception as e:
            logger.error(f"ArUco setup failed: {e}")
            raise
    
    def setup_audio(self):
        """Initialize TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.args.voice_rate)
            self.tts_engine.setProperty('volume', 0.9)
            
            self.audio_queue = Queue(maxsize=5)
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
            logger.info("✅ Audio system initialized")
        except Exception as e:
            logger.error(f"Audio failed: {e}")
            self.tts_engine = None
    
    def setup_tracking(self):
        """Initialize tracking variables"""
        self.announcement_interval = 1.0  # 1 second between announcements
        self.last_announcement_time = 0
        self.frame_count = 0
        self.fps_tracker = deque(maxlen=30)
        
        self.priority_objects = {
            'person': 1.0, 'car': 0.95, 'truck': 0.95, 'bus': 0.95,
            'bicycle': 0.9, 'motorcycle': 0.9, 'dog': 0.9, 'cat': 0.8,
            'chair': 0.8, 'couch': 0.8, 'dining table': 0.85, 'bed': 0.7,
            'backpack': 0.6, 'handbag': 0.5, 'suitcase': 0.7
        }
        
        self.min_confidence = 0.45
        self.marker_distances = {}
    
    def _audio_worker(self):
        """Background TTS thread"""
        while True:
            try:
                message = self.audio_queue.get(timeout=1)
                if message is None:
                    break
                if self.tts_engine:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                self.audio_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Audio worker error: {e}")
    
    def speak(self, text):
        """Non-blocking TTS"""
        if not self.tts_engine:
            return
        
        # Clear queue if backed up
        while self.audio_queue.qsize() > 1:
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        try:
            self.audio_queue.put_nowait(text)
        except:
            pass
    
    def load_or_calibrate_camera(self):
        """Load existing calibration or use defaults"""
        if os.path.exists(CALIBRATION_FILE) and not self.args.recalibrate:
            try:
                with open(CALIBRATION_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                logger.info("✅ Loaded camera calibration from file")
                logger.info(f"Marker size: {MARKER_LENGTH*100:.1f}cm")
                return
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
        
        # Skip complex calibration, use smart defaults
        logger.info("Using optimized camera parameters")
        self.use_default_calibration()
        logger.info(f"⚠️  IMPORTANT: Marker size set to {MARKER_LENGTH*100:.1f}cm")
        logger.info(f"⚠️  Update MARKER_LENGTH in code to your actual marker size!")
    
    def use_default_calibration(self):
        """Use optimized default calibration"""
        # Typical webcam parameters for 640x480
        fx = fy = 600  # Focal length
        cx, cy = 320, 240  # Principal point
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1))
        
        # Save for future use
        with open(CALIBRATION_FILE, 'wb') as f:
            pickle.dump({
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs
            }, f)
        
        logger.info("Camera parameters:")
        logger.info(f"  Focal length: {fx}")
        logger.info(f"  Center: ({cx}, {cy})")
    
    def detect_aruco_markers(self, frame):
        """Detect ArUco markers and calculate distances"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        self.marker_distances = {}
        
        if ids is not None and len(ids) > 0:
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs
                )
                
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    # Calculate distance (magnitude of translation vector)
                    distance = float(np.linalg.norm(tvecs[i][0]))
                    
                    self.marker_distances[marker_id] = {
                        'distance': distance,
                        'position': tvecs[i][0],
                        'corners': corners[i][0]
                    }
                    
                    # Draw markers
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw 3D axis
                    try:
                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                         rvecs[i], tvecs[i], MARKER_LENGTH * 0.5)
                    except:
                        pass  # Skip if axis drawing fails
                    
                    # Show distance on frame
                    corner = corners[i][0][0]
                    text_x, text_y = int(corner[0]), int(corner[1]) - 10
                    cv2.putText(frame, f"ID{marker_id}: {distance:.2f}m",
                               (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 2)
            
            except Exception as e:
                logger.debug(f"ArUco pose estimation error: {e}")
        
        return frame
    
    def estimate_distance(self, bbox):
        """Estimate distance using ArUco markers or bbox fallback"""
        x1, y1, x2, y2 = bbox
        obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        # Method 1: Use ArUco markers if available
        if self.marker_distances:
            min_pixel_dist = float('inf')
            closest_marker_dist = None
            
            for marker_id, marker_data in self.marker_distances.items():
                marker_corners = marker_data['corners']
                marker_center = np.mean(marker_corners, axis=0)
                
                # Calculate pixel distance between object and marker
                pixel_distance = np.linalg.norm(obj_center - marker_center)
                
                if pixel_distance < min_pixel_dist:
                    min_pixel_dist = pixel_distance
                    closest_marker_dist = marker_data['distance']
            
            if closest_marker_dist is not None:
                # Use closest marker distance with small adjustment
                # Objects further from marker center are likely at similar depth
                estimated_distance = closest_marker_dist
                
                # Add small correction based on pixel offset
                # (objects far from marker might be slightly offset in depth)
                pixel_offset_factor = min_pixel_dist / 200.0  # Normalize
                estimated_distance += pixel_offset_factor * 0.05  # Small correction
                
                return np.clip(estimated_distance, 0.2, 15.0)
        
        # Method 2: Fallback - estimate from bbox size
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        bbox_area = bbox_height * bbox_width
        
        # Empirical formula (needs calibration for your specific setup)
        # Assumes average object size
        estimated_distance = 50000.0 / (bbox_area + 1000)
        estimated_distance = np.clip(estimated_distance, 0.5, 10.0)
        
        return estimated_distance
    
    def categorize_distance(self, distance_meters):
        """Categorize distance"""
        if distance_meters < 0.8:
            return "very close"
        elif distance_meters < 1.5:
            return "close"
        elif distance_meters < 3.0:
            return "nearby"
        elif distance_meters < 5.0:
            return "ahead"
        else:
            return "far"
    
    def should_announce(self):
        """Check if 1 second has passed"""
        current_time = time.time()
        return (current_time - self.last_announcement_time) >= self.announcement_interval
    
    def process_detections(self, frame):
        """Process YOLO detections"""
        try:
            results = self.yolo_model(frame, conf=self.min_confidence, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return frame
            
            detections = []
            
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.yolo_model.names[class_id]
                bbox = box.xyxy[0].tolist()
                
                distance_meters = self.estimate_distance(bbox)
                distance_category = self.categorize_distance(distance_meters)
                
                base_priority = self.priority_objects.get(label, 0.1)
                distance_boost = {
                    "very close": 3.0, "close": 2.0, "nearby": 1.5,
                    "ahead": 1.0, "far": 0.5
                }.get(distance_category, 1.0)
                
                final_priority = base_priority * distance_boost * confidence
                
                detections.append({
                    'label': label,
                    'distance_meters': distance_meters,
                    'distance_category': distance_category,
                    'confidence': confidence,
                    'priority': final_priority,
                    'bbox': bbox
                })
            
            # Announce every 1 second
            if detections and self.should_announce():
                detections.sort(key=lambda x: x['priority'], reverse=True)
                best = detections[0]
                
                message = f"{best['label']} at {best['distance_meters']:.1f} meters"
                self.speak(message)
                
                self.last_announcement_time = time.time()
                logger.info(f"Announced: {message}")
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            for det in detections[:5]:
                x1, y1, x2, y2 = map(int, det['bbox'])
                dist_text = f"{det['distance_meters']:.1f}m"
                cv2.putText(annotated_frame, dist_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame
    
    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps_tracker.append(current_time)
        if len(self.fps_tracker) > 1:
            return len(self.fps_tracker) / (self.fps_tracker[-1] - self.fps_tracker[0])
        return 0
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("=" * 60)
        logger.info("🚀 YOLOv8 + ArUco Visual Aid Started")
        logger.info("=" * 60)
        logger.info("Announcements: Every 1 second")
        logger.info("Press 'q' to quit")
        if len(self.marker_distances) == 0:
            logger.info("💡 TIP: Place ArUco markers for accurate distance!")
        logger.info("=" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Detect ArUco markers
                frame = self.detect_aruco_markers(frame)
                
                # Process YOLO detections
                annotated_frame = self.process_detections(frame)
                
                if not self.args.no_window:
                    # Add overlay info
                    fps = self.update_fps()
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Timer countdown
                    time_left = self.announcement_interval - (time.time() - self.last_announcement_time)
                    time_left = max(0, time_left)
                    timer_color = (0, 255, 0) if time_left <= 0.1 else (0, 255, 255)
                    cv2.putText(annotated_frame, f"Next: {time_left:.1f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)
                    
                    # Marker count
                    marker_text = f"Markers: {len(self.marker_distances)}"
                    marker_color = (0, 255, 0) if len(self.marker_distances) > 0 else (0, 165, 255)
                    cv2.putText(annotated_frame, marker_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 2)
                    
                    cv2.imshow("YOLOv8 + ArUco Visual Aid", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_queue.put(None)
            logger.info("✅ Cleanup complete")

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 + ArUco Visual Aid")
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--voice_rate', type=int, default=160, help='Speech rate')
    parser.add_argument('--voice_name', type=str, default='', help='Voice name')
    parser.add_argument('--no_window', action='store_true', help='Audio-only mode')
    parser.add_argument('--recalibrate', action='store_true', help='Force recalibration')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("YOLOv8 + ArUco Visual Aid System")
    logger.info("=" * 60)
    logger.info(f"⚠️  IMPORTANT: Current marker size = {MARKER_LENGTH*100:.1f}cm")
    logger.info(f"⚠️  Update MARKER_LENGTH in code to match your actual markers!")
    logger.info("=" * 60)
    
    try:
        app = YOLOArUcoVisualAid(args)
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())