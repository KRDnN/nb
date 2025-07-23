import cv2
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from threading import Lock
from config import config
from utils import logger
from .orb_service import ORBService

class RTSPMonitor:
    """RTSP streaming and danger zone monitoring service"""
    
    def __init__(self):
        self.rtsp_url: str = ""
        self.current_project: Optional[str] = None
        self.orb_service = ORBService()
        self.danger_zones: Dict[int, List[Tuple[float, float]]] = {}
        self.current_best_idx: int = -1
        self.current_best_score: int = 0
        self._lock = Lock()
        
    def set_rtsp_url(self, url: str) -> bool:
        """Set RTSP stream URL"""
        try:
            with self._lock:
                self.rtsp_url = url
            logger.info(f"RTSP URL set: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to set RTSP URL: {e}")
            return False
    
    def set_project(self, project_name: str) -> bool:
        """Set current monitoring project"""
        try:
            with self._lock:
                self.current_project = project_name
                self._load_project_data(project_name)
                self.current_best_idx = -1
                self.current_best_score = 0
            
            logger.info(f"Project set: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set project: {e}")
            return False
    
    def _load_project_data(self, project_name: str):
        """Load project danger zones and ORB data"""
        try:
            # Load danger zones from meta.json
            meta_path = f"{config.PROJECT_FOLDER}/{project_name}/meta.json"
            self.danger_zones.clear()
            
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                
                for zone in meta_data.get('zones', []):
                    frame_idx = zone['frame']
                    points = zone['points']
                    self.danger_zones[frame_idx] = points
                    
                logger.info(f"Loaded {len(self.danger_zones)} danger zones")
                
            except FileNotFoundError:
                logger.warning(f"Meta file not found: {meta_path}")
            
            # Preload ORB features into cache
            self.orb_service.get_cached_features(project_name)
            
        except Exception as e:
            logger.error(f"Failed to load project data: {e}")
    
    def get_stream_generator(self) -> Generator[bytes, None, None]:
        """Generate RTSP stream with danger zone overlay"""
        if not self.rtsp_url:
            logger.error("No RTSP URL set")
            return
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                return
            
            logger.info(f"RTSP stream started: {self.rtsp_url}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from RTSP stream")
                    break
                
                # Process frame for danger zone detection
                processed_frame = self._process_frame(frame)
                
                # Encode frame for streaming
                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
                
        except Exception as e:
            logger.error(f"RTSP stream error: {e}")
        finally:
            if cap:
                cap.release()
                logger.info("RTSP stream stopped")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for danger zone detection and overlay"""
        try:
            with self._lock:
                current_project = self.current_project
                
            if not current_project:
                return frame
            
            # Extract ORB features from current frame
            _, live_descriptors = self.orb_service.extract_frame_features(frame)
            
            # Get cached project features
            project_features = self.orb_service.get_cached_features(current_project)
            
            if not project_features:
                return frame
            
            # Find best matching frame
            best_idx, best_score = self._find_best_match(live_descriptors, project_features)
            
            # Update tracking state
            with self._lock:
                self.current_best_idx = best_idx
                self.current_best_score = best_score
            
            # Draw danger zone overlay if match is good enough
            if best_score >= config.BEST_SCORE_THRESHOLD and best_idx in self.danger_zones:
                frame = self._draw_danger_zone(frame, best_idx, best_score)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame
    
    def _find_best_match(self, live_descriptors: np.ndarray, project_features: Dict) -> Tuple[int, int]:
        """Find best matching frame with optimized search"""
        try:
            # Quick check with current best match
            if (self.current_best_idx != -1 and 
                self.current_best_idx in project_features and
                live_descriptors is not None):
                
                saved_descriptors = project_features[self.current_best_idx]['descriptors']
                quick_score = self.orb_service._calculate_match_score(live_descriptors, saved_descriptors)
                
                # If score is still good enough, keep current match
                if quick_score >= int(self.current_best_score * config.RESET_SCORE_RATIO):
                    return self.current_best_idx, quick_score
            
            # Full search if quick check fails
            return self.orb_service.find_best_match(
                live_descriptors, project_features, self.current_best_idx
            )
            
        except Exception as e:
            logger.error(f"Match finding error: {e}")
            return -1, 0
    
    def _draw_danger_zone(self, frame: np.ndarray, frame_idx: int, score: int) -> np.ndarray:
        """Draw danger zone overlay on frame"""
        try:
            zone_points = self.danger_zones[frame_idx]
            poly = np.array(zone_points, np.int32)
            
            # Draw polygon outline
            cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
            
            # Draw semi-transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            
            # Add match info text
            text = f"DANGER ZONE - Frame: {frame_idx} (Score: {score})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            logger.debug(f"Danger zone drawn for frame {frame_idx} with score {score}")
            
        except Exception as e:
            logger.error(f"Failed to draw danger zone: {e}")
        
        return frame
    
    def get_match_status(self) -> Dict:
        """Get current matching status"""
        with self._lock:
            return {
                'project': self.current_project,
                'match_frame': self.current_best_idx,
                'match_score': self.current_best_score,
                'rtsp_url': self.rtsp_url,
                'danger_zone_active': (
                    self.current_best_score >= config.BEST_SCORE_THRESHOLD and 
                    self.current_best_idx in self.danger_zones
                )
            }
    
    def reset_tracking(self):
        """Reset tracking state"""
        with self._lock:
            self.current_best_idx = -1
            self.current_best_score = 0
        logger.info("Tracking state reset")
    
    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            self.rtsp_url = ""
            self.current_project = None
            self.danger_zones.clear()
            self.current_best_idx = -1
            self.current_best_score = 0
        
        self.orb_service.clear_cache()
        logger.info("RTSP monitor cleaned up")