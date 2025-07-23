import os
import cv2
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from config import config
from utils import logger

class ORBService:
    """Service for ORB feature extraction and matching operations"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.cache: Dict[str, Dict] = {}
        
    def extract_frame_features(self, frame: np.ndarray, scale: float = None) -> Tuple[List, np.ndarray]:
        """Extract ORB features from a single frame"""
        try:
            scale = scale or config.ORB_SCALE
            
            # Resize frame if needed
            if scale < 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            # Convert keypoints to serializable format
            kp_points = [kp.pt for kp in keypoints] if keypoints else []
            desc_array = descriptors.astype(np.uint8) if descriptors is not None else np.zeros((0, 32), np.uint8)
            
            return kp_points, desc_array
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [], np.zeros((0, 32), np.uint8)
    
    def extract_video_features(self, video_path: str, interval: int = None) -> Dict[int, Dict]:
        """Extract ORB features from all frames in a video"""
        interval = interval or config.ORB_INTERVAL
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Extracting features from {frame_count} frames with interval {interval}")
            
            # Extract frames
            frames_data = []
            for i in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_data.append((i, frame))
            
            cap.release()
            
            # Extract features in parallel
            features_data = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    lambda x: (x[0], self.extract_frame_features(x[1])),
                    frames_data
                ))
            
            for frame_idx, (keypoints, descriptors) in results:
                features_data[frame_idx] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }
            
            logger.info(f"Feature extraction completed for {len(features_data)} frames")
            return features_data
            
        except Exception as e:
            logger.error(f"Video feature extraction failed: {e}")
            return {}
    
    def save_features(self, project_name: str, features_data: Dict) -> bool:
        """Save extracted features to project directory"""
        try:
            project_dir = os.path.join(config.PROJECT_FOLDER, project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            orb_file = os.path.join(project_dir, 'orb_data.pkl')
            with open(orb_file, 'wb') as f:
                pickle.dump(features_data, f)
            
            logger.info(f"Features saved for project: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return False
    
    def load_features(self, project_name: str) -> Dict:
        """Load features from project directory"""
        try:
            orb_file = os.path.join(config.PROJECT_FOLDER, project_name, 'orb_data.pkl')
            
            if not os.path.exists(orb_file):
                logger.warning(f"ORB file not found: {orb_file}")
                return {}
            
            with open(orb_file, 'rb') as f:
                features_data = pickle.load(f)
            
            logger.info(f"Features loaded for project: {project_name}")
            return features_data
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return {}
    
    def find_best_match(self, live_descriptors: np.ndarray, project_features: Dict,
                       last_idx: Optional[int] = None) -> Tuple[int, int]:
        """Find best matching frame using ORB descriptors"""
        try:
            if live_descriptors is None or len(live_descriptors) == 0:
                return -1, 0
            
            frame_indices = list(project_features.keys())
            frame_indices.sort()
            
            best_idx, best_score = -1, 0
            
            # Fast search around last known position
            search_indices = []
            if last_idx is not None and last_idx in frame_indices:
                idx_pos = frame_indices.index(last_idx)
                window_start = max(0, idx_pos - config.FAST_SEARCH_WINDOW)
                window_end = min(len(frame_indices), idx_pos + config.FAST_SEARCH_WINDOW + 1)
                search_indices = frame_indices[window_start:window_end]
            
            # Search in window first
            searched = set()
            for frame_idx in search_indices:
                searched.add(frame_idx)
                score = self._calculate_match_score(live_descriptors, project_features[frame_idx]['descriptors'])
                if score > best_score:
                    best_score = score
                    best_idx = frame_idx
            
            # Global search if threshold not met
            if best_score < config.BEST_SCORE_THRESHOLD:
                for frame_idx in frame_indices:
                    if frame_idx in searched:
                        continue
                    
                    score = self._calculate_match_score(live_descriptors, project_features[frame_idx]['descriptors'])
                    if score > best_score:
                        best_score = score
                        best_idx = frame_idx
            
            return best_idx, best_score
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return -1, 0
    
    def _calculate_match_score(self, desc1: np.ndarray, desc2: np.ndarray) -> int:
        """Calculate matching score between two descriptor sets"""
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return 0
            
            matches = self.matcher.match(desc1, desc2)
            return len(matches)
            
        except Exception as e:
            logger.error(f"Match score calculation failed: {e}")
            return 0
    
    def get_cached_features(self, project_name: str) -> Dict:
        """Get cached features for a project"""
        if project_name not in self.cache:
            self.cache[project_name] = self.load_features(project_name)
        return self.cache[project_name]
    
    def clear_cache(self, project_name: Optional[str] = None):
        """Clear feature cache"""
        if project_name:
            self.cache.pop(project_name, None)
            logger.info(f"Cache cleared for project: {project_name}")
        else:
            self.cache.clear()
            logger.info("All cache cleared")