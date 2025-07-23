import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils import logger

class PolygonTracker:
    """Advanced polygon tracker using multiple tracking strategies"""
    
    def __init__(self):
        self.points: List[Tuple[float, float]] = []
        self.poly_complete: bool = False
        self.tracking_failed: bool = False
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.hist: Optional[np.ndarray] = None
        self.prev_global_pts: Optional[np.ndarray] = None
        
    def initialize(self, frame: np.ndarray, points: List[Tuple[float, float]]) -> bool:
        """Initialize tracker with first frame and polygon points"""
        try:
            self.points = [tuple(pt) for pt in points]
            self.poly_complete = True
            self.tracking_failed = False
            
            # Convert to grayscale
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = np.array(self.points, dtype=np.float32)
            
            # Create mask and histogram for color tracking
            mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
            
            # Get global feature points for affine transformation
            self.prev_global_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8
            )
            
            logger.info(f"Polygon tracker initialized with {len(self.points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize polygon tracker: {e}")
            return False
    
    def track(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """Track polygon in current frame using multi-strategy approach"""
        if not self.poly_complete or self.tracking_failed:
            return self.points
            
        try:
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = next_gray.shape[:2]
            
            # Strategy 1: Direct optical flow tracking
            if self._track_optical_flow(next_gray, w, h):
                return self.points
                
            # Strategy 2: Affine transformation fallback
            if self._track_affine_transform(next_gray, w, h):
                return self.points
                
            # Strategy 3: MeanShift color tracking fallback
            if self._track_meanshift(frame):
                return self.points
                
            # All strategies failed
            logger.warning("All tracking strategies failed")
            self.tracking_failed = True
            self.prev_gray = next_gray
            return self.points
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            self.tracking_failed = True
            return self.points
    
    def _track_optical_flow(self, next_gray: np.ndarray, w: int, h: int) -> bool:
        """Track using optical flow on polygon points"""
        try:
            prev_pts = np.array(self.points, np.float32).reshape(-1, 1, 2)
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, next_gray, prev_pts, None
            )
            
            # Check if enough points are tracked successfully
            pts_valid = st is not None and st.sum() >= len(self.points) * 0.6
            
            if next_pts is not None and pts_valid:
                # Clip points to frame boundaries
                pts_new = [
                    (float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1)))
                    for pt in next_pts
                ]
                
                self.points = pts_new
                self.prev_points = np.array(self.points, dtype=np.float32)
                self.prev_gray = next_gray
                return True
                
        except Exception as e:
            logger.error(f"Optical flow tracking failed: {e}")
            
        return False
    
    def _track_affine_transform(self, next_gray: np.ndarray, w: int, h: int) -> bool:
        """Track using global affine transformation"""
        try:
            if self.prev_global_pts is None or len(self.prev_global_pts) < 3:
                return False
                
            # Track global feature points
            curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, next_gray, self.prev_global_pts, None
            )
            
            st = st.reshape(-1)
            prev_g = self.prev_global_pts[st == 1]
            curr_g = curr_global_pts[st == 1]
            
            if len(prev_g) >= 3 and len(curr_g) >= 3:
                # Estimate affine transformation
                M, _ = cv2.estimateAffinePartial2D(prev_g, curr_g)
                
                if M is not None:
                    # Apply transformation to polygon
                    poly_np = np.array(self.points, np.float32).reshape(-1, 1, 2)
                    poly_trans = cv2.transform(poly_np, M)
                    
                    pts_new = [
                        (float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1)))
                        for pt in poly_trans
                    ]
                    
                    self.points = pts_new
                    self.prev_points = np.array(self.points, dtype=np.float32)
                    self.prev_gray = next_gray
                    self.prev_global_pts = curr_g.reshape(-1, 1, 2)
                    return True
            
            # Update global points for next iteration
            self.prev_global_pts = cv2.goodFeaturesToTrack(
                next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8
            )
            
        except Exception as e:
            logger.error(f"Affine transform tracking failed: {e}")
            
        return False
    
    def _track_meanshift(self, frame: np.ndarray) -> bool:
        """Track using MeanShift color-based tracking"""
        try:
            if self.hist is None or len(self.points) < 3:
                return False
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pts = np.array(self.points, dtype=np.int32)
            
            # Get bounding rectangle
            x, y, w_box, h_box = cv2.boundingRect(pts)
            
            # Calculate back projection
            back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
            
            # Apply MeanShift
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
            
            # Calculate displacement
            dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
            dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
            
            # Update polygon points
            self.points = [(pt[0] + dx, pt[1] + dy) for pt in self.points]
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True
            
        except Exception as e:
            logger.error(f"MeanShift tracking failed: {e}")
            
        return False
    
    def reset(self):
        """Reset tracker state"""
        self.points = []
        self.poly_complete = False
        self.tracking_failed = False
        self.prev_gray = None
        self.prev_points = None
        self.hist = None
        self.prev_global_pts = None
        logger.info("Polygon tracker reset")