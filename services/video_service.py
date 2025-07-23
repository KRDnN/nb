import os
import cv2
import json
from typing import List, Dict, Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from config import config
from utils import logger
import numpy as np

class VideoService:
    """Service for video file management and processing"""
    
    def __init__(self):
        pass
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return filename.lower().endswith(config.ALLOWED_EXTENSIONS)
    
    def upload_video(self, file: FileStorage) -> Tuple[bool, str]:
        """Upload and process video file"""
        try:
            if not file or not file.filename:
                return False, "No file provided"
            
            if not self.is_allowed_file(file.filename):
                return False, f"File type not allowed. Supported: {config.ALLOWED_EXTENSIONS}"
            
            # Secure filename
            filename = secure_filename(file.filename)
            if not filename:
                return False, "Invalid filename"
            
            # Save file
            file_path = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Validate video file
            if not self._validate_video(file_path):
                os.remove(file_path)
                return False, "Invalid video file"
            
            # Generate thumbnail
            self._generate_thumbnail(file_path, filename)
            
            logger.info(f"Video uploaded successfully: {filename}")
            return True, filename
            
        except Exception as e:
            logger.error(f"Video upload failed: {e}")
            return False, f"Upload failed: {str(e)}"
    
    def _validate_video(self, file_path: str) -> bool:
        """Validate video file can be opened"""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            return ret
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return False
    
    def _generate_thumbnail(self, video_path: str, filename: str):
        """Generate thumbnail for video"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                thumb_path = os.path.join(config.THUMB_FOLDER, filename + '.jpg')
                cv2.imwrite(thumb_path, frame)
                logger.info(f"Thumbnail generated: {filename}")
            cap.release()
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
    
    def get_video_list(self) -> List[Dict]:
        """Get list of uploaded videos with metadata"""
        try:
            videos = []
            
            if not os.path.exists(config.UPLOAD_FOLDER):
                return videos
            
            for filename in os.listdir(config.UPLOAD_FOLDER):
                if self.is_allowed_file(filename):
                    video_info = self._get_video_info(filename)
                    if video_info:
                        videos.append(video_info)
            
            # Sort by upload time (newest first)
            videos.sort(key=lambda x: x.get('mtime', 0), reverse=True)
            return videos
            
        except Exception as e:
            logger.error(f"Failed to get video list: {e}")
            return []
    
    def _get_video_info(self, filename: str) -> Optional[Dict]:
        """Get video information and metadata"""
        try:
            video_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            if not os.path.exists(video_path):
                return None
            
            # Get file stats
            stat = os.stat(video_path)
            
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = int(frame_count / fps) if fps > 0 else 0
            
            cap.release()
            
            # Check thumbnail
            thumb_path = os.path.join(config.THUMB_FOLDER, filename + '.jpg')
            has_thumbnail = os.path.exists(thumb_path)
            
            if not has_thumbnail:
                self._generate_thumbnail(video_path, filename)
                has_thumbnail = True
            
            # Check if project exists
            project_exists = os.path.exists(os.path.join(config.PROJECT_FOLDER, filename))
            
            return {
                'name': filename,
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'duration': duration,
                'frame_count': frame_count,
                'resolution': f"{width}x{height}",
                'fps': round(fps, 2),
                'thumbnail': f"thumbs/{filename}.jpg" if has_thumbnail else None,
                'has_project': project_exists
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info for {filename}: {e}")
            return None
    
    def get_frame(self, filename: str, frame_index: int) -> Optional[bytes]:
        """Extract specific frame from video"""
        try:
            video_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            if not os.path.exists(video_path):
                return None
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            _, encoded = cv2.imencode('.jpg', frame)
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to extract frame {frame_index} from {filename}: {e}")
            return None
    
    def get_frame_with_overlay(self, filename: str, frame_index: int, 
                              polygon: Optional[List] = None) -> Optional[bytes]:
        """Get frame with polygon overlay"""
        try:
            video_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            if not os.path.exists(video_path):
                return None
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Add polygon overlay if provided
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon, np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                
                # Add semi-transparent overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            
            _, encoded = cv2.imencode('.jpg', frame)
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to get frame with overlay: {e}")
            return None
    
    def delete_video(self, filename: str) -> bool:
        """Delete video file and associated data"""
        try:
            # Remove video file
            video_path = os.path.join(config.UPLOAD_FOLDER, filename)
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Remove thumbnail
            thumb_path = os.path.join(config.THUMB_FOLDER, filename + '.jpg')
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
            
            # Remove project directory
            project_path = os.path.join(config.PROJECT_FOLDER, filename)
            if os.path.exists(project_path):
                import shutil
                shutil.rmtree(project_path)
            
            logger.info(f"Video deleted: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {filename}: {e}")
            return False
    
    def get_project_list(self) -> List[str]:
        """Get list of available projects"""
        try:
            if not os.path.exists(config.PROJECT_FOLDER):
                return []
            
            projects = []
            for item in os.listdir(config.PROJECT_FOLDER):
                project_path = os.path.join(config.PROJECT_FOLDER, item)
                if os.path.isdir(project_path):
                    # Check if project has required files
                    meta_file = os.path.join(project_path, 'meta.json')
                    orb_file = os.path.join(project_path, 'orb_data.pkl')
                    
                    if os.path.exists(meta_file) or os.path.exists(orb_file):
                        projects.append(item)
            
            return sorted(projects)
            
        except Exception as e:
            logger.error(f"Failed to get project list: {e}")
            return []