import os
import json
from typing import Dict, List, Optional, Tuple
from config import config
from utils import logger
from .orb_service import ORBService

class ProjectService:
    """Service for managing projects and danger zones"""
    
    def __init__(self):
        self.orb_service = ORBService()
    
    def create_project(self, video_filename: str) -> bool:
        """Create a new project for a video"""
        try:
            project_dir = os.path.join(config.PROJECT_FOLDER, video_filename)
            os.makedirs(project_dir, exist_ok=True)
            
            # Initialize empty meta.json
            meta_data = {
                'video_filename': video_filename,
                'zones': [],
                'created_at': None,
                'updated_at': None
            }
            
            meta_path = os.path.join(project_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            logger.info(f"Project created: {video_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project {video_filename}: {e}")
            return False
    
    def extract_orb_features(self, video_filename: str) -> Tuple[bool, str]:
        """Extract ORB features for a video project"""
        try:
            video_path = os.path.join(config.UPLOAD_FOLDER, video_filename)
            if not os.path.exists(video_path):
                return False, "Video file not found"
            
            # Extract features
            logger.info(f"Starting ORB extraction for {video_filename}")
            features_data = self.orb_service.extract_video_features(video_path)
            
            if not features_data:
                return False, "Failed to extract features"
            
            # Save features
            if not self.orb_service.save_features(video_filename, features_data):
                return False, "Failed to save features"
            
            logger.info(f"ORB extraction completed for {video_filename}")
            return True, f"Extracted features from {len(features_data)} frames"
            
        except Exception as e:
            logger.error(f"ORB extraction failed for {video_filename}: {e}")
            return False, f"Extraction failed: {str(e)}"
    
    def get_project_meta(self, project_name: str) -> Optional[Dict]:
        """Get project metadata"""
        try:
            meta_path = os.path.join(config.PROJECT_FOLDER, project_name, 'meta.json')
            
            if not os.path.exists(meta_path):
                return None
            
            with open(meta_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to get project meta for {project_name}: {e}")
            return None
    
    def save_danger_zones(self, project_name: str, zones_data: Dict) -> bool:
        """Save danger zones for a project"""
        try:
            project_dir = os.path.join(config.PROJECT_FOLDER, project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Load existing meta or create new
            meta_path = os.path.join(project_dir, 'meta.json')
            meta_data = {'zones': []}
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
            
            # Update zones
            zones_list = []
            for zone_id, zone_info in zones_data.items():
                zones_list.append({
                    'id': zone_id,
                    'frame': zone_info['frame'],
                    'points': zone_info['points']
                })
            
            meta_data['zones'] = zones_list
            meta_data['video_filename'] = project_name
            
            # Save updated meta
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            logger.info(f"Danger zones saved for project {project_name}: {len(zones_list)} zones")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save danger zones for {project_name}: {e}")
            return False
    
    def delete_danger_zone(self, project_name: str, zone_id: str) -> bool:
        """Delete a specific danger zone"""
        try:
            meta_path = os.path.join(config.PROJECT_FOLDER, project_name, 'meta.json')
            
            if not os.path.exists(meta_path):
                return False
            
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            # Remove zone with matching ID
            original_count = len(meta_data.get('zones', []))
            meta_data['zones'] = [
                zone for zone in meta_data.get('zones', [])
                if str(zone.get('id')) != str(zone_id)
            ]
            
            removed = original_count - len(meta_data['zones'])
            
            if removed > 0:
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2)
                
                logger.info(f"Deleted zone {zone_id} from project {project_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete zone {zone_id} from {project_name}: {e}")
            return False
    
    def edit_danger_zone(self, project_name: str, zone_id: str, 
                        new_points: Optional[List] = None, 
                        new_frame: Optional[int] = None) -> bool:
        """Edit an existing danger zone"""
        try:
            meta_path = os.path.join(config.PROJECT_FOLDER, project_name, 'meta.json')
            
            if not os.path.exists(meta_path):
                return False
            
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            # Find and update zone
            updated = False
            for zone in meta_data.get('zones', []):
                if str(zone.get('id')) == str(zone_id):
                    if new_points is not None:
                        zone['points'] = new_points
                    if new_frame is not None:
                        zone['frame'] = new_frame
                    updated = True
                    break
            
            if updated:
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2)
                
                logger.info(f"Updated zone {zone_id} in project {project_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to edit zone {zone_id} in {project_name}: {e}")
            return False
    
    def get_danger_zones(self, project_name: str) -> Dict:
        """Get all danger zones for a project"""
        try:
            meta_data = self.get_project_meta(project_name)
            if not meta_data:
                return {}
            
            zones = {}
            for zone in meta_data.get('zones', []):
                zone_id = zone.get('id')
                if zone_id is not None:
                    zones[zone_id] = {
                        'frame': zone.get('frame'),
                        'points': zone.get('points', [])
                    }
            
            return zones
            
        except Exception as e:
            logger.error(f"Failed to get danger zones for {project_name}: {e}")
            return {}
    
    def delete_project(self, project_name: str) -> bool:
        """Delete entire project"""
        try:
            project_dir = os.path.join(config.PROJECT_FOLDER, project_name)
            
            if os.path.exists(project_dir):
                import shutil
                shutil.rmtree(project_dir)
                logger.info(f"Project deleted: {project_name}")
            
            # Clear from ORB cache
            self.orb_service.clear_cache(project_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project {project_name}: {e}")
            return False
    
    def get_project_stats(self, project_name: str) -> Dict:
        """Get project statistics"""
        try:
            stats = {
                'name': project_name,
                'has_orb_data': False,
                'has_meta': False,
                'zone_count': 0,
                'orb_frame_count': 0
            }
            
            project_dir = os.path.join(config.PROJECT_FOLDER, project_name)
            
            # Check ORB data
            orb_path = os.path.join(project_dir, 'orb_data.pkl')
            if os.path.exists(orb_path):
                stats['has_orb_data'] = True
                try:
                    orb_data = self.orb_service.load_features(project_name)
                    stats['orb_frame_count'] = len(orb_data)
                except:
                    pass
            
            # Check meta data
            meta_path = os.path.join(project_dir, 'meta.json')
            if os.path.exists(meta_path):
                stats['has_meta'] = True
                try:
                    meta_data = self.get_project_meta(project_name)
                    if meta_data:
                        stats['zone_count'] = len(meta_data.get('zones', []))
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get project stats for {project_name}: {e}")
            return {'name': project_name, 'error': str(e)}
    
    def export_project(self, project_name: str) -> Optional[Dict]:
        """Export project data"""
        try:
            project_dir = os.path.join(config.PROJECT_FOLDER, project_name)
            
            if not os.path.exists(project_dir):
                return None
            
            export_data = {
                'project_name': project_name,
                'meta': self.get_project_meta(project_name),
                'has_orb_data': os.path.exists(os.path.join(project_dir, 'orb_data.pkl')),
                'stats': self.get_project_stats(project_name)
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export project {project_name}: {e}")
            return None