import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration settings"""
    
    # Directories
    UPLOAD_FOLDER: str = 'uploads'
    THUMB_FOLDER: str = 'static/thumbs'
    PROJECT_FOLDER: str = 'projects'
    STATIC_FOLDER: str = 'static'
    TEMPLATE_FOLDER: str = 'templates'
    
    # File settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: tuple = ('.mp4', '.avi', '.mov', '.mkv')
    
    # ORB settings
    ORB_FEATURES: int = 100
    ORB_SCALE: float = 0.33
    ORB_INTERVAL: int = 10
    
    # Tracking settings
    BEST_SCORE_THRESHOLD: int = 30
    RESET_SCORE_RATIO: float = 0.6
    FAST_SEARCH_WINDOW: int = 10
    
    # Flask settings
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 5000))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.THUMB_FOLDER, exist_ok=True)
        os.makedirs(self.PROJECT_FOLDER, exist_ok=True)
        os.makedirs(self.STATIC_FOLDER, exist_ok=True)

# Global config instance
config = Config()