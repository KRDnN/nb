import logging
import sys
from typing import Optional
from config import config

def setup_logger(name: str = __name__, level: Optional[str] = None) -> logging.Logger:
    """Setup and return a configured logger"""
    
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if config.LOG_FILE:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger('rtsp_monitor')