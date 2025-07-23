#!/usr/bin/env python3
"""
RTSP Danger Zone Monitoring System
Refactored version with modular architecture, improved performance, and better maintainability.
"""

import os
import sys
from flask import Flask
from config import config
from utils import logger

def create_app():
    """Application factory pattern"""
    
    # Create Flask app
    app = Flask(__name__, 
                template_folder=config.TEMPLATE_FOLDER,
                static_folder=config.STATIC_FOLDER)
    
    # Configure app
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE
    
    # Register blueprints
    from routes import main_bp, api_bp, rtsp_bp, video_bp, project_bp
    from routes.main_routes import main_bp as main_routes
    from routes.rtsp_routes import rtsp_bp as rtsp_routes
    # Note: Additional route modules would be imported here
    
    app.register_blueprint(main_routes)
    app.register_blueprint(rtsp_routes)
    
    # Initialize logging
    logger.info("RTSP Danger Zone Monitoring System starting...")
    logger.info(f"Configuration: DEBUG={config.DEBUG}, HOST={config.HOST}, PORT={config.PORT}")
    
    return app

def main():
    """Main entry point"""
    try:
        # Create and configure app
        app = create_app()
        
        # Run the application
        logger.info("Starting Flask development server...")
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True  # Enable threading for RTSP streaming
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()