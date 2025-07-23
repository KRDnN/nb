from flask import Blueprint

# Create blueprints for modular routing
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
rtsp_bp = Blueprint('rtsp', __name__, url_prefix='/rtsp')
video_bp = Blueprint('video', __name__, url_prefix='/videos')
project_bp = Blueprint('project', __name__, url_prefix='/projects')

__all__ = ['main_bp', 'api_bp', 'rtsp_bp', 'video_bp', 'project_bp']