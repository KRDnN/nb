from flask import render_template, request, jsonify
from routes import main_bp
from services import RTSPMonitor
from services.video_service import VideoService
from services.project_service import ProjectService
from utils import logger

# Global service instances (will be moved to proper dependency injection later)
rtsp_monitor = RTSPMonitor()
video_service = VideoService()
project_service = ProjectService()

@main_bp.route("/", methods=["GET", "POST"])
def index():
    """Main dashboard page"""
    try:
        if request.method == "POST":
            rtsp_url = request.form.get("rtsp_url", "")
            if rtsp_url:
                rtsp_monitor.set_rtsp_url(rtsp_url)
        
        # Get available projects
        projects = video_service.get_project_list()
        
        # Get current RTSP status
        status = rtsp_monitor.get_match_status()
        
        return render_template("index.html", 
                             rtsp_url=status['rtsp_url'],
                             projects=projects,
                             current_project=status['project'])
        
    except Exception as e:
        logger.error(f"Index route error: {e}")
        return render_template("error.html", error=str(e)), 500

@main_bp.route("/status")
def get_status():
    """Get current system status"""
    try:
        return jsonify(rtsp_monitor.get_match_status())
    except Exception as e:
        logger.error(f"Status route error: {e}")
        return jsonify({'error': str(e)}), 500

@main_bp.route("/dashboard")
def dashboard():
    """System dashboard with statistics"""
    try:
        # Get video statistics
        videos = video_service.get_video_list()
        
        # Get project statistics
        projects = []
        for project_name in video_service.get_project_list():
            stats = project_service.get_project_stats(project_name)
            projects.append(stats)
        
        # System stats
        system_stats = {
            'total_videos': len(videos),
            'total_projects': len(projects),
            'active_projects': len([p for p in projects if p.get('has_orb_data')]),
            'total_zones': sum(p.get('zone_count', 0) for p in projects)
        }
        
        return render_template("dashboard.html",
                             videos=videos,
                             projects=projects,
                             system_stats=system_stats)
        
    except Exception as e:
        logger.error(f"Dashboard route error: {e}")
        return render_template("error.html", error=str(e)), 500

@main_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("error.html", error="Page not found"), 404

@main_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template("error.html", error="Internal server error"), 500