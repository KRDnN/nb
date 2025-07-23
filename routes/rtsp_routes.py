from flask import Response, request, jsonify
from routes import rtsp_bp
from services import RTSPMonitor
from utils import logger

# Global service instance
rtsp_monitor = RTSPMonitor()

@rtsp_bp.route("/feed")
def video_feed():
    """RTSP video stream with danger zone overlay"""
    try:
        return Response(
            rtsp_monitor.get_stream_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"RTSP feed error: {e}")
        return f"Stream error: {e}", 500

@rtsp_bp.route("/set_url", methods=['POST'])
def set_rtsp_url():
    """Set RTSP stream URL"""
    try:
        data = request.get_json()
        rtsp_url = data.get('rtsp_url', '')
        
        if not rtsp_url:
            return jsonify({'error': 'RTSP URL is required'}), 400
        
        success = rtsp_monitor.set_rtsp_url(rtsp_url)
        
        if success:
            return jsonify({'status': 'success', 'rtsp_url': rtsp_url})
        else:
            return jsonify({'error': 'Failed to set RTSP URL'}), 500
            
    except Exception as e:
        logger.error(f"Set RTSP URL error: {e}")
        return jsonify({'error': str(e)}), 500

@rtsp_bp.route("/set_project/<project_name>", methods=['POST'])
def set_monitoring_project(project_name):
    """Set project for RTSP monitoring"""
    try:
        success = rtsp_monitor.set_project(project_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'project': project_name,
                'message': f'Monitoring project set to: {project_name}'
            })
        else:
            return jsonify({'error': f'Failed to set project: {project_name}'}), 500
            
    except Exception as e:
        logger.error(f"Set monitoring project error: {e}")
        return jsonify({'error': str(e)}), 500

@rtsp_bp.route("/status")
def get_rtsp_status():
    """Get current RTSP monitoring status"""
    try:
        return jsonify(rtsp_monitor.get_match_status())
    except Exception as e:
        logger.error(f"RTSP status error: {e}")
        return jsonify({'error': str(e)}), 500

@rtsp_bp.route("/reset", methods=['POST'])
def reset_tracking():
    """Reset tracking state"""
    try:
        rtsp_monitor.reset_tracking()
        return jsonify({'status': 'success', 'message': 'Tracking state reset'})
    except Exception as e:
        logger.error(f"Reset tracking error: {e}")
        return jsonify({'error': str(e)}), 500

@rtsp_bp.route("/stop", methods=['POST'])
def stop_monitoring():
    """Stop RTSP monitoring"""
    try:
        rtsp_monitor.cleanup()
        return jsonify({'status': 'success', 'message': 'RTSP monitoring stopped'})
    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        return jsonify({'error': str(e)}), 500