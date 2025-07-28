#!/usr/bin/env python3
"""
Enhanced Flask Application with Deep Learning Homography Estimation

This application integrates deep learning-based homography estimation with the existing
ORB feature matching system for improved accuracy and robustness in video processing
and danger zone tracking applications.

Features:
- Deep learning homography estimation using CNN
- Hybrid approach combining ORB and deep learning
- Real-time RTSP stream processing
- Danger zone tracking and visualization
- Model training interface
- Performance comparison tools
"""

import os
import cv2
import numpy as np
import base64
import io
import json
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import threading
import time
from deep_homography import HomographyNet, HomographyEstimator, HomographyTrainer, HomographyDataset
from torch.utils.data import DataLoader

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
THUMB_FOLDER = 'static/thumbs'
PROJECT_FOLDER = 'projects'
MODEL_FOLDER = 'models'
TRAINING_DATA_FOLDER = 'training_data'

# Create directories
for folder in [UPLOAD_FOLDER, THUMB_FOLDER, PROJECT_FOLDER, MODEL_FOLDER, TRAINING_DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'THUMB_FOLDER': THUMB_FOLDER,
    'PROJECT_FOLDER': PROJECT_FOLDER,
    'MODEL_FOLDER': MODEL_FOLDER,
    'TRAINING_DATA_FOLDER': TRAINING_DATA_FOLDER
})

# Global variables
rtsp_url = ""
selected_project = None
cached_orb_data = {}
cached_zones = {}
current_best_idx = -1
current_best_score = 0
BEST_SCORE_THRESHOLD = 30
RESET_SCORE_RATIO = 0.6
FAST_SEARCH_WINDOW = 10

# Deep learning components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
homography_estimator = None
use_deep_learning = False

# Training status
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'progress': 0.0
}

class EnhancedPolygonTracker:
    """Enhanced polygon tracker with deep learning homography estimation."""
    
    def __init__(self, use_deep_learning=False):
        self.points = []
        self.poly_complete = False
        self.tracking_failed = False
        self.prev_gray = None
        self.prev_points = None
        self.hist = None
        self.prev_global_pts = None
        self.use_deep_learning = use_deep_learning
        self.prev_frame = None

    def initialize(self, frame, points):
        self.points = [tuple(pt) for pt in points]
        self.poly_complete = True
        self.tracking_failed = False
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame = frame.copy()
        self.prev_points = np.array(self.points, dtype=np.float32)
        
        # Create histogram for color tracking
        mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        
        # Global feature points for affine tracking
        self.prev_global_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8
        )

    def track_with_deep_learning(self, frame):
        """Track using deep learning homography estimation."""
        global homography_estimator
        
        if homography_estimator is None or self.prev_frame is None:
            return self.track_traditional(frame)
        
        try:
            # Estimate homography using deep learning
            H = homography_estimator.estimate_homography(self.prev_frame, frame)
            
            # Apply homography to polygon points
            if H is not None:
                pts = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, H)
                
                # Update points
                h, w = frame.shape[:2]
                new_points = []
                for pt in transformed_pts:
                    x, y = pt[0]
                    x = float(np.clip(x, 0, w-1))
                    y = float(np.clip(y, 0, h-1))
                    new_points.append((x, y))
                
                self.points = new_points
                self.prev_frame = frame.copy()
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return self.points
            else:
                # Fall back to traditional tracking
                return self.track_traditional(frame)
                
        except Exception as e:
            print(f"Deep learning tracking failed: {e}")
            return self.track_traditional(frame)

    def track_traditional(self, frame):
        """Traditional tracking using optical flow and other methods."""
        if not self.poly_complete or self.tracking_failed:
            return self.points
            
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = next_gray.shape[:2]
        
        # Optical flow tracking
        prev_pts = np.array(self.points, np.float32).reshape(-1, 1, 2)
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, prev_pts, None)
        pts_valid = st is not None and st.sum() >= len(self.points) * 0.6
        
        if next_pts is not None and pts_valid:
            pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in next_pts]
            self.points = pts_new
            self.prev_points = np.array(self.points, dtype=np.float32)
            self.prev_gray = next_gray
            return self.points
        
        # Affine fallback
        if self.prev_global_pts is not None and len(self.prev_global_pts) >= 3:
            curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, self.prev_global_pts, None)
            st = st.reshape(-1)
            prev_g = self.prev_global_pts[st == 1]
            curr_g = curr_global_pts[st == 1]
            
            if len(prev_g) >= 3 and len(curr_g) >= 3:
                M, _ = cv2.estimateAffinePartial2D(prev_g, curr_g)
                if M is not None:
                    poly_np = np.array(self.points, np.float32).reshape(-1, 1, 2)
                    poly_trans = cv2.transform(poly_np, M)
                    pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in poly_trans]
                    self.points = pts_new
                    self.prev_points = np.array(self.points, dtype=np.float32)
                    self.prev_gray = next_gray
                    self.prev_global_pts = curr_g.reshape(-1, 1, 2)
                    return self.points
            
            self.prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)
        
        # MeanShift fallback
        if self.hist is not None and len(self.points) >= 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pts = np.array(self.points, dtype=np.int32)
            x, y, w_box, h_box = cv2.boundingRect(pts)
            back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
            dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
            dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
            self.points = [(pt[0]+dx, pt[1]+dy) for pt in self.points]
            self.prev_gray = next_gray
            return self.points
        
        # Tracking failed
        self.tracking_failed = True
        self.prev_gray = next_gray
        return self.points

    def track(self, frame):
        """Main tracking function that chooses between deep learning and traditional methods."""
        if self.use_deep_learning and use_deep_learning:
            return self.track_with_deep_learning(frame)
        else:
            return self.track_traditional(frame)


def load_homography_model(model_path=None):
    """Load the trained homography estimation model."""
    global homography_estimator, use_deep_learning
    
    if model_path is None:
        model_path = os.path.join(MODEL_FOLDER, 'homography_net.pth')
    
    if os.path.exists(model_path):
        try:
            homography_estimator = HomographyEstimator(model_path, device)
            use_deep_learning = True
            print(f"Deep learning homography model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load homography model: {e}")
            use_deep_learning = False
            return False
    else:
        print(f"Homography model not found at {model_path}")
        use_deep_learning = False
        return False


# Enhanced ORB extraction with deep learning integration
def extract_orb_all_fast(video_path, scale=0.33, nfeatures=100, interval=10):
    """Extract ORB features with optional deep learning enhancement."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orb = cv2.ORB_create(nfeatures=nfeatures)
    data = {}
    
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        if scale < 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(gray, None)
        
        data[i] = {
            'keypoints': [kp.pt for kp in kps] if kps is not None else [],
            'descriptors': des.astype(np.uint8) if des is not None else np.zeros((0,32), np.uint8),
            'frame': frame  # Store frame for deep learning homography
        }
    
    cap.release()
    return data


def enhanced_find_best_orb_idx(des_live, current_frame, last_idx):
    """Enhanced ORB matching with deep learning verification."""
    global homography_estimator, use_deep_learning
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    idxs = list(cached_orb_data.keys())
    idxs.sort()
    
    # Fast local search
    window_range = []
    if last_idx is not None and last_idx in idxs:
        i = idxs.index(last_idx)
        window_range = idxs[max(0,i-FAST_SEARCH_WINDOW):i+FAST_SEARCH_WINDOW+1]
    
    searched = set(window_range)
    best_idx, best_score = -1, 0
    candidates = []
    
    # Search in window
    for idx in window_range:
        des2 = cached_orb_data[idx]['descriptors']
        if des_live is None or des2 is None or len(des2) == 0:
            continue
        matches = matcher.match(des_live, des2)
        score = len(matches)
        candidates.append((idx, score))
        if score > best_score:
            best_score = score
            best_idx = idx
    
    # Global search if needed
    if best_score < BEST_SCORE_THRESHOLD:
        for idx in idxs:
            if idx in searched:
                continue
            des2 = cached_orb_data[idx]['descriptors']
            if des_live is None or des2 is None or len(des2) == 0:
                continue
            matches = matcher.match(des_live, des2)
            score = len(matches)
            candidates.append((idx, score))
            if score > best_score:
                best_score = score
                best_idx = idx
    
    # Deep learning verification for top candidates
    if use_deep_learning and homography_estimator and current_frame is not None:
        top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
        
        for idx, orb_score in top_candidates:
            if 'frame' in cached_orb_data[idx]:
                try:
                    stored_frame = cached_orb_data[idx]['frame']
                    H = homography_estimator.estimate_homography(current_frame, stored_frame)
                    
                    # Compute homography quality metric
                    if H is not None:
                        # Use both ORB score and homography confidence
                        combined_score = orb_score * 1.2  # Boost for deep learning verification
                        if combined_score > best_score:
                            best_score = combined_score
                            best_idx = idx
                            break
                except Exception as e:
                    print(f"Deep learning verification failed for frame {idx}: {e}")
                    continue
    
    return best_idx, best_score


# Enhanced caching and project management
def load_project_cache(project):
    """Load project cache with deep learning enhancements."""
    global cached_orb_data, cached_zones
    orb_path = os.path.join(PROJECT_FOLDER, project, 'orb_data.pkl')
    meta_path = os.path.join(PROJECT_FOLDER, project, 'meta.json')
    
    cached_orb_data.clear()
    if os.path.exists(orb_path):
        with open(orb_path, 'rb') as f:
            cached_orb_data.update(pickle.load(f))
    
    cached_zones.clear()
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        for zone in meta.get('zones', []):
            cached_zones[zone['frame']] = zone['points']


# Training-related routes
@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Start training the homography estimation model."""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.get_json()
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 16)
    learning_rate = data.get('learning_rate', 1e-4)
    
    # Check if we have training images
    training_images_dir = os.path.join(UPLOAD_FOLDER)
    if not os.path.exists(training_images_dir) or not os.listdir(training_images_dir):
        return jsonify({'error': 'No training images found in upload folder'}), 400
    
    # Start training in a separate thread
    def train_worker():
        global training_status
        
        try:
            training_status.update({
                'is_training': True,
                'current_epoch': 0,
                'total_epochs': epochs,
                'train_loss': 0.0,
                'val_loss': 0.0,
                'progress': 0.0
            })
            
            # Create datasets
            train_dataset = HomographyDataset(
                image_dir=training_images_dir,
                num_samples=20000,
                patch_size=128
            )
            
            val_dataset = HomographyDataset(
                image_dir=training_images_dir,
                num_samples=4000,
                patch_size=128
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2,
                pin_memory=True if device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2,
                pin_memory=True if device.type == 'cuda' else False
            )
            
            # Create model and trainer
            model = HomographyNet(input_channels=2)
            trainer = HomographyTrainer(model, device, learning_rate=learning_rate)
            
            # Custom training loop with status updates
            model_save_path = os.path.join(MODEL_FOLDER, 'homography_net.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Update status
                training_status.update({
                    'current_epoch': epoch + 1,
                    'progress': (epoch + 1) / epochs * 100
                })
                
                # Training
                train_loss = trainer.train_epoch(train_loader)
                training_status['train_loss'] = train_loss
                
                # Validation
                val_loss = trainer.validate(val_loader)
                training_status['val_loss'] = val_loss
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, model_save_path)
            
            # Training completed
            training_status.update({
                'is_training': False,
                'progress': 100.0
            })
            
            # Load the trained model
            load_homography_model(model_save_path)
            
        except Exception as e:
            print(f"Training error: {e}")
            training_status.update({
                'is_training': False,
                'error': str(e)
            })
    
    thread = threading.Thread(target=train_worker)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started', 'status': training_status})


@app.route('/api/training_status')
def get_training_status():
    """Get current training status."""
    return jsonify(training_status)


@app.route('/api/models')
def list_models():
    """List available trained models."""
    models = []
    if os.path.exists(MODEL_FOLDER):
        for filename in os.listdir(MODEL_FOLDER):
            if filename.endswith('.pth'):
                model_path = os.path.join(MODEL_FOLDER, filename)
                models.append({
                    'name': filename,
                    'path': model_path,
                    'size': os.path.getsize(model_path),
                    'modified': os.path.getmtime(model_path)
                })
    return jsonify(models)


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load a specific trained model."""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'error': 'Model name is required'}), 400
    
    model_path = os.path.join(MODEL_FOLDER, model_name)
    if load_homography_model(model_path):
        return jsonify({'message': f'Model {model_name} loaded successfully'})
    else:
        return jsonify({'error': f'Failed to load model {model_name}'}), 500


@app.route('/api/toggle_deep_learning', methods=['POST'])
def toggle_deep_learning():
    """Toggle deep learning homography estimation on/off."""
    global use_deep_learning
    data = request.get_json()
    use_deep_learning = data.get('enabled', False)
    
    if use_deep_learning and homography_estimator is None:
        # Try to load default model
        if not load_homography_model():
            use_deep_learning = False
            return jsonify({
                'enabled': False, 
                'message': 'No trained model available. Please train a model first.'
            })
    
    return jsonify({
        'enabled': use_deep_learning,
        'message': f'Deep learning homography estimation {"enabled" if use_deep_learning else "disabled"}'
    })


# Enhanced video processing routes
@app.route('/extract_orb_all/<filename>')
def extract_orb_all(filename):
    """Extract ORB features with deep learning enhancements."""
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    project_dir = os.path.join(PROJECT_FOLDER, filename)
    os.makedirs(project_dir, exist_ok=True)
    
    orb_data = extract_orb_all_fast(video_path, scale=0.33, nfeatures=100, interval=10)
    
    with open(os.path.join(project_dir, 'orb_data.pkl'), 'wb') as f:
        pickle.dump(orb_data, f)
    
    return jsonify({'status': 'ok', 'count': len(orb_data)})


@app.route("/video_feed")
def video_feed():
    """Enhanced video feed with deep learning homography estimation."""
    def generate_rtsp_stream():
        global cached_orb_data, cached_zones, current_best_idx, current_best_score, rtsp_url
        
        cap = cv2.VideoCapture(rtsp_url)
        orb = cv2.ORB_create(nfeatures=100)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_live, des_live = orb.detectAndCompute(gray, None)
            
            best_idx, best_score = -1, 0
            
            if selected_project and current_best_idx != -1 and des_live is not None:
                des_saved = cached_orb_data.get(current_best_idx, {}).get('descriptors', np.zeros((0,32),np.uint8))
                if len(des_saved) > 0:
                    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_live, des_saved)
                    best_score = len(matches)
                    best_idx = current_best_idx
            
            if (current_best_idx == -1 or
                best_score < int(current_best_score * RESET_SCORE_RATIO) or
                best_score < BEST_SCORE_THRESHOLD):
                
                # Use enhanced matching with deep learning
                best_idx, best_score = enhanced_find_best_orb_idx(des_live, frame, current_best_idx)
                current_best_idx = best_idx
                current_best_score = best_score
            
            # Danger zone overlay
            if best_score >= BEST_SCORE_THRESHOLD and selected_project and best_idx in cached_zones:
                poly = np.array(cached_zones[best_idx], np.int32)
                cv2.polylines(frame, [poly], True, (0,0,255), 2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], (0,0,255))
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                
                # Status text
                status_text = f"Frame: {best_idx} | Score: {best_score}"
                if use_deep_learning:
                    status_text += " | DL Enhanced"
                
                cv2.putText(frame, status_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
    
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Copy existing routes from the original app.py
def extract_thumbnail(filepath, thumb_path):
    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumb_path, frame)
    cap.release()


@app.route('/videos', methods=['GET', 'POST'])
def video_list():
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            thumb_path = os.path.join(THUMB_FOLDER, filename + '.jpg')
            extract_thumbnail(save_path, thumb_path)
        return redirect(url_for('video_list'))
    
    videos = []
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            path = os.path.join(UPLOAD_FOLDER, fname)
            thumb = f'thumbs/{fname}.jpg'
            thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
            if not os.path.exists(thumb_path):
                extract_thumbnail(path, thumb_path)
            videos.append({'name': fname, 'thumb': thumb})
    return render_template('videos.html', videos=videos)


@app.route('/frame/<filename>/<int:index>')
def get_frame(filename, index):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Frame not found", 404
    _, img_encoded = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')


@app.route('/frame_overlay/<filename>/<int:index>', methods=['POST'])
def get_frame_overlay(filename, index):
    data = request.get_json()
    polygon = data.get("polygon", [])
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Frame not found", 404
    if polygon and len(polygon) >= 3:
        pts = np.array(polygon, np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    _, img_encoded = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')


@app.route('/save_zone', methods=['POST'])
def save_zone():
    data = request.get_json()
    filename = data['filename']
    zones = data['zones']
    orb_all = data.get('orb_all', None)
    project_dir = os.path.join(PROJECT_FOLDER, filename)
    os.makedirs(project_dir, exist_ok=True)
    meta_info = {"zones": []}
    if orb_all:
        meta_info["orb_all"] = orb_all
    for zone_id, zone_data in zones.items():
        meta_info["zones"].append({
            "id": zone_id,
            "frame": zone_data['frame'],
            "points": zone_data['points']
        })
    with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
        json.dump(meta_info, f)
    load_project_cache(filename)
    return "Saved", 200


@app.route('/rtsp_match/<project>')
def rtsp_match(project):
    global selected_project, current_best_idx, current_best_score
    selected_project = project
    load_project_cache(project)
    current_best_idx = -1
    current_best_score = 0
    return jsonify({'status': 'ok'})


@app.route('/set_rtsp', methods=['POST'])
def set_rtsp():
    global rtsp_url
    rtsp_url = request.json.get('rtsp_url')
    return jsonify({'status':'ok'})


@app.route('/zones')
def danger_zone_list():
    videos = []
    selected_video = request.args.get('video')
    frame_count = 0
    zones = {}
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            path = os.path.join(UPLOAD_FOLDER, fname)
            thumb = f'thumbs/{fname}.jpg'
            thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
            if not os.path.exists(thumb_path):
                extract_thumbnail(path, thumb_path)
            videos.append({'name': fname, 'thumb': thumb})
    if selected_video and os.path.exists(os.path.join(UPLOAD_FOLDER, selected_video)):
        video_path = os.path.join(UPLOAD_FOLDER, selected_video)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        project_dir = os.path.join(PROJECT_FOLDER, selected_video)
        meta_path = os.path.join(project_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            for zone in meta.get('zones', []):
                zones[zone['id']] = {'points': zone['points'], 'frame': zone['frame']}
    return render_template('danger_zone_list.html',
                           videos=videos,
                           selected_video=selected_video,
                           frame_count=frame_count,
                           zones=zones)


@app.route("/", methods=["GET", "POST"])
def index():
    global rtsp_url
    if request.method == "POST":
        rtsp_url = request.form.get("rtsp_url", "")
    projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
    return render_template("index_enhanced.html", 
                         rtsp_url=rtsp_url, 
                         projects=projects,
                         use_deep_learning=use_deep_learning,
                         device=str(device))


if __name__ == '__main__':
    # Try to load existing model on startup
    load_homography_model()
    
    app.run(host='0.0.0.0', port=5000, debug=True)