# import os
# import cv2
# import numpy as np
# import base64
# import io
# import json
# from concurrent.futures import ThreadPoolExecutor
# from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/uploads'
# THUMB_FOLDER = 'static/thumbs'
# PROJECT_FOLDER = 'projects'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(THUMB_FOLDER, exist_ok=True)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['THUMB_FOLDER'] = THUMB_FOLDER
# app.config['PROJECT_FOLDER'] = PROJECT_FOLDER

# def get_video_info(filepath):
#     cap = cv2.VideoCapture(filepath)
#     if not cap.isOpened():
#         return {'name': os.path.basename(filepath), 'duration': 0, 'resolution': 'N/A'}
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     duration = int(frame_count / fps) if fps > 0 else 0
#     cap.release()
#     return {
#         'name': os.path.basename(filepath),
#         'duration': duration,
#         'resolution': f"{width}x{height}"
#     }

# def extract_thumbnail(filepath, thumb_path):
#     cap = cv2.VideoCapture(filepath)
#     success, frame = cap.read()
#     if success:
#         cv2.imwrite(thumb_path, frame)
#     cap.release()

# # ----------- Robust Polygon Tracker for Flask ----------
# def track_polygon_history_robust(frames, poly_init):
#     poly_hist = [poly_init[:]]
#     poly_pts = poly_init[:]
#     prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
#     mask = np.zeros(prev_gray.shape, dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(poly_pts, np.int32)], 255)
#     hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
#     cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
#     prev_global_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)
#     tracking_failed = False

#     for i in range(1, len(frames)):
#         frame = frames[i]
#         if tracking_failed:
#             poly_hist.append(poly_pts[:])
#             prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             continue

#         next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         h, w = next_gray.shape[:2]
#         # 1. 내부 Optical Flow
#         prev_pts = np.array(poly_pts, np.float32).reshape(-1, 1, 2)
#         next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)
#         pts_valid = st.sum() >= len(poly_pts) * 0.6 if st is not None else False

#         if next_pts is not None and pts_valid:
#             poly_pts = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in next_pts]
#             poly_hist.append(poly_pts[:])
#             prev_gray = next_gray
#             continue

#         # 2. 글로벌 affine
#         if prev_global_pts is not None and len(prev_global_pts) >= 3:
#             curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_global_pts, None)
#             st = st.reshape(-1)
#             prev_g = prev_global_pts[st == 1]
#             curr_g = curr_global_pts[st == 1]
#             if len(prev_g) >= 3 and len(curr_g) >= 3:
#                 M, inliers = cv2.estimateAffinePartial2D(prev_g, curr_g)
#                 if M is not None:
#                     poly_np = np.array(poly_pts, np.float32).reshape(-1, 1, 2)
#                     poly_trans = cv2.transform(poly_np, M)
#                     poly_pts = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in poly_trans]
#                     poly_hist.append(poly_pts[:])
#                     prev_gray = next_gray
#                     prev_global_pts = curr_g.reshape(-1, 1, 2)
#                     continue
#             prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

#         # 3. meanShift fallback (히스토그램 기반)
#         if hist is not None and len(poly_pts) >= 3:
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             pts = np.array(poly_pts, dtype=np.int32)
#             x, y, w_box, h_box = cv2.boundingRect(pts)
#             back_proj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
#             term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#             _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
#             dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
#             dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
#             poly_pts = [(pt[0]+dx, pt[1]+dy) for pt in poly_pts]
#             poly_hist.append(poly_pts[:])
#             prev_gray = next_gray
#             continue

#         # 실패시 이전 값 그대로, flag
#         tracking_failed = True
#         poly_hist.append(poly_pts[:])
#         prev_gray = next_gray
#     return poly_hist

# # ----------- 빠른 ORB 추출 (resize + nfeatures 제한 + 병렬) -----------
# def extract_orb_frame(frame, scale=0.33, nfeatures=150):
#     if scale < 1.0:
#         h, w = frame.shape[:2]
#         frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     orb = cv2.ORB_create(nfeatures=nfeatures)
#     kps, des = orb.detectAndCompute(gray, None)
#     keypoints = [kp.pt for kp in kps] if kps is not None else []
#     return keypoints, des.tolist() if des is not None else []

# def extract_orb_all_fast(video_path, scale=0.33, nfeatures=150, step=1, max_workers=4):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = []
#     for i in range(0, frame_count, step):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         frames.append(frame if ret else None)
#     cap.release()
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(
#             lambda f: extract_orb_frame(f, scale, nfeatures) if f is not None else ([], []), frames
#         ))
#     all_keypoints = [r[0] for r in results]
#     all_des = [r[1] for r in results]
#     return all_keypoints, all_des

# # ----------- 동영상 업로드/리스트 (/videos) -----------
# @app.route('/videos', methods=['GET', 'POST'])
# def video_list():
#     if request.method == 'POST':
#         file = request.files.get('video')
#         if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(save_path)
#             thumb_path = os.path.join(THUMB_FOLDER, filename + '.jpg')
#             extract_thumbnail(save_path, thumb_path)
#         return redirect(url_for('video_list'))

#     videos = []
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             info = get_video_info(path)
#             info['thumb'] = thumb
#             videos.append(info)
#     return render_template('videos.html', videos=videos)

# @app.route('/frame/<filename>/<int:index>')
# def get_frame(filename, index):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     success, frame = cap.read()
#     cap.release()
#     if not success:
#         return "Frame not found", 404
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# # ---- 전체 프레임 ORB 추출 (속도 최적화 적용) ----
# @app.route('/extract_orb_all/<filename>')
# def extract_orb_all(filename):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     all_keypoints, all_des = extract_orb_all_fast(video_path, scale=0.33, nfeatures=150, step=1, max_workers=4)
#     return jsonify({'all_keypoints': all_keypoints, 'all_des': all_des})

# @app.route('/extract_orb/<filename>/<int:frame_idx>')
# def extract_orb(filename, frame_idx):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return jsonify({'keypoints': []})
#     keypoints, _ = extract_orb_frame(frame, scale=0.33, nfeatures=150)
#     return jsonify({'keypoints': keypoints})

# # ----------- 위험구역 트래킹(프레임별 polygon 반환, robust) --------
# @app.route('/track_zone_history', methods=['POST'])
# def track_zone_history():
#     data = request.get_json()
#     filename = data['filename']
#     from_idx = int(data['from_frame'])
#     to_idx = int(data['to_frame'])
#     polygon = data['polygon']   # [[x,y], ...]
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     for i in range(from_idx, to_idx+1):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     if not frames or len(polygon) < 3:
#         return jsonify({"history": []})
#     poly_hist = track_polygon_history_robust(frames, polygon)
#     frame_polys = [{"frame": from_idx + i, "points": poly_hist[i]} for i in range(len(poly_hist))]
#     return jsonify({"history": frame_polys})

# @app.route('/frame_overlay/<filename>/<int:index>', methods=['POST'])
# def get_frame_overlay(filename, index):
#     data = request.get_json()
#     polygon = data.get("polygon", [])
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     pts = np.array(polygon, np.int32)
#     if pts.shape[0] >= 3:
#         cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
#         cv2.fillPoly(frame, [pts], (0, 0, 255, 40))
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# @app.route('/save_zone', methods=['POST'])
# def save_zone():
#     data = request.get_json()
#     filename = data['filename']
#     zones = data['zones']
#     orb_all = data.get('orb_all', None)  # 전체 프레임 ORB도 저장
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     meta_info = {"zones": []}
#     if orb_all:
#         meta_info["orb_all"] = orb_all
#     for zone_id, zone_data in zones.items():
#         meta_info["zones"].append({
#             "id": zone_id,
#             "frame": zone_data['frame'],
#             "points": zone_data['points']
#         })
#     with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
#         json.dump(meta_info, f)
#     return "Saved", 200

# @app.route('/zones')
# def danger_zone_list():
#     videos = []
#     selected_video = request.args.get('video')
#     frame_count = 0
#     zones = {}
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             info = get_video_info(path)
#             info['thumb'] = thumb
#             videos.append(info)
#     if selected_video and os.path.exists(os.path.join(UPLOAD_FOLDER, selected_video)):
#         video_path = os.path.join(UPLOAD_FOLDER, selected_video)
#         cap = cv2.VideoCapture(video_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         project_dir = os.path.join(PROJECT_FOLDER, selected_video)
#         meta_path = os.path.join(project_dir, 'meta.json')
#         if os.path.exists(meta_path):
#             with open(meta_path, 'r') as f:
#                 meta = json.load(f)
#             for zone in meta.get('zones', []):
#                 zones[zone['id']] = {'points': zone['points'], 'frame': zone['frame']}
#     return render_template('danger_zone_list.html',
#                            videos=videos,
#                            selected_video=selected_video,
#                            frame_count=frame_count,
#                            zones=zones)

# rtsp_url = ""
# @app.route("/", methods=["GET", "POST"])
# def index():
#     global rtsp_url
#     if request.method == "POST":
#         rtsp_url = request.form.get("rtsp_url", "")
#     projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
#     return render_template("index.html", rtsp_url=rtsp_url, projects=projects)

# @app.route("/video_feed")
# def video_feed():
#     global rtsp_url
#     def gen():
#         cap = cv2.VideoCapture(rtsp_url)
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             _, jpeg = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#         cap.release()
#     return send_file(io.BytesIO(), mimetype="multipart/x-mixed-replace; boundary=frame") \
#         if not rtsp_url else app.response_class(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/get_project_overlay")
# def get_project_overlay():
#     project = request.args.get("project")
#     meta_path = os.path.join(PROJECT_FOLDER, project, "meta.json")
#     if not os.path.exists(meta_path):
#         return jsonify({"polygon":[],"match_contour":[],"match_frame":-1,"match_score":0})
#     with open(meta_path, "r") as f:
#         meta = json.load(f)
#     zones = meta.get("zones", [])
#     polygon = zones[0]["points"] if zones else []
#     resp = {
#         "polygon": polygon,
#         "match_contour": [],
#         "match_frame": zones[0]["frame"] if zones else -1,
#         "match_score": 1.0
#     }
#     return jsonify(resp)

# @app.route("/rtsp_match_overlay", methods=["POST"])
# def rtsp_match_overlay():
#     req = request.get_json()
#     rtsp_url = req.get("rtsp_url")
#     project = req.get("project")
#     meta_path = os.path.join(PROJECT_FOLDER, project, "meta.json")
#     if not (rtsp_url and os.path.exists(meta_path)):
#         return jsonify({})
#     cap = cv2.VideoCapture(rtsp_url)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return jsonify({})
#     gray_rtsp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     orb = cv2.ORB_create()
#     kps_rtsp, des_rtsp = orb.detectAndCompute(gray_rtsp, None)
#     if des_rtsp is None or len(kps_rtsp)==0:
#         return jsonify({})
#     with open(meta_path, "r") as f:
#         meta = json.load(f)
#     if "orb_all" in meta and "all_des" in meta["orb_all"]:
#         all_des = meta["orb_all"]["all_des"]
#         all_kps = meta["orb_all"]["all_keypoints"]
#     else:
#         project_vid = os.path.join(UPLOAD_FOLDER, project)
#         cap2 = cv2.VideoCapture(project_vid)
#         frame_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
#         all_des, all_kps = [], []
#         for i in range(frame_count):
#             cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, f2 = cap2.read()
#             if not ret:
#                 all_des.append([])
#                 all_kps.append([])
#                 continue
#             gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#             kps2, des2 = orb.detectAndCompute(gray2, None)
#             all_kps.append([kp.pt for kp in kps2] if kps2 else [])
#             all_des.append(des2.tolist() if des2 is not None else [])
#         cap2.release()
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     best_score, best_idx = 0, -1
#     for i, des2 in enumerate(all_des):
#         if des2 is None or len(des2)==0:
#             continue
#         des2_np = np.array(des2, dtype=np.uint8)
#         matches = bf.match(des_rtsp, des2_np)
#         score = len(matches)
#         if score > best_score:
#             best_score = score
#             best_idx = i
#     if best_idx < 0:
#         return jsonify({})
#     project_vid = os.path.join(UPLOAD_FOLDER, project)
#     cap = cv2.VideoCapture(project_vid)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
#     ret, best_frame = cap.read()
#     cap.release()
#     matched_zones = []
#     for zone in meta.get("zones", []):
#         if int(zone["frame"]) == best_idx:
#             matched_zones.append(zone["points"])
#     _, jpg = cv2.imencode('.jpg', best_frame)
#     jpg_b64 = base64.b64encode(jpg).decode('utf-8')
#     return jsonify({
#         'frame': jpg_b64,
#         'match_idx': best_idx,
#         'score': float(best_score) / max(len(kps_rtsp), 1),
#         'zones': matched_zones
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

# import os
# import cv2
# import numpy as np
# import base64
# import io
# import json
# from concurrent.futures import ThreadPoolExecutor
# from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/uploads'
# THUMB_FOLDER = 'static/thumbs'
# PROJECT_FOLDER = 'projects'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(THUMB_FOLDER, exist_ok=True)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['THUMB_FOLDER'] = THUMB_FOLDER
# app.config['PROJECT_FOLDER'] = PROJECT_FOLDER

# # ================= Polygon Tracker Class & Robust Tracking ===================
# class PolygonTracker:
#     def __init__(self):
#         self.points = []
#         self.poly_complete = False
#         self.tracking_failed = False
#         self.prev_gray = None
#         self.prev_points = None
#         self.hist = None
#         self.prev_global_pts = None

#     def initialize(self, frame, points):
#         self.points = [tuple(pt) for pt in points]
#         self.poly_complete = True
#         self.tracking_failed = False
#         self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.prev_points = np.array(self.points, dtype=np.float32)
#         mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
#         cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
#         cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
#         self.prev_global_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

#     def track(self, frame):
#         if not self.poly_complete or self.tracking_failed:
#             return self.points
#         next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         h, w = next_gray.shape[:2]
#         prev_pts = np.array(self.points, np.float32).reshape(-1, 1, 2)
#         next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, prev_pts, None)
#         pts_valid = st is not None and st.sum() >= len(self.points) * 0.6

#         # Optical flow
#         if next_pts is not None and pts_valid:
#             pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in next_pts]
#             self.points = pts_new
#             self.prev_points = np.array(self.points, dtype=np.float32)
#             self.prev_gray = next_gray
#             return self.points

#         # Affine fallback
#         if self.prev_global_pts is not None and len(self.prev_global_pts) >= 3:
#             curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, self.prev_global_pts, None)
#             st = st.reshape(-1)
#             prev_g = self.prev_global_pts[st == 1]
#             curr_g = curr_global_pts[st == 1]
#             if len(prev_g) >= 3 and len(curr_g) >= 3:
#                 M, inliers = cv2.estimateAffinePartial2D(prev_g, curr_g)
#                 if M is not None:
#                     poly_np = np.array(self.points, np.float32).reshape(-1, 1, 2)
#                     poly_trans = cv2.transform(poly_np, M)
#                     pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in poly_trans]
#                     self.points = pts_new
#                     self.prev_points = np.array(self.points, dtype=np.float32)
#                     self.prev_gray = next_gray
#                     self.prev_global_pts = curr_g.reshape(-1, 1, 2)
#                     return self.points
#             self.prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

#         # MeanShift fallback
#         if self.hist is not None and len(self.points) >= 3:
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             pts = np.array(self.points, dtype=np.int32)
#             x, y, w_box, h_box = cv2.boundingRect(pts)
#             back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
#             term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#             _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
#             dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
#             dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
#             self.points = [(pt[0]+dx, pt[1]+dy) for pt in self.points]
#             self.prev_gray = next_gray
#             return self.points

#         # Tracking failed
#         self.tracking_failed = True
#         self.prev_gray = next_gray
#         return self.points

# # 세션별 Polygon Tracker (zone_id별로 여러개 동시 관리)
# TRACKING_SESSIONS = {}

# # ====================== ORB 추출 및 영상 정보 ========================
# def get_video_info(filepath):
#     cap = cv2.VideoCapture(filepath)
#     if not cap.isOpened():
#         return {'name': os.path.basename(filepath), 'duration': 0, 'resolution': 'N/A'}
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     duration = int(frame_count / fps) if fps > 0 else 0
#     cap.release()
#     return {
#         'name': os.path.basename(filepath),
#         'duration': duration,
#         'resolution': f"{width}x{height}"
#     }

# def extract_thumbnail(filepath, thumb_path):
#     cap = cv2.VideoCapture(filepath)
#     success, frame = cap.read()
#     if success:
#         cv2.imwrite(thumb_path, frame)
#     cap.release()

# def extract_orb_frame(frame, scale=0.33, nfeatures=150):
#     if scale < 1.0:
#         h, w = frame.shape[:2]
#         frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     orb = cv2.ORB_create(nfeatures=nfeatures)
#     kps, des = orb.detectAndCompute(gray, None)
#     keypoints = [kp.pt for kp in kps] if kps is not None else []
#     return keypoints, des.tolist() if des is not None else []

# def extract_orb_all_fast(video_path, scale=0.33, nfeatures=150, step=1, max_workers=4):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = []
#     for i in range(0, frame_count, step):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         frames.append(frame if ret else None)
#     cap.release()
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(
#             lambda f: extract_orb_frame(f, scale, nfeatures) if f is not None else ([], []), frames
#         ))
#     all_keypoints = [r[0] for r in results]
#     all_des = [r[1] for r in results]
#     return all_keypoints, all_des

# # ======================== ROUTES ===========================
# @app.route('/videos', methods=['GET', 'POST'])
# def video_list():
#     if request.method == 'POST':
#         file = request.files.get('video')
#         if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(save_path)
#             thumb_path = os.path.join(THUMB_FOLDER, filename + '.jpg')
#             extract_thumbnail(save_path, thumb_path)
#         return redirect(url_for('video_list'))

#     videos = []
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             info = get_video_info(path)
#             info['thumb'] = thumb
#             videos.append(info)
#     return render_template('videos.html', videos=videos)

# @app.route('/frame/<filename>/<int:index>')
# def get_frame(filename, index):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     success, frame = cap.read()
#     cap.release()
#     if not success:
#         return "Frame not found", 404
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# @app.route('/extract_orb_all/<filename>')
# def extract_orb_all(filename):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     all_keypoints, all_des = extract_orb_all_fast(video_path, scale=0.33, nfeatures=150, step=1, max_workers=4)
#     return jsonify({'all_keypoints': all_keypoints, 'all_des': all_des})

# @app.route('/extract_orb/<filename>/<int:frame_idx>')
# def extract_orb(filename, frame_idx):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return jsonify({'keypoints': []})
#     keypoints, _ = extract_orb_frame(frame, scale=0.33, nfeatures=150)
#     return jsonify({'keypoints': keypoints})

# @app.route('/track_zone_history', methods=['POST'])
# def track_zone_history():
#     data = request.get_json()
#     filename = data['filename']
#     from_idx = int(data['from_frame'])
#     to_idx = int(data['to_frame'])
#     polygon = data['polygon']   # [[x,y], ...]
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     for i in range(from_idx, to_idx+1):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     if not frames or len(polygon) < 3:
#         return jsonify({"history": []})
#     tracker = PolygonTracker()
#     tracker.initialize(frames[0], polygon)
#     poly_hist = [polygon]
#     for i in range(1, len(frames)):
#         pts = tracker.track(frames[i])
#         poly_hist.append(pts[:])
#     frame_polys = [{"frame": from_idx + i, "points": poly_hist[i]} for i in range(len(poly_hist))]
#     return jsonify({"history": frame_polys})

# @app.route('/frame_overlay/<filename>/<int:index>', methods=['POST'])
# def get_frame_overlay(filename, index):
#     data = request.get_json()
#     polygon = data.get("polygon", [])
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     pts = np.array(polygon, np.int32)
#     if pts.shape[0] >= 3:
#         cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [pts], (0, 0, 255))
#         frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# @app.route('/save_zone', methods=['POST'])
# def save_zone():
#     data = request.get_json()
#     filename = data['filename']
#     zones = data['zones']
#     orb_all = data.get('orb_all', None)
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     meta_info = {"zones": []}
#     if orb_all:
#         meta_info["orb_all"] = orb_all
#     for zone_id, zone_data in zones.items():
#         meta_info["zones"].append({
#             "id": zone_id,
#             "frame": zone_data['frame'],
#             "points": zone_data['points']
#         })
#     with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
#         json.dump(meta_info, f)
#     return "Saved", 200

# @app.route('/zones')
# def danger_zone_list():
#     videos = []
#     selected_video = request.args.get('video')
#     frame_count = 0
#     zones = {}
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             info = get_video_info(path)
#             info['thumb'] = thumb
#             videos.append(info)
#     if selected_video and os.path.exists(os.path.join(UPLOAD_FOLDER, selected_video)):
#         video_path = os.path.join(UPLOAD_FOLDER, selected_video)
#         cap = cv2.VideoCapture(video_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         project_dir = os.path.join(PROJECT_FOLDER, selected_video)
#         meta_path = os.path.join(project_dir, 'meta.json')
#         if os.path.exists(meta_path):
#             with open(meta_path, 'r') as f:
#                 meta = json.load(f)
#             for zone in meta.get('zones', []):
#                 zones[zone['id']] = {'points': zone['points'], 'frame': zone['frame']}
#     return render_template('danger_zone_list.html',
#                            videos=videos,
#                            selected_video=selected_video,
#                            frame_count=frame_count,
#                            zones=zones)

# rtsp_url = ""
# @app.route("/", methods=["GET", "POST"])
# def index():
#     global rtsp_url
#     if request.method == "POST":
#         rtsp_url = request.form.get("rtsp_url", "")
#     projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
#     return render_template("index.html", rtsp_url=rtsp_url, projects=projects)

# @app.route("/video_feed")
# def video_feed():
#     global rtsp_url
#     def gen():
#         cap = cv2.VideoCapture(rtsp_url)
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             _, jpeg = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#         cap.release()
#     return send_file(io.BytesIO(), mimetype="multipart/x-mixed-replace; boundary=frame") \
#         if not rtsp_url else app.response_class(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/get_project_overlay")
# def get_project_overlay():
#     project = request.args.get("project")
#     meta_path = os.path.join(PROJECT_FOLDER, project, "meta.json")
#     if not os.path.exists(meta_path):
#         return jsonify({"polygon":[],"match_contour":[],"match_frame":-1,"match_score":0})
#     with open(meta_path, "r") as f:
#         meta = json.load(f)
#     zones = meta.get("zones", [])
#     polygon = zones[0]["points"] if zones else []
#     resp = {
#         "polygon": polygon,
#         "match_contour": [],
#         "match_frame": zones[0]["frame"] if zones else -1,
#         "match_score": 1.0
#     }
#     return jsonify(resp)

# @app.route("/rtsp_match_overlay", methods=["POST"])
# def rtsp_match_overlay():
#     req = request.get_json()
#     rtsp_url = req.get("rtsp_url")
#     project = req.get("project")
#     meta_path = os.path.join(PROJECT_FOLDER, project, "meta.json")
#     if not (rtsp_url and os.path.exists(meta_path)):
#         return jsonify({})
#     cap = cv2.VideoCapture(rtsp_url)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return jsonify({})
#     gray_rtsp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     orb = cv2.ORB_create()
#     kps_rtsp, des_rtsp = orb.detectAndCompute(gray_rtsp, None)
#     if des_rtsp is None or len(kps_rtsp)==0:
#         return jsonify({})
#     with open(meta_path, "r") as f:
#         meta = json.load(f)
#     if "orb_all" in meta and "all_des" in meta["orb_all"]:
#         all_des = meta["orb_all"]["all_des"]
#         all_kps = meta["orb_all"]["all_keypoints"]
#     else:
#         project_vid = os.path.join(UPLOAD_FOLDER, project)
#         cap2 = cv2.VideoCapture(project_vid)
#         frame_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
#         all_des, all_kps = [], []
#         for i in range(frame_count):
#             cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, f2 = cap2.read()
#             if not ret:
#                 all_des.append([])
#                 all_kps.append([])
#                 continue
#             gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
#             kps2, des2 = orb.detectAndCompute(gray2, None)
#             all_kps.append([kp.pt for kp in kps2] if kps2 else [])
#             all_des.append(des2.tolist() if des2 is not None else [])
#         cap2.release()
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     best_score, best_idx = 0, -1
#     for i, des2 in enumerate(all_des):
#         if des2 is None or len(des2)==0:
#             continue
#         des2_np = np.array(des2, dtype=np.uint8)
#         matches = bf.match(des_rtsp, des2_np)
#         score = len(matches)
#         if score > best_score:
#             best_score = score
#             best_idx = i
#     if best_idx < 0:
#         return jsonify({})
#     project_vid = os.path.join(UPLOAD_FOLDER, project)
#     cap = cv2.VideoCapture(project_vid)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
#     ret, best_frame = cap.read()
#     cap.release()
#     matched_zones = []
#     for zone in meta.get("zones", []):
#         if int(zone["frame"]) == best_idx:
#             matched_zones.append(zone["points"])
#     _, jpg = cv2.imencode('.jpg', best_frame)
#     jpg_b64 = base64.b64encode(jpg).decode('utf-8')
#     return jsonify({
#         'frame': jpg_b64,
#         'match_idx': best_idx,
#         'score': float(best_score) / max(len(kps_rtsp), 1),
#         'zones': matched_zones
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

# import os
# import cv2
# import numpy as np
# import base64
# import io
# import json
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/uploads'
# THUMB_FOLDER = 'static/thumbs'
# PROJECT_FOLDER = 'projects'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(THUMB_FOLDER, exist_ok=True)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['THUMB_FOLDER'] = THUMB_FOLDER
# app.config['PROJECT_FOLDER'] = PROJECT_FOLDER

# # ==== 글로벌 캐시 구조 ====
# rtsp_url = ""
# selected_project = None
# cached_orb_data = {}
# cached_zones = {}
# current_best_idx = -1
# current_best_score = 0
# BEST_SCORE_THRESHOLD = 30
# RESET_SCORE_RATIO = 0.6

# # ================= Polygon Tracker Class & Robust Tracking ===================
# class PolygonTracker:
#     def __init__(self):
#         self.points = []
#         self.poly_complete = False
#         self.tracking_failed = False
#         self.prev_gray = None
#         self.prev_points = None
#         self.hist = None
#         self.prev_global_pts = None

#     def initialize(self, frame, points):
#         self.points = [tuple(pt) for pt in points]
#         self.poly_complete = True
#         self.tracking_failed = False
#         self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.prev_points = np.array(self.points, dtype=np.float32)
#         mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
#         cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
#         cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
#         self.prev_global_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

#     def track(self, frame):
#         if not self.poly_complete or self.tracking_failed:
#             return self.points
#         next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         h, w = next_gray.shape[:2]
#         prev_pts = np.array(self.points, np.float32).reshape(-1, 1, 2)
#         next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, prev_pts, None)
#         pts_valid = st is not None and st.sum() >= len(self.points) * 0.6
#         if next_pts is not None and pts_valid:
#             pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in next_pts]
#             self.points = pts_new
#             self.prev_points = np.array(self.points, dtype=np.float32)
#             self.prev_gray = next_gray
#             return self.points
#         # Affine fallback
#         if self.prev_global_pts is not None and len(self.prev_global_pts) >= 3:
#             curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, self.prev_global_pts, None)
#             st = st.reshape(-1)
#             prev_g = self.prev_global_pts[st == 1]
#             curr_g = curr_global_pts[st == 1]
#             if len(prev_g) >= 3 and len(curr_g) >= 3:
#                 M, _ = cv2.estimateAffinePartial2D(prev_g, curr_g)
#                 if M is not None:
#                     poly_np = np.array(self.points, np.float32).reshape(-1, 1, 2)
#                     poly_trans = cv2.transform(poly_np, M)
#                     pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in poly_trans]
#                     self.points = pts_new
#                     self.prev_points = np.array(self.points, dtype=np.float32)
#                     self.prev_gray = next_gray
#                     self.prev_global_pts = curr_g.reshape(-1, 1, 2)
#                     return self.points
#             self.prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)
#         # MeanShift fallback
#         if self.hist is not None and len(self.points) >= 3:
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             pts = np.array(self.points, dtype=np.int32)
#             x, y, w_box, h_box = cv2.boundingRect(pts)
#             back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
#             term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#             _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
#             dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
#             dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
#             self.points = [(pt[0]+dx, pt[1]+dy) for pt in self.points]
#             self.prev_gray = next_gray
#             return self.points
#         self.tracking_failed = True
#         self.prev_gray = next_gray
#         return self.points

# # ======= ORB 캐시/로드/전체추출(pkl) =======
# def extract_orb_all_fast(video_path, scale=0.33, nfeatures=150):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     orb = cv2.ORB_create(nfeatures=nfeatures)
#     all_keypoints, all_descriptors = [], []
#     for i in range(0, frame_count, 1):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             all_keypoints.append([])
#             all_descriptors.append(np.zeros((0,32), dtype=np.uint8))
#             continue
#         if scale < 1.0:
#             h, w = frame.shape[:2]
#             frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         kps, des = orb.detectAndCompute(gray, None)
#         all_keypoints.append([kp.pt for kp in kps] if kps is not None else [])
#         all_descriptors.append(des.astype(np.uint8) if des is not None else np.zeros((0,32), dtype=np.uint8))
#     cap.release()
#     return all_keypoints, all_descriptors

# def load_project_cache(project):
#     global cached_orb_data, cached_zones
#     orb_path = os.path.join(PROJECT_FOLDER, project, 'orb_data.pkl')
#     meta_path = os.path.join(PROJECT_FOLDER, project, 'meta.json')
#     cached_orb_data.clear()
#     if os.path.exists(orb_path):
#         with open(orb_path, 'rb') as f:
#             cached_orb_data.update(pickle.load(f))
#     cached_zones.clear()
#     if os.path.exists(meta_path):
#         with open(meta_path, 'r') as f:
#             meta = json.load(f)
#         for zone in meta.get('zones', []):
#             cached_zones[zone['frame']] = zone['points']

# @app.route('/extract_orb_all/<filename>')
# def extract_orb_all(filename):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     keypoints, descriptors = extract_orb_all_fast(video_path)
#     orb_data = {idx: {'keypoints': kp, 'descriptors': des} for idx, (kp, des) in enumerate(zip(keypoints, descriptors))}
#     with open(os.path.join(project_dir, 'orb_data.pkl'), 'wb') as f:
#         pickle.dump(orb_data, f)
#     return jsonify({'status': 'ok', 'count': len(orb_data)})

# # ========== Danger Zone 관리, 저장, 삭제, 수정 ===========
# @app.route('/save_zone', methods=['POST'])
# def save_zone():
#     data = request.get_json()
#     filename = data['filename']
#     zones = data['zones']
#     orb_all = data.get('orb_all', None)
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     meta_info = {"zones": []}
#     if orb_all:
#         meta_info["orb_all"] = orb_all
#     for zone_id, zone_data in zones.items():
#         meta_info["zones"].append({
#             "id": zone_id,
#             "frame": zone_data['frame'],
#             "points": zone_data['points']
#         })
#     with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
#         json.dump(meta_info, f)
#     load_project_cache(filename)
#     return "Saved", 200

# @app.route('/delete_zone/<filename>/<zone_id>', methods=['POST'])
# def delete_zone(filename, zone_id):
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     meta_path = os.path.join(project_dir, 'meta.json')
#     if not os.path.exists(meta_path):
#         return 'Meta not found', 404
#     with open(meta_path, 'r') as f:
#         meta = json.load(f)
#     zones = meta.get("zones", [])
#     zones = [z for z in zones if str(z["id"]) != str(zone_id)]
#     meta["zones"] = zones
#     with open(meta_path, 'w') as f:
#         json.dump(meta, f)
#     load_project_cache(filename)
#     return "Deleted", 200

# @app.route('/edit_zone/<filename>/<zone_id>', methods=['POST'])
# def edit_zone(filename, zone_id):
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     meta_path = os.path.join(project_dir, 'meta.json')
#     data = request.get_json()
#     new_points = data.get('points')
#     new_frame = data.get('frame')
#     if not os.path.exists(meta_path):
#         return 'Meta not found', 404
#     with open(meta_path, 'r') as f:
#         meta = json.load(f)
#     found = False
#     for zone in meta.get("zones", []):
#         if str(zone["id"]) == str(zone_id):
#             if new_points: zone["points"] = new_points
#             if new_frame is not None: zone["frame"] = new_frame
#             found = True
#     if found:
#         with open(meta_path, 'w') as f:
#             json.dump(meta, f)
#         load_project_cache(filename)
#         return "Edited", 200
#     return "Not found", 404

# # ===== 썸네일/동영상 리스트 =========
# def extract_thumbnail(filepath, thumb_path):
#     cap = cv2.VideoCapture(filepath)
#     ret, frame = cap.read()
#     if ret:
#         cv2.imwrite(thumb_path, frame)
#     cap.release()

# @app.route('/videos', methods=['GET', 'POST'])
# def video_list():
#     if request.method == 'POST':
#         file = request.files.get('video')
#         if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(save_path)
#             thumb_path = os.path.join(THUMB_FOLDER, filename + '.jpg')
#             extract_thumbnail(save_path, thumb_path)
#         return redirect(url_for('video_list'))
#     videos = []
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             videos.append({'name': fname, 'thumb': thumb})
#     return render_template('videos.html', videos=videos)

# @app.route('/frame/<filename>/<int:index>')
# def get_frame(filename, index):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# # ===== Danger Zone overlay, ORB, track, etc 유지 =====
# @app.route('/frame_overlay/<filename>/<int:index>', methods=['POST'])
# def get_frame_overlay(filename, index):
#     data = request.get_json()
#     polygon = data.get("polygon", [])
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     if polygon and len(polygon) >= 3:
#         pts = np.array(polygon, np.int32)
#         cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [pts], (0, 0, 255))
#         frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# # ============= RTSP/ORB 실시간 매칭 + DangerZone Overlay 캐시방식 =============
# @app.route('/rtsp_match/<project>')
# def rtsp_match(project):
#     global selected_project, current_best_idx, current_best_score
#     selected_project = project
#     load_project_cache(project)
#     current_best_idx = -1
#     current_best_score = 0
#     return jsonify({'status': 'ok'})

# @app.route('/set_rtsp', methods=['POST'])
# def set_rtsp():
#     global rtsp_url
#     rtsp_url = request.json.get('rtsp_url')
#     return jsonify({'status':'ok'})

# @app.route("/video_feed")
# def video_feed():
#     def generate_rtsp_stream():
#         global cached_orb_data, cached_zones, current_best_idx, current_best_score, rtsp_url
#         cap = cv2.VideoCapture(rtsp_url)
#         orb = cv2.ORB_create()
#         matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: break
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             kp_live, des_live = orb.detectAndCompute(gray, None)
#             best_idx, best_score = -1, 0
#             if selected_project and current_best_idx != -1 and des_live is not None:
#                 des_saved = cached_orb_data.get(current_best_idx, {}).get('descriptors', np.zeros((0,32),np.uint8))
#                 if len(des_saved) > 0:
#                     matches = matcher.match(des_live, des_saved)
#                     best_score = len(matches)
#                     best_idx = current_best_idx
#             if (current_best_idx == -1 or
#                 best_score < int(current_best_score * RESET_SCORE_RATIO) or
#                 best_score < BEST_SCORE_THRESHOLD):
#                 tmp_best_idx, tmp_best_score = -1, 0
#                 for idx, data in cached_orb_data.items():
#                     des_saved = data['descriptors']
#                     if des_live is None or des_saved is None or len(des_saved) == 0:
#                         continue
#                     matches = matcher.match(des_live, des_saved)
#                     if len(matches) > tmp_best_score:
#                         tmp_best_score = len(matches)
#                         tmp_best_idx = idx
#                 best_idx, best_score = tmp_best_idx, tmp_best_score
#                 current_best_idx = best_idx
#                 current_best_score = best_score
#             # Danger zone overlay
#             if best_score >= BEST_SCORE_THRESHOLD and selected_project and best_idx in cached_zones:
#                 poly = np.array(cached_zones[best_idx], np.int32)
#                 cv2.polylines(frame, [poly], True, (0,0,255), 2)
#                 overlay = frame.copy()
#                 cv2.fillPoly(overlay, [poly], (0,0,255))
#                 frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
#                 cv2.putText(frame, f"Matched Frame: {best_idx} (Score: {best_score})", (10, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         cap.release()
#     return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # ========== 위험구역 관리 페이지 ==========
# @app.route('/zones')
# def danger_zone_list():
#     videos = []
#     selected_video = request.args.get('video')
#     frame_count = 0
#     zones = {}
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             videos.append({'name': fname, 'thumb': thumb})
#     if selected_video and os.path.exists(os.path.join(UPLOAD_FOLDER, selected_video)):
#         video_path = os.path.join(UPLOAD_FOLDER, selected_video)
#         cap = cv2.VideoCapture(video_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         project_dir = os.path.join(PROJECT_FOLDER, selected_video)
#         meta_path = os.path.join(project_dir, 'meta.json')
#         if os.path.exists(meta_path):
#             with open(meta_path, 'r') as f:
#                 meta = json.load(f)
#             for zone in meta.get('zones', []):
#                 zones[zone['id']] = {'points': zone['points'], 'frame': zone['frame']}
#     return render_template('danger_zone_list.html',
#                            videos=videos,
#                            selected_video=selected_video,
#                            frame_count=frame_count,
#                            zones=zones)

# # ========= 인덱스 ===========
# @app.route("/", methods=["GET", "POST"])
# def index():
#     global rtsp_url
#     if request.method == "POST":
#         rtsp_url = request.form.get("rtsp_url", "")
#     projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
#     return render_template("index.html", rtsp_url=rtsp_url, projects=projects)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
# import os
# import cv2
# import numpy as np
# import base64
# import io
# import json
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/uploads'
# THUMB_FOLDER = 'static/thumbs'
# PROJECT_FOLDER = 'projects'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(THUMB_FOLDER, exist_ok=True)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['THUMB_FOLDER'] = THUMB_FOLDER
# app.config['PROJECT_FOLDER'] = PROJECT_FOLDER

# # ==== 글로벌 캐시 구조 ====
# rtsp_url = ""
# selected_project = None
# cached_orb_data = {}
# cached_zones = {}
# current_best_idx = -1
# current_best_score = 0
# BEST_SCORE_THRESHOLD = 30
# RESET_SCORE_RATIO = 0.6
# FAST_SEARCH_WINDOW = 10

# # ================= Polygon Tracker ===================
# class PolygonTracker:
#     # ... (생략, 기존과 동일) ...

#     def __init__(self):
#         self.points = []
#         self.poly_complete = False
#         self.tracking_failed = False
#         self.prev_gray = None
#         self.prev_points = None
#         self.hist = None
#         self.prev_global_pts = None

#     def initialize(self, frame, points):
#         self.points = [tuple(pt) for pt in points]
#         self.poly_complete = True
#         self.tracking_failed = False
#         self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.prev_points = np.array(self.points, dtype=np.float32)
#         mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
#         cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
#         cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
#         self.prev_global_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

#     def track(self, frame):
#         if not self.poly_complete or self.tracking_failed:
#             return self.points
#         next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         h, w = next_gray.shape[:2]
#         prev_pts = np.array(self.points, np.float32).reshape(-1, 1, 2)
#         next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, prev_pts, None)
#         pts_valid = st is not None and st.sum() >= len(self.points) * 0.6
#         if next_pts is not None and pts_valid:
#             pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in next_pts]
#             self.points = pts_new
#             self.prev_points = np.array(self.points, dtype=np.float32)
#             self.prev_gray = next_gray
#             return self.points
#         # Affine fallback
#         if self.prev_global_pts is not None and len(self.prev_global_pts) >= 3:
#             curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, self.prev_global_pts, None)
#             st = st.reshape(-1)
#             prev_g = self.prev_global_pts[st == 1]
#             curr_g = curr_global_pts[st == 1]
#             if len(prev_g) >= 3 and len(curr_g) >= 3:
#                 M, _ = cv2.estimateAffinePartial2D(prev_g, curr_g)
#                 if M is not None:
#                     poly_np = np.array(self.points, np.float32).reshape(-1, 1, 2)
#                     poly_trans = cv2.transform(poly_np, M)
#                     pts_new = [(float(np.clip(pt[0][0], 0, w-1)), float(np.clip(pt[0][1], 0, h-1))) for pt in poly_trans]
#                     self.points = pts_new
#                     self.prev_points = np.array(self.points, dtype=np.float32)
#                     self.prev_gray = next_gray
#                     self.prev_global_pts = curr_g.reshape(-1, 1, 2)
#                     return self.points
#             self.prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)
#         # MeanShift fallback
#         if self.hist is not None and len(self.points) >= 3:
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             pts = np.array(self.points, dtype=np.int32)
#             x, y, w_box, h_box = cv2.boundingRect(pts)
#             back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
#             term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#             _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
#             dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
#             dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
#             self.points = [(pt[0]+dx, pt[1]+dy) for pt in self.points]
#             self.prev_gray = next_gray
#             return self.points
#         self.tracking_failed = True
#         self.prev_gray = next_gray
#         return self.points

# # ========== ORB 추출/저장/캐시 ==========
# def extract_orb_all_fast(video_path, scale=0.33, nfeatures=150):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     orb = cv2.ORB_create(nfeatures=nfeatures)
#     all_keypoints, all_descriptors = [], []
#     for i in range(0, frame_count, 1):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             all_keypoints.append([])
#             all_descriptors.append(np.zeros((0,32), dtype=np.uint8))
#             continue
#         if scale < 1.0:
#             h, w = frame.shape[:2]
#             frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         kps, des = orb.detectAndCompute(gray, None)
#         all_keypoints.append([kp.pt for kp in kps] if kps is not None else [])
#         all_descriptors.append(des.astype(np.uint8) if des is not None else np.zeros((0,32), dtype=np.uint8))
#     cap.release()
#     return all_keypoints, all_descriptors

# def load_project_cache(project):
#     global cached_orb_data, cached_zones
#     orb_path = os.path.join(PROJECT_FOLDER, project, 'orb_data.pkl')
#     meta_path = os.path.join(PROJECT_FOLDER, project, 'meta.json')
#     cached_orb_data.clear()
#     if os.path.exists(orb_path):
#         with open(orb_path, 'rb') as f:
#             cached_orb_data.update(pickle.load(f))
#     cached_zones.clear()
#     if os.path.exists(meta_path):
#         with open(meta_path, 'r') as f:
#             meta = json.load(f)
#         for zone in meta.get('zones', []):
#             cached_zones[zone['frame']] = zone['points']

# @app.route('/extract_orb_all/<filename>')
# def extract_orb_all(filename):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     keypoints, descriptors = extract_orb_all_fast(video_path)
#     orb_data = {idx: {'keypoints': kp, 'descriptors': des} for idx, (kp, des) in enumerate(zip(keypoints, descriptors))}
#     with open(os.path.join(project_dir, 'orb_data.pkl'), 'wb') as f:
#         pickle.dump(orb_data, f)
#     return jsonify({'status': 'ok', 'count': len(orb_data)})

# # ===== Danger Zone 관리, 저장, 삭제, 수정 =====
# @app.route('/save_zone', methods=['POST'])
# def save_zone():
#     data = request.get_json()
#     filename = data['filename']
#     zones = data['zones']
#     orb_all = data.get('orb_all', None)
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     os.makedirs(project_dir, exist_ok=True)
#     meta_info = {"zones": []}
#     if orb_all:
#         meta_info["orb_all"] = orb_all
#     for zone_id, zone_data in zones.items():
#         meta_info["zones"].append({
#             "id": zone_id,
#             "frame": zone_data['frame'],
#             "points": zone_data['points']
#         })
#     with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
#         json.dump(meta_info, f)
#     load_project_cache(filename)
#     return "Saved", 200

# @app.route('/delete_zone/<filename>/<zone_id>', methods=['POST'])
# def delete_zone(filename, zone_id):
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     meta_path = os.path.join(project_dir, 'meta.json')
#     if not os.path.exists(meta_path):
#         return 'Meta not found', 404
#     with open(meta_path, 'r') as f:
#         meta = json.load(f)
#     zones = meta.get("zones", [])
#     zones = [z for z in zones if str(z["id"]) != str(zone_id)]
#     meta["zones"] = zones
#     with open(meta_path, 'w') as f:
#         json.dump(meta, f)
#     load_project_cache(filename)
#     return "Deleted", 200

# @app.route('/edit_zone/<filename>/<zone_id>', methods=['POST'])
# def edit_zone(filename, zone_id):
#     project_dir = os.path.join(PROJECT_FOLDER, filename)
#     meta_path = os.path.join(project_dir, 'meta.json')
#     data = request.get_json()
#     new_points = data.get('points')
#     new_frame = data.get('frame')
#     if not os.path.exists(meta_path):
#         return 'Meta not found', 404
#     with open(meta_path, 'r') as f:
#         meta = json.load(f)
#     found = False
#     for zone in meta.get("zones", []):
#         if str(zone["id"]) == str(zone_id):
#             if new_points: zone["points"] = new_points
#             if new_frame is not None: zone["frame"] = new_frame
#             found = True
#     if found:
#         with open(meta_path, 'w') as f:
#             json.dump(meta, f)
#         load_project_cache(filename)
#         return "Edited", 200
#     return "Not found", 404

# # ===== 썸네일/동영상 리스트 =========
# def extract_thumbnail(filepath, thumb_path):
#     cap = cv2.VideoCapture(filepath)
#     ret, frame = cap.read()
#     if ret:
#         cv2.imwrite(thumb_path, frame)
#     cap.release()

# @app.route('/videos', methods=['GET', 'POST'])
# def video_list():
#     if request.method == 'POST':
#         file = request.files.get('video')
#         if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(save_path)
#             thumb_path = os.path.join(THUMB_FOLDER, filename + '.jpg')
#             extract_thumbnail(save_path, thumb_path)
#         return redirect(url_for('video_list'))
#     videos = []
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             videos.append({'name': fname, 'thumb': thumb})
#     return render_template('videos.html', videos=videos)

# @app.route('/frame/<filename>/<int:index>')
# def get_frame(filename, index):
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# # ===== Danger Zone overlay, ORB, track, etc 유지 =====
# @app.route('/frame_overlay/<filename>/<int:index>', methods=['POST'])
# def get_frame_overlay(filename, index):
#     data = request.get_json()
#     polygon = data.get("polygon", [])
#     video_path = os.path.join(UPLOAD_FOLDER, filename)
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Frame not found", 404
#     if polygon and len(polygon) >= 3:
#         pts = np.array(polygon, np.int32)
#         cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [pts], (0, 0, 255))
#         frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# # ============= RTSP/ORB 실시간 매칭 + DangerZone Overlay 캐시방식 =============
# def find_best_orb_idx(des_live, last_idx):
#     """빠른 탐색: last_idx 부근 우선, 그 외 전체 검색"""
#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     idxs = list(cached_orb_data.keys())
#     window_range = []
#     if last_idx is not None and last_idx in idxs:
#         window_range = [i for i in range(last_idx-FAST_SEARCH_WINDOW, last_idx+FAST_SEARCH_WINDOW+1) if i in idxs]
#     searched = set(window_range)
#     best_idx, best_score = -1, 0
#     # 1. fast local search
#     for idx in window_range:
#         des2 = cached_orb_data[idx]['descriptors']
#         if des_live is None or des2 is None or len(des2) == 0:
#             continue
#         matches = matcher.match(des_live, des2)
#         score = len(matches)
#         if score > best_score:
#             best_score = score
#             best_idx = idx
#     # 2. global search (if fast not enough)
#     if best_score < BEST_SCORE_THRESHOLD:
#         for idx in idxs:
#             if idx in searched: continue
#             des2 = cached_orb_data[idx]['descriptors']
#             if des_live is None or des2 is None or len(des2) == 0:
#                 continue
#             matches = matcher.match(des_live, des2)
#             score = len(matches)
#             if score > best_score:
#                 best_score = score
#                 best_idx = idx
#     return best_idx, best_score

# @app.route('/rtsp_match/<project>')
# def rtsp_match(project):
#     global selected_project, current_best_idx, current_best_score
#     selected_project = project
#     load_project_cache(project)
#     current_best_idx = -1
#     current_best_score = 0
#     return jsonify({'status': 'ok'})

# @app.route('/set_rtsp', methods=['POST'])
# def set_rtsp():
#     global rtsp_url
#     rtsp_url = request.json.get('rtsp_url')
#     return jsonify({'status':'ok'})

# @app.route("/video_feed")
# def video_feed():
#     def generate_rtsp_stream():
#         global cached_orb_data, cached_zones, current_best_idx, current_best_score, rtsp_url
#         cap = cv2.VideoCapture(rtsp_url)
#         orb = cv2.ORB_create()
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: break
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             kp_live, des_live = orb.detectAndCompute(gray, None)
#             best_idx, best_score = -1, 0
#             if selected_project and current_best_idx != -1 and des_live is not None:
#                 des_saved = cached_orb_data.get(current_best_idx, {}).get('descriptors', np.zeros((0,32),np.uint8))
#                 if len(des_saved) > 0:
#                     matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_live, des_saved)
#                     best_score = len(matches)
#                     best_idx = current_best_idx
#             if (current_best_idx == -1 or
#                 best_score < int(current_best_score * RESET_SCORE_RATIO) or
#                 best_score < BEST_SCORE_THRESHOLD):
#                 best_idx, best_score = find_best_orb_idx(des_live, current_best_idx)
#                 current_best_idx = best_idx
#                 current_best_score = best_score
#             # Danger zone overlay
#             if best_score >= BEST_SCORE_THRESHOLD and selected_project and best_idx in cached_zones:
#                 poly = np.array(cached_zones[best_idx], np.int32)
#                 cv2.polylines(frame, [poly], True, (0,0,255), 2)
#                 overlay = frame.copy()
#                 cv2.fillPoly(overlay, [poly], (0,0,255))
#                 frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
#                 cv2.putText(frame, f"Matched Frame: {best_idx} (Score: {best_score})", (10, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         cap.release()
#     return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # ========== 위험구역 관리 페이지 ==========
# @app.route('/zones')
# def danger_zone_list():
#     videos = []
#     selected_video = request.args.get('video')
#     frame_count = 0
#     zones = {}
#     for fname in os.listdir(UPLOAD_FOLDER):
#         if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             path = os.path.join(UPLOAD_FOLDER, fname)
#             thumb = f'thumbs/{fname}.jpg'
#             thumb_path = os.path.join(THUMB_FOLDER, fname + '.jpg')
#             if not os.path.exists(thumb_path):
#                 extract_thumbnail(path, thumb_path)
#             videos.append({'name': fname, 'thumb': thumb})
#     if selected_video and os.path.exists(os.path.join(UPLOAD_FOLDER, selected_video)):
#         video_path = os.path.join(UPLOAD_FOLDER, selected_video)
#         cap = cv2.VideoCapture(video_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         project_dir = os.path.join(PROJECT_FOLDER, selected_video)
#         meta_path = os.path.join(project_dir, 'meta.json')
#         if os.path.exists(meta_path):
#             with open(meta_path, 'r') as f:
#                 meta = json.load(f)
#             for zone in meta.get('zones', []):
#                 zones[zone['id']] = {'points': zone['points'], 'frame': zone['frame']}
#     return render_template('danger_zone_list.html',
#                            videos=videos,
#                            selected_video=selected_video,
#                            frame_count=frame_count,
#                            zones=zones)

# # ========= 인덱스 ===========
# @app.route("/", methods=["GET", "POST"])
# def index():
#     global rtsp_url
#     if request.method == "POST":
#         rtsp_url = request.form.get("rtsp_url", "")
#     projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
#     return render_template("index.html", rtsp_url=rtsp_url, projects=projects)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
import os
import cv2
import numpy as np
import base64
import io
import json
import pickle
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
THUMB_FOLDER = 'static/thumbs'
PROJECT_FOLDER = 'projects'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)
os.makedirs(PROJECT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMB_FOLDER'] = THUMB_FOLDER
app.config['PROJECT_FOLDER'] = PROJECT_FOLDER

# ==== 글로벌 캐시 구조 ====
rtsp_url = ""
selected_project = None
cached_orb_data = {}
cached_zones = {}
current_best_idx = -1
current_best_score = 0
BEST_SCORE_THRESHOLD = 30
RESET_SCORE_RATIO = 0.6
FAST_SEARCH_WINDOW = 10

# ================= Polygon Tracker ===================
class PolygonTracker:
    def __init__(self):
        self.points = []
        self.poly_complete = False
        self.tracking_failed = False
        self.prev_gray = None
        self.prev_points = None
        self.hist = None
        self.prev_global_pts = None

    def initialize(self, frame, points):
        self.points = [tuple(pt) for pt in points]
        self.poly_complete = True
        self.tracking_failed = False
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = np.array(self.points, dtype=np.float32)
        mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.prev_global_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

    def track(self, frame):
        if not self.poly_complete or self.tracking_failed:
            return self.points
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = next_gray.shape[:2]
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
        self.tracking_failed = True
        self.prev_gray = next_gray
        return self.points

# ========== ORB 추출/저장/캐시 ==========
def extract_orb_all_fast(video_path, scale=0.33, nfeatures=100, interval=10):
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
            'descriptors': des.astype(np.uint8) if des is not None else np.zeros((0,32), np.uint8)
        }
    cap.release()
    return data

def load_project_cache(project):
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

@app.route('/extract_orb_all/<filename>')
def extract_orb_all(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    project_dir = os.path.join(PROJECT_FOLDER, filename)
    os.makedirs(project_dir, exist_ok=True)
    orb_data = extract_orb_all_fast(video_path, scale=0.33, nfeatures=100, interval=10)
    with open(os.path.join(project_dir, 'orb_data.pkl'), 'wb') as f:
        pickle.dump(orb_data, f)
    return jsonify({'status': 'ok', 'count': len(orb_data)})

# ========== Danger Zone 관리, 저장, 삭제, 수정 ===========
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

@app.route('/delete_zone/<filename>/<zone_id>', methods=['POST'])
def delete_zone(filename, zone_id):
    project_dir = os.path.join(PROJECT_FOLDER, filename)
    meta_path = os.path.join(project_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return 'Meta not found', 404
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    zones = meta.get("zones", [])
    zones = [z for z in zones if str(z["id"]) != str(zone_id)]
    meta["zones"] = zones
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    load_project_cache(filename)
    return "Deleted", 200

@app.route('/edit_zone/<filename>/<zone_id>', methods=['POST'])
def edit_zone(filename, zone_id):
    project_dir = os.path.join(PROJECT_FOLDER, filename)
    meta_path = os.path.join(project_dir, 'meta.json')
    data = request.get_json()
    new_points = data.get('points')
    new_frame = data.get('frame')
    if not os.path.exists(meta_path):
        return 'Meta not found', 404
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    found = False
    for zone in meta.get("zones", []):
        if str(zone["id"]) == str(zone_id):
            if new_points: zone["points"] = new_points
            if new_frame is not None: zone["frame"] = new_frame
            found = True
    if found:
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        load_project_cache(filename)
        return "Edited", 200
    return "Not found", 404

# ===== 썸네일/동영상 리스트 =========
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

# ===== Danger Zone overlay, ORB, track, etc 유지 =====
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

# ============= RTSP/ORB 실시간 매칭 + DangerZone Overlay 캐시방식 =============
def find_best_orb_idx(des_live, last_idx):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    idxs = list(cached_orb_data.keys())
    idxs.sort()
    window_range = []
    if last_idx is not None and last_idx in idxs:
        i = idxs.index(last_idx)
        window_range = idxs[max(0,i-FAST_SEARCH_WINDOW):i+FAST_SEARCH_WINDOW+1]
    searched = set(window_range)
    best_idx, best_score = -1, 0
    for idx in window_range:
        des2 = cached_orb_data[idx]['descriptors']
        if des_live is None or des2 is None or len(des2) == 0:
            continue
        matches = matcher.match(des_live, des2)
        score = len(matches)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score < BEST_SCORE_THRESHOLD:
        for idx in idxs:
            if idx in searched: continue
            des2 = cached_orb_data[idx]['descriptors']
            if des_live is None or des2 is None or len(des2) == 0:
                continue
            matches = matcher.match(des_live, des2)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx, best_score

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

@app.route("/video_feed")
def video_feed():
    def generate_rtsp_stream():
        global cached_orb_data, cached_zones, current_best_idx, current_best_score, rtsp_url
        cap = cv2.VideoCapture(rtsp_url)
        orb = cv2.ORB_create(nfeatures=100)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
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
                best_idx, best_score = find_best_orb_idx(des_live, current_best_idx)
                current_best_idx = best_idx
                current_best_score = best_score
            # Danger zone overlay
            if best_score >= BEST_SCORE_THRESHOLD and selected_project and best_idx in cached_zones:
                poly = np.array(cached_zones[best_idx], np.int32)
                cv2.polylines(frame, [poly], True, (0,0,255), 2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], (0,0,255))
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                cv2.putText(frame, f"Matched Frame: {best_idx} (Score: {best_score})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== 위험구역 관리 페이지 ==========
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

# ========= 인덱스 ===========
@app.route("/", methods=["GET", "POST"])
def index():
    global rtsp_url
    if request.method == "POST":
        rtsp_url = request.form.get("rtsp_url", "")
    projects = [d for d in os.listdir(PROJECT_FOLDER) if os.path.isdir(os.path.join(PROJECT_FOLDER, d))]
    return render_template("index.html", rtsp_url=rtsp_url, projects=projects)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
