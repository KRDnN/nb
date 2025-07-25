import sys
import cv2
import numpy as np
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QListWidget, QListWidgetItem, QMessageBox, QDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer
# ---------- 유틸 -----------
def cv_to_qpixmap(cv_img):
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(cv_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def arrange_frames_in_grid(frames):
    n = len(frames)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w, c = frames[0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)
    grid = []
    for r in range(rows):
        row = []
        for c_ in range(cols):
            idx = r * cols + c_
            if idx < n:
                row.append(frames[idx])
            else:
                row.append(blank.copy())
        grid.append(np.hstack(row))
    return np.vstack(grid), rows, cols

def extract_orb_features(image):
    orb = cv2.ORB_create(50000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    kp_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2))
    descriptors = descriptors if descriptors is not None else np.empty((0, 32), dtype=np.uint8)
    return kp_coords, descriptors

def get_base_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

# --------- Polygon Tracker ---------
class PolygonTracker:
    def __init__(self, cube_height=50):
        self.points = []
        self.poly_complete = False
        self.tracking_failed = False
        self.prev_gray = None
        self.prev_points = None
        self.poly_pts_inside = []
        self.superpixel_centers = []
        self.last_valid_superpixel_centers = []
        self.orb_kp = []
        self.cube_height = cube_height
        self.prev_global_pts = None
        self.hist = None

    def reset(self):
        self.points = []
        self.poly_complete = False
        self.tracking_failed = False
        self.prev_gray = None
        self.prev_points = None
        self.poly_pts_inside = []
        self.superpixel_centers = []
        self.last_valid_superpixel_centers = []
        self.orb_kp = []
        self.prev_global_pts = None
        self.hist = None

    def compute_histogram(self, frame, poly_pts):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def initialize(self, frame, points):
        self.points = [tuple(pt) for pt in points]
        self.poly_complete = True
        self.tracking_failed = False
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = np.array(self.points, dtype=np.float32)
        self.hist = self.compute_histogram(frame, self.points)

        mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [self.prev_points.astype(np.int32)], 255)
        self.orb_kp = cv2.ORB_create().detect(self.prev_gray, mask=mask)
        ys, xs = np.where(mask == 255)
        self.poly_pts_inside = list(zip(xs[::30], ys[::30]))
        self.prev_global_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)
        try:
            slic = cv2.ximgproc.createSuperpixelSLIC(frame, region_size=20)
            slic.iterate(10)
            labels = slic.getLabels()
            num_labels = slic.getNumberOfSuperpixels()
            centers = []
            for i in range(num_labels):
                ys, xs = np.where((labels == i) & (mask == 255))
                if len(xs) > 0:
                    centers.append((np.mean(xs), np.mean(ys)))
            self.superpixel_centers = centers
            self.last_valid_superpixel_centers = centers.copy()
        except Exception:
            self.superpixel_centers = []
            self.last_valid_superpixel_centers = []

    def track(self, next_frame):
        if not self.poly_complete or self.tracking_failed:
            return

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        h, w = next_gray.shape[:2]
        internal_tracked = False

        # 1. Optical Flow (local)
        if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) >= 3:
            p0 = self.prev_points.reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, p0, None)
            if p1 is not None:
                clipped_pts = []
                for pt in p1.reshape(-1, 2):
                    x, y = pt
                    x_clip = min(max(x, 0), w - 1)
                    y_clip = min(max(y, 0), h - 1)
                    clipped_pts.append((float(x_clip), float(y_clip)))
                if len(clipped_pts) >= 3 and sum([0 <= x < w and 0 <= y < h for x, y in clipped_pts]) >= len(clipped_pts) * 0.6:
                    self.points = clipped_pts
                    self.prev_points = np.array(self.points, dtype=np.float32)
                    internal_tracked = True

        # 2. Global Affine
        global_tracked = False
        if not internal_tracked or len(self.points) < 3:
            if self.prev_global_pts is not None and len(self.prev_global_pts) >= 3:
                curr_global_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, self.prev_global_pts, None)
                st = st.reshape(-1)
                prev_g = self.prev_global_pts[st == 1]
                curr_g = curr_global_pts[st == 1]
                if len(prev_g) >= 3 and len(curr_g) >= 3:
                    M, inliers = cv2.estimateAffinePartial2D(prev_g, curr_g)
                    if M is not None:
                        poly = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
                        poly_trans = cv2.transform(poly, M)
                        self.points = [
                            (float(min(max(pt[0][0], 0), w - 1)),
                             float(min(max(pt[0][1], 0), h - 1)))
                            for pt in poly_trans
                        ]
                        self.prev_points = np.array(self.points, dtype=np.float32)
                        global_tracked = True
                self.prev_global_pts = curr_g.reshape(-1, 1, 2)
            else:
                self.prev_global_pts = cv2.goodFeaturesToTrack(next_gray, maxCorners=100, qualityLevel=0.01, minDistance=8)

        # 3. ORB + Homography
        orb_tracked = False
        if not internal_tracked and not global_tracked and self.orb_kp:
            orb = cv2.ORB_create()
            kp2, des2 = orb.compute(next_gray, self.orb_kp)
            if des2 is not None and hasattr(self, 'prev_orb_des') and self.prev_orb_des is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.prev_orb_des, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) >= 10:
                    src_pts = np.float32([self.orb_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        poly = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
                        poly_trans = cv2.perspectiveTransform(poly, H)
                        self.points = [
                            (float(min(max(pt[0][0], 0), w - 1)),
                             float(min(max(pt[0][1], 0), h - 1)))
                            for pt in poly_trans
                        ]
                        self.prev_points = np.array(self.points, dtype=np.float32)
                        orb_tracked = True

        # 4. Histogram meanShift
        hist_tracked = False
        if (not internal_tracked and not global_tracked and not orb_tracked) or len(self.points) < 3:
            if self.hist is not None and len(self.points) >= 3:
                hsv = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)
                pts = np.array(self.points, dtype=np.int32)
                x, y, w_box, h_box = cv2.boundingRect(pts)
                back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                _, new_window = cv2.meanShift(back_proj, (x, y, w_box, h_box), term_crit)
                dx = (new_window[0] + new_window[2] // 2) - (x + w_box // 2)
                dy = (new_window[1] + new_window[3] // 2) - (y + h_box // 2)
                self.points = [(pt[0] + dx, pt[1] + dy) for pt in self.points]
                self.prev_points = np.array(self.points, dtype=np.float32)
                hist_tracked = True

        # Update ORB
        self.prev_gray = next_gray.copy()
        orb = cv2.ORB_create()
        self.orb_kp = orb.detect(next_gray, None)
        self.prev_orb_des = None
        if self.orb_kp:
            _, self.prev_orb_des = orb.compute(next_gray, self.orb_kp)

        if len(self.points) < 3:
            self.tracking_failed = True

        # 공통 후처리
        self.prev_gray = next_gray.copy()
        self.orb_kp = cv2.ORB_create().detect(next_gray, None)
        if len(self.points) < 3:
            self.tracking_failed = True

        if self.poly_pts_inside:
            inner_np = np.array(self.poly_pts_inside, dtype=np.float32).reshape(-1, 1, 2)
            tracked, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, inner_np, None)
            new_inside = []
            for pt in tracked.reshape(-1, 2):
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    new_inside.append((float(x), float(y)))
            self.poly_pts_inside = new_inside

        if self.superpixel_centers:
            sp_np = np.array(self.superpixel_centers, dtype=np.float32).reshape(-1, 1, 2)
            tracked, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, sp_np, None)
            new_sp = []
            for pt in tracked.reshape(-1, 2):
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    new_sp.append((float(x), float(y)))
            if len(new_sp) >= len(self.superpixel_centers) * 0.6:
                self.superpixel_centers = new_sp
                self.last_valid_superpixel_centers = new_sp.copy()
            else:
                self.superpixel_centers = self.last_valid_superpixel_centers.copy()
        if len(self.points) == 0 or all((x < 1 or x >= w - 1 or y < 1 or y >= h - 1) for x, y in self.points):
            self.tracking_failed = True

class ImageDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("프레임 그리드 결과")
        layout = QVBoxLayout()
        scroll = QScrollArea()
        label = QLabel()
        label.setPixmap(cv_to_qpixmap(image))
        label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.resize(1000, 800)

class VideoAllInOneApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame/Polygon ORB & Tracking All-in-One")
        self.resize(1280, 900)

        # 공통 상태
        self.frames = []         # 원본 프레임
        self.resized_frames = [] # 축소 프레임(저장용)
        self.frame_grid = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_frame_h = 0
        self.grid_frame_w = 0
        self.nx1_saved_path = ""
        self.orb_features = None

        self.tracker = PolygonTracker()
        self.current_frame_index = 0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.scale_ratio = 0.5  # 축소 배율 (저장/매칭용)

        self.setup_ui()

    def setup_ui(self):
        btns = QHBoxLayout()
        self.btn_load_video = QPushButton("동영상 불러오기 (원본 재생)")
        self.btn_load_video.clicked.connect(self.load_video_original)
        btns.addWidget(self.btn_load_video)

        self.btn_save_nx1 = QPushButton("Nx1 프레임/특징점 저장 (축소본)")
        self.btn_save_nx1.clicked.connect(self.save_video_as_nx1)
        btns.addWidget(self.btn_save_nx1)

        self.btn_load_features = QPushButton("Nx1 특징 불러오기")
        self.btn_load_features.clicked.connect(self.load_features)
        btns.addWidget(self.btn_load_features)

        self.btn_analyze = QPushButton("동영상과 저장 Nx1 분석")
        self.btn_analyze.clicked.connect(self.analyze_new_video)
        btns.addWidget(self.btn_analyze)

        self.btn_template_match = QPushButton("이미지와 Nx1 템플릿 매칭")
        self.btn_template_match.clicked.connect(self.template_match)
        btns.addWidget(self.btn_template_match)

        self.btn_play = QPushButton("재생")
        self.btn_play.clicked.connect(self.toggle_playback)
        btns.addWidget(self.btn_play)

        self.btn_point = QPushButton("점 찍기")
        self.btn_point.clicked.connect(self.enable_pointing)
        btns.addWidget(self.btn_point)

        self.btn_poly_complete = QPushButton("점 찍기 완료")
        self.btn_poly_complete.clicked.connect(self.complete_polygon)
        btns.addWidget(self.btn_poly_complete)

        self.btn_match_all = QPushButton("전체 프레임 ORB 매칭")
        self.btn_match_all.clicked.connect(self.match_all_frames_with_nx1)
        btns.addWidget(self.btn_match_all)


        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.mousePressEvent = self.on_click
        self.image_label.mouseMoveEvent = self.on_drag

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_move)
        self.thumb_list = QListWidget()
        self.thumb_list.setFixedHeight(80)
        self.thumb_list.currentRowChanged.connect(self.on_thumb_select)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btns)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.thumb_list)
        self.setLayout(main_layout)

    # ---- 원본 영상 로딩 (트래킹/재생용) ----
    def load_video_original(self):
        fname, _ = QFileDialog.getOpenFileName(self, "동영상 선택", "", "Videos (*.mp4 *.avi *.mov)")
        if not fname:
            return
        cap = cv2.VideoCapture(fname)
        self.frames.clear()
        ret, frame = cap.read()
        while ret:
            self.frames.append(frame.copy())
            ret, frame = cap.read()
        cap.release()
        if not self.frames:
            QMessageBox.warning(self, "경고", "프레임을 추출하지 못했습니다.")
            return
        self.slider.setMaximum(len(self.frames) - 1)
        self.generate_thumbnails()
        self.show_frame(0)
        self.enable_pointing()

    # ---- Nx1 저장 (축소본만 별도 저장) ----
    def save_video_as_nx1(self):
        if not self.frames:
            QMessageBox.warning(self, "오류", "먼저 영상을 불러오세요.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Nx1/그리드 저장 폴더 선택")
        if not save_dir:
            return

        base = "saved_video"
        save_path = os.path.join(save_dir, base)
        os.makedirs(save_path, exist_ok=True)

        self.resized_frames = []
        for frame in self.frames:
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (w // 8, h // 8))
            self.resized_frames.append(resized)

        grid_img, rows, cols = arrange_frames_in_grid(self.resized_frames)
        frame_h, frame_w = self.resized_frames[0].shape[:2]
        self.frame_grid = grid_img
        self.grid_rows = rows
        self.grid_cols = cols
        self.grid_frame_h = frame_h
        self.grid_frame_w = frame_w
        self.nx1_saved_path = save_path

        kp_list, des_list, coords_list = [], [], []
        full_kp_coords, full_descriptors, full_frame_indices = [], [], []

        for i, frame in enumerate(self.resized_frames):
            kp, des = extract_orb_features(frame)
            kp_list.append(kp)
            des_list.append(des)
            coords_list.append((i // cols, i % cols))
            for k in kp:
                full_kp_coords.append(k)
                full_frame_indices.append(i)
            full_descriptors.append(des)

        np.savez_compressed(os.path.join(save_path, f"{base}_features.npz"),
            keypoints=np.array(kp_list, dtype=object),
            descriptors=np.array(des_list, dtype=object),
            grid_coords=np.array(coords_list, dtype=object),
            frame_h=frame_h,
            frame_w=frame_w,
            rows=rows,
            cols=cols,
            all_keypoints=np.array(full_kp_coords, dtype=object),
            all_descriptors=np.array(full_descriptors, dtype=object),
            frame_indices=np.array(full_frame_indices, dtype=object),
            grid_image=grid_img
        )
        grid_path = os.path.join(save_path, f"{base}_grid.jpg")
        cv2.imwrite(grid_path, grid_img)
        QMessageBox.information(self, "완료", f"✔ {grid_path} 와 특징점 저장 완료")

        dlg = ImageDialog(grid_img)
        dlg.exec_()

    # ---- Nx1 특징 불러오기 ----
    def load_features(self):
        f_npz, _ = QFileDialog.getOpenFileName(self, "특징 파일 선택", "", "NPZ Files (*.npz)")
        if not f_npz:
            return
        data = np.load(f_npz, allow_pickle=True)
        self.orb_features = data
        self.frame_grid = data['grid_image']
        self.grid_rows = int(data['rows'])
        self.grid_cols = int(data['cols'])
        self.grid_frame_h = int(data['frame_h'])
        self.grid_frame_w = int(data['frame_w'])
        QMessageBox.information(self, "불러오기 완료", f"프레임 개수: {len(data['keypoints'])}\n행렬 크기: {data['rows']}x{data['cols']}")

    # ---- 분석/매칭 (Nx1 축소본 기반) ----
    def analyze_new_video(self):
        if self.orb_features is None:
            QMessageBox.warning(self, "오류", "Nx1 프레임 특징을 먼저 불러오세요.")
            return
        data = self.orb_features
        grid_img = data['grid_image']
        frame_h = int(data['frame_h'])
        frame_w = int(data['frame_w'])
        fname, _ = QFileDialog.getOpenFileName(self, "분석할 동영상 선택", "", "Videos (*.mp4 *.avi *.mov)")
        if not fname:
            return
        cap = cv2.VideoCapture(fname)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "오류", "프레임을 읽지 못했습니다.")
            return
        resized = cv2.resize(frame, (frame_w, frame_h))
        gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if len(grid_img.shape) == 3:
            grid_img_gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        else:
            grid_img_gray = grid_img
        if grid_img_gray.shape[0] < gray_template.shape[0] or grid_img_gray.shape[1] < gray_template.shape[1]:
            QMessageBox.warning(self, "오류", "분석 템플릿이 그리드 이미지보다 큽니다.")
            return
        result = cv2.matchTemplate(grid_img_gray, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        cols = int(data['cols'])
        r = max_loc[1] // frame_h
        c = max_loc[0] // frame_w
        QMessageBox.information(self, "일치 분석 결과",
            f"""가장 유사한 프레임 위치: 행 {r}, 열 {c}
매칭 유사도: {max_val:.3f}
매칭 좌표: x={max_loc[0]}, y={max_loc[1]}""")

    def template_match(self):
        if self.orb_features is None:
            QMessageBox.warning(self, "오류", "Nx1 프레임 특징을 먼저 불러오세요.")
            return
        data = self.orb_features
        grid_img = data['grid_image']
        if len(grid_img.shape) == 3:
            grid_img_gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        else:
            grid_img_gray = grid_img

        img_path, _ = QFileDialog.getOpenFileName(self, "비교할 이미지 선택", "", "Images (*.png *.jpg *.bmp)")
        if not img_path:
            return

        template = cv2.imread(img_path, 0)
        th, tw = template.shape[:2]
        gh, gw = grid_img_gray.shape[:2]
        if th > gh or tw > gw:
            QMessageBox.warning(self, "오류", "비교 템플릿이 Nx1 그리드보다 큽니다.")
            return

        res = cv2.matchTemplate(grid_img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        x, y = max_loc

        show_img = cv2.cvtColor(grid_img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(show_img, (x, y), (x + tw, y + th), (255, 0, 0), 2)
        cv2.imwrite("match_result.jpg", show_img)
        QMessageBox.information(self, "템플릿 매칭", f"매칭 위치: x={x}, y={y}, 유사도: {max_val:.3f}\nmatch_result.jpg로 저장됨")

    # ----- 트래킹 & 시각화 (원본 프레임) -----
    def generate_thumbnails(self):
        self.thumb_list.clear()
        self.thumbnails = []
        for idx, frame in enumerate(self.frames):
            thumb = cv2.resize(frame, (80, 60))
            thumb_img = QImage(thumb.data, thumb.shape[1], thumb.shape[0], thumb.strides[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(thumb_img)
            item = QListWidgetItem(f"Frame {idx}")
            item.setIcon(QIcon(pixmap.scaled(80, 60, Qt.KeepAspectRatio)))
            self.thumb_list.addItem(item)
            self.thumbnails.append(pixmap)

    def on_thumb_select(self, idx):
        if idx < 0 or idx >= len(self.frames):
            return
        self.show_frame(idx)

    def on_slider_move(self, val):
        new_index = int(val)
        if hasattr(self, "frames") and self.frames:
            if new_index > self.current_frame_index and self.tracker.poly_complete and not self.tracker.tracking_failed:
                self.track_polygon()
            self.show_frame(new_index)

    def show_frame(self, index):
        if not self.frames:
            return
        self.current_frame_index = index
        frame = self.frames[index].copy()
        self.draw_points(frame)
        self.draw_cube(frame)
        self.draw_orb_features(frame)
        self.draw_superpixel_centers(frame)
        self.draw_polygon_mask(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def enable_pointing(self):
        self.tracker.reset()
        self.show_frame(self.current_frame_index)

    def on_click(self, event):
        if self.tracker.poly_complete:
            return
        pos = event.pos()
        self.tracker.points.append((pos.x(), pos.y()))
        self.show_frame(self.current_frame_index)

    def on_drag(self, event):
        if self.tracker.poly_complete or event.buttons() != Qt.LeftButton:
            return
        pos = event.pos()
        self.tracker.points.append((pos.x(), pos.y()))
        self.show_frame(self.current_frame_index)

    def draw_points(self, frame):
        for pt in self.tracker.points:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
        if self.tracker.poly_complete and len(self.tracker.points) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in self.tracker.points], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                line_points = list(zip(np.linspace(p1[0], p2[0], num=10), np.linspace(p1[1], p2[1], num=10)))
                for x, y in line_points:
                    cv2.circle(frame, (int(x), int(y)), 1, (100, 0, 255), -1)

    def draw_orb_features(self, frame):
        if self.tracker.poly_complete:
            h, w = frame.shape[:2]
            valid_kp = []
            for point in self.tracker.orb_kp:
                x, y = int(point.pt[0]), int(point.pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    valid_kp.append(point)
            self.tracker.orb_kp = valid_kp

    def draw_superpixel_centers(self, frame):
        h, w = frame.shape[:2]
        valid_sp = []
        for x, y in self.tracker.superpixel_centers:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                valid_sp.append((x, y))
        self.tracker.superpixel_centers = valid_sp

    def draw_polygon_mask(self, frame):
        if not self.tracker.poly_pts_inside:
            return
        pts = np.array(self.tracker.points, dtype=np.int32)
        if pts.shape[0] >= 3:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            red_overlay = np.zeros_like(frame, dtype=np.uint8)
            red_overlay[:] = (0, 0, 255)
            alpha = 0.3
            blended = cv2.addWeighted(frame, 1.0, red_overlay, alpha, 0)
            frame[:] = np.where(mask[:, :, np.newaxis] == 255, blended, frame)

    def draw_cube(self, frame):
        if not self.tracker.poly_complete or len(self.tracker.points) < 3:
            return
        base = np.array(self.tracker.points, dtype=np.float32)
        base_int = base.astype(np.int32)
        direction = np.array([0, -self.tracker.cube_height], dtype=np.float32)
        top = base + direction
        top_int = top.astype(np.int32)
        for i in range(len(base)):
            cv2.line(frame, tuple(base_int[i]), tuple(base_int[(i+1)%len(base)]), (255, 0, 0), 1)
            cv2.line(frame, tuple(top_int[i]), tuple(top_int[(i+1)%len(top)]), (255, 0, 0), 1)
            cv2.line(frame, tuple(base_int[i]), tuple(top_int[i]), (255, 0, 0), 1)

    def complete_polygon(self):
        self.tracker.poly_complete = True
        self.tracker.tracking_failed = False
        if self.frames:
            frame = self.frames[self.current_frame_index]
            self.tracker.initialize(frame, self.tracker.points)
            self.show_frame(self.current_frame_index)

    def track_polygon(self):
        if self.current_frame_index + 1 >= len(self.frames):
            return
        next_frame = self.frames[self.current_frame_index + 1]
        self.tracker.track(next_frame)
        self.show_frame(self.current_frame_index + 1)

    def toggle_playback(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
        else:
            self.playing = True
            self.timer.start(100)

    def play_next_frame(self):
        if not self.playing or self.current_frame_index + 1 >= len(self.frames):
            self.playing = False
            self.timer.stop()
            return
        if self.tracker.poly_complete and not self.tracker.tracking_failed:
            self.track_polygon()
        self.current_frame_index += 1
        self.slider.setValue(self.current_frame_index)
        self.show_frame(self.current_frame_index)

    def match_all_frames_with_nx1(self):
        if self.orb_features is None or not self.frames:
            QMessageBox.warning(self, "오류", "Nx1 특징과 원본 영상이 모두 필요합니다.")
            return

        data = self.orb_features
        all_descriptors = data["all_descriptors"]
        frame_indices = data["frame_indices"]
        grid_img = data["grid_image"].copy()
        frame_h = int(data["frame_h"])
        frame_w = int(data["frame_w"])
        rows = int(data["rows"])
        cols = int(data["cols"])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        grid_matches = []

        for idx, frame in enumerate(self.frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(des) == 0:
                grid_matches.append((-1, 0))
                continue

            best_score = -1
            best_frame_idx = -1
            for i, ref_des in enumerate(all_descriptors):
                if ref_des is None or len(ref_des) == 0:
                    continue
                matches = bf.match(des, ref_des)
                score = sum([m.distance for m in matches]) / len(matches) if matches else 1e9
                if score < best_score or best_score == -1:
                    best_score = score
                    best_frame_idx = frame_indices[i]

            grid_matches.append((best_frame_idx, best_score))

        # 시각화
        vis = grid_img.copy()
        for i, (fidx, score) in enumerate(grid_matches):
            if fidx < 0:
                continue
            r = fidx // cols
            c = fidx % cols
            x = c * frame_w
            y = r * frame_h
            cv2.rectangle(vis, (x, y), (x + frame_w, y + frame_h), (0, 0, 255), 2)
            cv2.putText(vis, f"#{i}", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        dlg = ImageDialog(vis)
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoAllInOneApp()
    win.show()
    sys.exit(app.exec_())