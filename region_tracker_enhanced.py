import sys
import cv2
import numpy as np
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QListWidget, QListWidgetItem, QMessageBox, QDialog, 
    QScrollArea, QCheckBox, QSpinBox, QComboBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer

# ---------- 유틸 함수 -----------
def cv_to_qpixmap(cv_img):
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(cv_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def extract_region_features(image, mask=None):
    """영역별 다양한 특징 추출"""
    features = {}
    
    # 1. ORB 특징점
    orb = cv2.ORB_create(10000)
    kp, desc = orb.detectAndCompute(image, mask)
    features['orb_kp'] = kp
    features['orb_desc'] = desc
    
    # 2. 컬러 히스토그램 (HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [30, 32, 32], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    features['color_hist'] = hist
    
    # 3. LBP (Local Binary Pattern) 텍스처
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lbp = compute_lbp(gray, mask)
    features['lbp_hist'] = lbp
    
    # 4. 모멘트 특징
    if mask is not None:
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            features['centroid'] = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
            features['area'] = moments['m00']
        else:
            features['centroid'] = (0, 0)
            features['area'] = 0
    
    # 5. Edge 밀도
    edges = cv2.Canny(gray, 50, 150)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    edge_density = np.sum(edges > 0) / np.sum(mask > 0) if mask is not None else np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    features['edge_density'] = edge_density
    
    return features

def compute_lbp(gray, mask=None, radius=3, n_points=24):
    """Local Binary Pattern 계산"""
    lbp = np.zeros_like(gray)
    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            binary_string = ""
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                if x < gray.shape[0] and y < gray.shape[1]:
                    binary_string += "1" if gray[x, y] >= center else "0"
            lbp[i, j] = int(binary_string, 2) if len(binary_string) == n_points else 0
    
    # 히스토그램 계산
    if mask is not None:
        hist = cv2.calcHist([lbp], [0], mask, [256], [0, 256])
    else:
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def compare_features(feat1, feat2, weights=None):
    """다중 특징 비교 및 유사도 계산"""
    if weights is None:
        weights = {
            'orb': 0.3,
            'color': 0.25, 
            'lbp': 0.2,
            'geometry': 0.15,
            'edge': 0.1
        }
    
    similarity_scores = {}
    
    # 1. ORB 매칭
    if feat1.get('orb_desc') is not None and feat2.get('orb_desc') is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(feat1['orb_desc'], feat2['orb_desc'])
        if len(matches) > 0:
            good_matches = [m for m in matches if m.distance < 50]
            orb_score = len(good_matches) / max(len(feat1['orb_desc']), len(feat2['orb_desc']))
        else:
            orb_score = 0
    else:
        orb_score = 0
    similarity_scores['orb'] = orb_score
    
    # 2. 컬러 히스토그램 비교
    color_score = cv2.compareHist(feat1['color_hist'], feat2['color_hist'], cv2.HISTCMP_CORREL)
    similarity_scores['color'] = max(0, color_score)
    
    # 3. LBP 텍스처 비교
    lbp_score = cv2.compareHist(feat1['lbp_hist'].reshape(-1, 1), feat2['lbp_hist'].reshape(-1, 1), cv2.HISTCMP_CORREL)
    similarity_scores['lbp'] = max(0, lbp_score)
    
    # 4. 기하학적 유사도
    area_ratio = min(feat1['area'], feat2['area']) / max(feat1['area'], feat2['area']) if max(feat1['area'], feat2['area']) > 0 else 0
    centroid_dist = np.linalg.norm(np.array(feat1['centroid']) - np.array(feat2['centroid']))
    geometry_score = area_ratio * np.exp(-centroid_dist / 100)  # 거리에 따른 지수 감소
    similarity_scores['geometry'] = geometry_score
    
    # 5. Edge 밀도 비교
    edge_score = 1 - abs(feat1['edge_density'] - feat2['edge_density'])
    similarity_scores['edge'] = max(0, edge_score)
    
    # 가중 평균 계산
    total_score = sum(similarity_scores[key] * weights[key.split('_')[0]] for key in similarity_scores)
    
    return total_score, similarity_scores

# --------- Enhanced Region Tracker ---------
class EnhancedRegionTracker:
    def __init__(self):
        self.regions = []  # 여러 영역 추적
        self.region_features = []
        self.tracking_methods = ['optical_flow', 'template_matching', 'feature_matching', 'mean_shift']
        self.current_method = 'optical_flow'
        self.adaptive_threshold = 0.7
        self.frame_buffer = []
        self.max_buffer_size = 10
        
    def add_region(self, frame, points, region_id=None):
        """새로운 영역 추가"""
        if region_id is None:
            region_id = len(self.regions)
            
        # 폴리곤 마스크 생성
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # 영역 특징 추출
        features = extract_region_features(frame, mask)
        
        region_info = {
            'id': region_id,
            'points': points,
            'mask': mask,
            'features': features,
            'last_seen_frame': 0,
            'confidence': 1.0,
            'tracking_method': self.current_method,
            'lost_count': 0
        }
        
        self.regions.append(region_info)
        return region_id
    
    def update_regions(self, frame, frame_idx):
        """모든 영역 업데이트"""
        for region in self.regions:
            if region['lost_count'] > 5:  # 5프레임 이상 놓치면 스킵
                continue
                
            success = self._track_single_region(frame, region, frame_idx)
            if success:
                region['last_seen_frame'] = frame_idx
                region['lost_count'] = 0
                region['confidence'] = min(1.0, region['confidence'] + 0.1)
            else:
                region['lost_count'] += 1
                region['confidence'] = max(0.0, region['confidence'] - 0.2)
    
    def _track_single_region(self, frame, region, frame_idx):
        """단일 영역 추적"""
        # 현재 사용 중인 추적 방법 시도
        method = region['tracking_method']
        
        if method == 'optical_flow':
            success = self._track_optical_flow(frame, region)
        elif method == 'template_matching':
            success = self._track_template_matching(frame, region)
        elif method == 'feature_matching':
            success = self._track_feature_matching(frame, region)
        elif method == 'mean_shift':
            success = self._track_mean_shift(frame, region)
        else:
            success = False
            
        # 실패 시 다른 방법 시도
        if not success:
            for backup_method in self.tracking_methods:
                if backup_method != method:
                    if backup_method == 'optical_flow':
                        success = self._track_optical_flow(frame, region)
                    elif backup_method == 'template_matching':
                        success = self._track_template_matching(frame, region)
                    elif backup_method == 'feature_matching':
                        success = self._track_feature_matching(frame, region)
                    elif backup_method == 'mean_shift':
                        success = self._track_mean_shift(frame, region)
                        
                    if success:
                        region['tracking_method'] = backup_method
                        break
        
        return success
    
    def _track_optical_flow(self, frame, region):
        """Optical Flow 추적"""
        if not hasattr(region, 'prev_gray') or region.get('prev_gray') is None:
            region['prev_gray'] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 영역 경계점들에 대한 optical flow
        points = np.array(region['points'], dtype=np.float32).reshape(-1, 1, 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            region['prev_gray'], gray, points, None
        )
        
        # 유효한 점들만 선택
        good_points = new_points[status == 1]
        if len(good_points) >= 3:
            region['points'] = good_points.reshape(-1, 2).tolist()
            # 마스크 업데이트
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(region['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            region['mask'] = mask
            region['prev_gray'] = gray
            return True
            
        return False
    
    def _track_template_matching(self, frame, region):
        """템플릿 매칭 추적"""
        # 이전 프레임의 영역을 템플릿으로 사용
        if not hasattr(region, 'template') or region.get('template') is None:
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, region['template'], cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > self.adaptive_threshold:
            # 새로운 위치로 영역 이동
            h, w = region['template'].shape
            new_center = (max_loc[0] + w//2, max_loc[1] + h//2)
            old_center = np.mean(region['points'], axis=0)
            offset = np.array(new_center) - old_center
            
            # 모든 점들을 이동
            new_points = []
            for point in region['points']:
                new_points.append([point[0] + offset[0], point[1] + offset[1]])
            
            region['points'] = new_points
            # 마스크 업데이트
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array(region['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            region['mask'] = mask
            return True
            
        return False
    
    def _track_feature_matching(self, frame, region):
        """특징점 매칭 추적"""
        current_features = extract_region_features(frame, region['mask'])
        
        similarity, _ = compare_features(region['features'], current_features)
        
        if similarity > self.adaptive_threshold:
            # 특징점이 충분히 유사하면 성공
            region['features'] = current_features
            return True
            
        return False
    
    def _track_mean_shift(self, frame, region):
        """Mean Shift 추적"""
        if not hasattr(region, 'hist') or region.get('hist') is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            region['hist'] = cv2.calcHist([hsv], [0, 1], region['mask'], [30, 32], [0, 180, 0, 256])
            cv2.normalize(region['hist'], region['hist'], 0, 255, cv2.NORM_MINMAX)
            return True
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], region['hist'], [0, 180, 0, 256], 1)
        
        # 영역의 bounding box 계산
        pts = np.array(region['points'], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Mean shift 적용
        term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, new_window = cv2.meanShift(back_proj, (x, y, w, h), term_criteria)
        
        # 영역 이동
        dx = new_window[0] - x
        dy = new_window[1] - y
        
        new_points = []
        for point in region['points']:
            new_points.append([point[0] + dx, point[1] + dy])
            
        region['points'] = new_points
        # 마스크 업데이트
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(region['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        region['mask'] = mask
        
        return True
    
    def search_regions_in_frame(self, frame, threshold=0.8):
        """프레임에서 등록된 영역들 검색"""
        found_regions = []
        
        for region in self.regions:
            if region['lost_count'] > 10:  # 너무 오래 놓친 영역은 제외
                continue
                
            # 전체 프레임에서 슬라이딩 윈도우로 검색
            candidates = self._sliding_window_search(frame, region, threshold)
            
            for candidate in candidates:
                found_regions.append({
                    'region_id': region['id'],
                    'location': candidate['location'],
                    'confidence': candidate['confidence'],
                    'method': candidate['method']
                })
                
        return found_regions
    
    def _sliding_window_search(self, frame, region, threshold):
        """슬라이딩 윈도우를 이용한 영역 검색"""
        candidates = []
        
        # 원본 영역 크기 계산
        pts = np.array(region['points'], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # 다양한 스케일로 검색
        scales = [0.8, 1.0, 1.2]
        step_size = max(w//4, h//4, 20)
        
        for scale in scales:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            
            for y_pos in range(0, frame.shape[0] - scaled_h, step_size):
                for x_pos in range(0, frame.shape[1] - scaled_w, step_size):
                    # 현재 윈도우에서 특징 추출
                    roi = frame[y_pos:y_pos+scaled_h, x_pos:x_pos+scaled_w]
                    roi_mask = np.ones((scaled_h, scaled_w), dtype=np.uint8) * 255
                    
                    roi_features = extract_region_features(roi, roi_mask)
                    similarity, _ = compare_features(region['features'], roi_features)
                    
                    if similarity > threshold:
                        candidates.append({
                            'location': (x_pos, y_pos, scaled_w, scaled_h),
                            'confidence': similarity,
                            'method': 'sliding_window'
                        })
        
        # 중복 제거 및 정렬
        candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        return candidates[:5]  # 상위 5개만 반환

class RegionTrackingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Region Tracking System")
        self.resize(1400, 1000)
        
        # 상태 변수
        self.frames = []
        self.current_frame_index = 0
        self.tracker = EnhancedRegionTracker()
        self.current_region_points = []
        self.region_drawing_mode = False
        self.search_mode = False
        
        # UI 설정
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QHBoxLayout()
        
        # 왼쪽 패널 (컨트롤)
        left_panel = QVBoxLayout()
        
        # 파일 로딩
        file_group = QGroupBox("파일 관리")
        file_layout = QVBoxLayout()
        
        self.btn_load_video = QPushButton("동영상 불러오기")
        self.btn_load_video.clicked.connect(self.load_video)
        file_layout.addWidget(self.btn_load_video)
        
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)
        
        # 영역 관리
        region_group = QGroupBox("영역 관리")
        region_layout = QVBoxLayout()
        
        self.btn_add_region = QPushButton("영역 추가 모드")
        self.btn_add_region.clicked.connect(self.toggle_region_drawing)
        region_layout.addWidget(self.btn_add_region)
        
        self.btn_clear_regions = QPushButton("모든 영역 삭제")
        self.btn_clear_regions.clicked.connect(self.clear_regions)
        region_layout.addWidget(self.btn_clear_regions)
        
        self.region_list = QListWidget()
        self.region_list.setMaximumHeight(150)
        region_layout.addWidget(self.region_list)
        
        region_group.setLayout(region_layout)
        left_panel.addWidget(region_group)
        
        # 추적 설정
        tracking_group = QGroupBox("추적 설정")
        tracking_layout = QGridLayout()
        
        tracking_layout.addWidget(QLabel("추적 방법:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['optical_flow', 'template_matching', 'feature_matching', 'mean_shift'])
        tracking_layout.addWidget(self.method_combo, 0, 1)
        
        tracking_layout.addWidget(QLabel("신뢰도 임계값:"), 1, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(50, 95)
        self.threshold_spin.setValue(70)
        self.threshold_spin.setSuffix("%")
        tracking_layout.addWidget(self.threshold_spin, 1, 1)
        
        self.adaptive_check = QCheckBox("적응적 임계값")
        self.adaptive_check.setChecked(True)
        tracking_layout.addWidget(self.adaptive_check, 2, 0, 1, 2)
        
        tracking_group.setLayout(tracking_layout)
        left_panel.addWidget(tracking_group)
        
        # 검색 기능
        search_group = QGroupBox("영역 검색")
        search_layout = QVBoxLayout()
        
        self.btn_search_regions = QPushButton("현재 프레임에서 영역 검색")
        self.btn_search_regions.clicked.connect(self.search_regions)
        search_layout.addWidget(self.btn_search_regions)
        
        self.btn_search_all = QPushButton("전체 프레임 분석")
        self.btn_search_all.clicked.connect(self.analyze_all_frames)
        search_layout.addWidget(self.btn_search_all)
        
        search_group.setLayout(search_layout)
        left_panel.addWidget(search_group)
        
        # 재생 컨트롤
        playback_group = QGroupBox("재생 제어")
        playback_layout = QVBoxLayout()
        
        self.btn_play = QPushButton("재생/일시정지")
        self.btn_play.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.btn_play)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        playback_layout.addWidget(self.frame_slider)
        
        playback_group.setLayout(playback_layout)
        left_panel.addWidget(playback_group)
        
        left_panel.addStretch()
        
        # 오른쪽 패널 (이미지 표시)
        right_panel = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.mouseMoveEvent = self.on_image_drag
        self.image_label.mouseReleaseEvent = self.on_image_release
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        right_panel.addWidget(scroll_area)
        
        # 정보 표시
        self.info_label = QLabel("정보가 여기에 표시됩니다.")
        self.info_label.setMaximumHeight(100)
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        right_panel.addWidget(self.info_label)
        
        # 레이아웃 조합
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(350)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        self.setLayout(main_layout)
        
        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
    
    def load_video(self):
        """동영상 로딩"""
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
        
        if self.frames:
            self.frame_slider.setMaximum(len(self.frames) - 1)
            self.current_frame_index = 0
            self.show_current_frame()
            self.info_label.setText(f"동영상 로딩 완료: {len(self.frames)}개 프레임")
    
    def show_current_frame(self):
        """현재 프레임 표시"""
        if not self.frames:
            return
            
        frame = self.frames[self.current_frame_index].copy()
        
        # 등록된 영역들 그리기
        for i, region in enumerate(self.tracker.regions):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i % 5]
            
            # 영역 경계 그리기
            pts = np.array(region['points'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            
            # 신뢰도 및 ID 표시
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, f"ID:{region['id']} ({region['confidence']:.2f})", 
                       tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 추적 상태 표시
            if region['lost_count'] > 0:
                cv2.putText(frame, f"LOST:{region['lost_count']}", 
                           (centroid[0], centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 현재 그리고 있는 영역 표시
        if self.current_region_points:
            for i, point in enumerate(self.current_region_points):
                cv2.circle(frame, tuple(map(int, point)), 3, (255, 255, 255), -1)
                if i > 0:
                    cv2.line(frame, tuple(map(int, self.current_region_points[i-1])), 
                            tuple(map(int, point)), (255, 255, 255), 1)
        
        # 검색 결과 표시
        if hasattr(self, 'search_results') and self.search_results:
            for result in self.search_results:
                x, y, w, h = result['location']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(frame, f"Found: {result['confidence']:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Qt로 표시
        pixmap = cv_to_qpixmap(frame)
        self.image_label.setPixmap(pixmap)
        self.frame_slider.setValue(self.current_frame_index)
    
    def toggle_region_drawing(self):
        """영역 그리기 모드 토글"""
        self.region_drawing_mode = not self.region_drawing_mode
        if self.region_drawing_mode:
            self.btn_add_region.setText("영역 추가 완료")
            self.current_region_points = []
        else:
            self.btn_add_region.setText("영역 추가 모드")
            if len(self.current_region_points) >= 3:
                # 영역 추가
                region_id = self.tracker.add_region(self.frames[self.current_frame_index], self.current_region_points)
                self.update_region_list()
                self.info_label.setText(f"영역 {region_id} 추가됨")
            self.current_region_points = []
        self.show_current_frame()
    
    def on_image_click(self, event):
        """이미지 클릭 이벤트"""
        if self.region_drawing_mode:
            x, y = event.pos().x(), event.pos().y()
            self.current_region_points.append([x, y])
            self.show_current_frame()
    
    def on_image_drag(self, event):
        """이미지 드래그 이벤트"""
        pass
    
    def on_image_release(self, event):
        """이미지 릴리즈 이벤트"""
        pass
    
    def clear_regions(self):
        """모든 영역 삭제"""
        self.tracker.regions.clear()
        self.update_region_list()
        self.show_current_frame()
    
    def update_region_list(self):
        """영역 리스트 업데이트"""
        self.region_list.clear()
        for region in self.tracker.regions:
            item_text = f"영역 {region['id']} - {region['tracking_method']} (신뢰도: {region['confidence']:.2f})"
            self.region_list.addItem(item_text)
    
    def on_frame_change(self, value):
        """프레임 변경"""
        self.current_frame_index = value
        
        # 추적 업데이트
        if self.tracker.regions:
            self.tracker.update_regions(self.frames[self.current_frame_index], self.current_frame_index)
            self.update_region_list()
        
        self.show_current_frame()
    
    def toggle_playback(self):
        """재생/일시정지"""
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("재생")
        else:
            self.timer.start(100)  # 100ms 간격
            self.playing = True
            self.btn_play.setText("일시정지")
    
    def next_frame(self):
        """다음 프레임으로 이동"""
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.on_frame_change(self.current_frame_index)
        else:
            self.toggle_playback()
    
    def search_regions(self):
        """현재 프레임에서 영역 검색"""
        if not self.tracker.regions or not self.frames:
            QMessageBox.warning(self, "경고", "등록된 영역과 프레임이 필요합니다.")
            return
        
        threshold = self.threshold_spin.value() / 100.0
        self.search_results = self.tracker.search_regions_in_frame(
            self.frames[self.current_frame_index], threshold
        )
        
        self.show_current_frame()
        self.info_label.setText(f"검색 완료: {len(self.search_results)}개 영역 발견")
    
    def analyze_all_frames(self):
        """전체 프레임 분석"""
        if not self.tracker.regions or not self.frames:
            QMessageBox.warning(self, "경고", "등록된 영역과 프레임이 필요합니다.")
            return
        
        QMessageBox.information(self, "분석 시작", "전체 프레임 분석을 시작합니다. 시간이 오래 걸릴 수 있습니다.")
        
        detection_results = {}
        threshold = self.threshold_spin.value() / 100.0
        
        for frame_idx, frame in enumerate(self.frames):
            results = self.tracker.search_regions_in_frame(frame, threshold)
            if results:
                detection_results[frame_idx] = results
            
            # 진행률 표시 (간단히)
            if frame_idx % 10 == 0:
                self.info_label.setText(f"분석 진행중... {frame_idx}/{len(self.frames)}")
                QApplication.processEvents()
        
        # 결과 요약
        total_detections = sum(len(results) for results in detection_results.values())
        self.info_label.setText(
            f"분석 완료: {len(detection_results)}개 프레임에서 총 {total_detections}개 영역 발견"
        )
        
        # 결과를 파일로 저장할 수도 있음
        QMessageBox.information(self, "분석 완료", 
                               f"전체 분석이 완료되었습니다.\n"
                               f"검출된 프레임: {len(detection_results)}개\n"
                               f"총 검출 횟수: {total_detections}개")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RegionTrackingApp()
    win.show()
    sys.exit(app.exec_())