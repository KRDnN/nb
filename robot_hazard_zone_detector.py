import sys
import cv2
import numpy as np
import os
import math
import json
from collections import defaultdict, deque
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QListWidget, QListWidgetItem, QMessageBox, QDialog, 
    QScrollArea, QCheckBox, QSpinBox, QComboBox, QGroupBox, QGridLayout,
    QTextEdit, QTabWidget, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# ---------- 유틸 함수 -----------
def cv_to_qpixmap(cv_img):
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(cv_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def compute_homography_confidence(matches, src_pts, dst_pts, H):
    """호모그래피 변환의 신뢰도 계산"""
    if len(matches) < 10:
        return 0.0
    
    # RANSAC inlier 비율
    src_transformed = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)
    distances = np.linalg.norm(src_transformed.reshape(-1, 2) - dst_pts.reshape(-1, 2), axis=1)
    inliers = np.sum(distances < 5.0)
    inlier_ratio = inliers / len(matches)
    
    # 매칭 품질 (거리 기반)
    match_quality = 1.0 - (np.mean([m.distance for m in matches]) / 256.0)
    
    # 기하학적 일관성
    geometric_score = min(1.0, inliers / 20.0)  # 20개 이상의 inlier면 1.0
    
    confidence = (inlier_ratio * 0.4 + match_quality * 0.3 + geometric_score * 0.3)
    return min(1.0, confidence)

def extract_environmental_features(image):
    """환경 특징 추출 (SLAM용)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 1. ORB 특징점 (회전/스케일 불변)
    orb = cv2.ORB_create(2000)  # 더 많은 특징점
    kp_orb, desc_orb = orb.detectAndCompute(gray, None)
    
    # 2. SIFT 특징점 (고품질, 스케일 불변)
    try:
        sift = cv2.SIFT_create(1000)
        kp_sift, desc_sift = sift.detectAndCompute(gray, None)
    except:
        kp_sift, desc_sift = [], None
    
    # 3. 코너 검출 (Harris, Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    
    # 4. 선분 검출 (건물 윤곽, 도로 등)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # 5. 수직/수평 구조 검출 (건물 특징)
    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if abs(angle) < 15 or abs(angle) > 165:  # 수평선
                horizontal_lines.append(line[0])
            elif abs(abs(angle) - 90) < 15:  # 수직선
                vertical_lines.append(line[0])
    
    return {
        'orb_kp': kp_orb,
        'orb_desc': desc_orb,
        'sift_kp': kp_sift,
        'sift_desc': desc_sift,
        'corners': corners if corners is not None else np.array([]),
        'lines': lines if lines is not None else np.array([]),
        'vertical_lines': vertical_lines,
        'horizontal_lines': horizontal_lines,
        'image_size': gray.shape
    }

def match_features_robust(desc1, desc2, ratio_thresh=0.7):
    """강건한 특징점 매칭 (Lowe's ratio test 적용)"""
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # ORB의 경우 NORM_HAMMING, SIFT의 경우 NORM_L2
    if desc1.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    try:
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    except:
        return []

# --------- 위험구역 데이터베이스 ---------
class HazardZoneDatabase:
    def __init__(self):
        self.zones = []  # 위험구역 리스트
        self.global_features = []  # 전역 특징점 맵
        self.zone_counter = 0
        
    def add_hazard_zone(self, reference_image, zone_polygon, zone_name="", description=""):
        """위험구역 추가"""
        zone_id = self.zone_counter
        self.zone_counter += 1
        
        # 영역 마스크 생성
        mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)
        pts = np.array(zone_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # 주변 환경 특징 추출 (위험구역 + 주변 컨텍스트)
        expanded_mask = cv2.dilate(mask, np.ones((50, 50), np.uint8))  # 주변 50px 포함
        
        # 전체 이미지에서 특징 추출
        env_features = extract_environmental_features(reference_image)
        
        # 위험구역 특징 추출
        zone_features = extract_environmental_features(reference_image)
        
        # 위험구역 내부 특징만 필터링
        zone_orb_kp = []
        zone_orb_desc = []
        if env_features['orb_kp'] and env_features['orb_desc'] is not None:
            for i, kp in enumerate(env_features['orb_kp']):
                if mask[int(kp.pt[1]), int(kp.pt[0])] > 0:
                    zone_orb_kp.append(kp)
                    zone_orb_desc.append(env_features['orb_desc'][i])
        
        # 주변 컨텍스트 특징 (위험구역 인식을 위한 랜드마크)
        context_orb_kp = []
        context_orb_desc = []
        if env_features['orb_kp'] and env_features['orb_desc'] is not None:
            for i, kp in enumerate(env_features['orb_kp']):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if expanded_mask[y, x] > 0 and mask[y, x] == 0:  # 주변에 있지만 위험구역은 아님
                    context_orb_kp.append(kp)
                    context_orb_desc.append(env_features['orb_desc'][i])
        
        hazard_zone = {
            'id': zone_id,
            'name': zone_name or f"위험구역_{zone_id}",
            'description': description,
            'polygon': zone_polygon,
            'reference_image': reference_image.copy(),
            'mask': mask,
            'expanded_mask': expanded_mask,
            
            # 환경 특징
            'env_features': env_features,
            
            # 위험구역 내부 특징
            'zone_orb_kp': zone_orb_kp,
            'zone_orb_desc': np.array(zone_orb_desc) if zone_orb_desc else np.array([]),
            
            # 컨텍스트 특징 (주변 랜드마크)
            'context_orb_kp': context_orb_kp,
            'context_orb_desc': np.array(context_orb_desc) if context_orb_desc else np.array([]),
            
            # 메타데이터
            'creation_time': None,
            'detection_count': 0,
            'false_positive_count': 0,
            'confidence_threshold': 0.3
        }
        
        self.zones.append(hazard_zone)
        return zone_id
    
    def detect_zones_in_image(self, current_image, min_confidence=0.3):
        """현재 이미지에서 위험구역 검출"""
        detections = []
        current_features = extract_environmental_features(current_image)
        
        for zone in self.zones:
            detection = self._match_zone_with_image(zone, current_image, current_features)
            if detection and detection['confidence'] >= min_confidence:
                detections.append(detection)
        
        return detections
    
    def _match_zone_with_image(self, zone, current_image, current_features):
        """단일 위험구역과 현재 이미지 매칭"""
        
        # 1. 전역 특징 매칭 (ORB)
        orb_matches = match_features_robust(
            zone['env_features']['orb_desc'], 
            current_features['orb_desc']
        )
        
        # 2. SIFT 매칭 (사용 가능한 경우)
        sift_matches = []
        if (zone['env_features']['sift_desc'] is not None and 
            current_features['sift_desc'] is not None):
            sift_matches = match_features_robust(
                zone['env_features']['sift_desc'],
                current_features['sift_desc']
            )
        
        # 최소 매칭 수 확인
        total_matches = len(orb_matches) + len(sift_matches)
        if total_matches < 10:
            return None
        
        # 3. 호모그래피 계산
        homography_result = self._compute_scene_homography(
            zone, current_image, current_features, orb_matches, sift_matches
        )
        
        if not homography_result:
            return None
        
        H, confidence, inlier_matches = homography_result
        
        # 4. 위험구역 투영
        zone_projection = self._project_hazard_zone(zone, H, current_image.shape)
        
        if not zone_projection:
            return None
        
        # 5. 검증 단계
        verification_score = self._verify_detection(
            zone, current_image, zone_projection, current_features
        )
        
        final_confidence = confidence * verification_score
        
        if final_confidence < zone['confidence_threshold']:
            return None
        
        return {
            'zone_id': zone['id'],
            'zone_name': zone['name'],
            'confidence': final_confidence,
            'projected_polygon': zone_projection['polygon'],
            'homography': H,
            'match_count': len(inlier_matches),
            'detection_method': 'homography_projection'
        }
    
    def _compute_scene_homography(self, zone, current_image, current_features, orb_matches, sift_matches):
        """장면 간 호모그래피 계산"""
        
        # ORB 매칭 포인트 추출
        orb_src_pts = []
        orb_dst_pts = []
        for match in orb_matches:
            src_pt = zone['env_features']['orb_kp'][match.queryIdx].pt
            dst_pt = current_features['orb_kp'][match.trainIdx].pt
            orb_src_pts.append(src_pt)
            orb_dst_pts.append(dst_pt)
        
        # SIFT 매칭 포인트 추가
        sift_src_pts = []
        sift_dst_pts = []
        for match in sift_matches:
            src_pt = zone['env_features']['sift_kp'][match.queryIdx].pt
            dst_pt = current_features['sift_kp'][match.trainIdx].pt
            sift_src_pts.append(src_pt)
            sift_dst_pts.append(dst_pt)
        
        # 전체 매칭점 결합
        all_src_pts = np.array(orb_src_pts + sift_src_pts, dtype=np.float32)
        all_dst_pts = np.array(orb_dst_pts + sift_dst_pts, dtype=np.float32)
        
        if len(all_src_pts) < 4:
            return None
        
        # 호모그래피 계산 (RANSAC)
        H, mask = cv2.findHomography(
            all_src_pts, all_dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=2000
        )
        
        if H is None:
            return None
        
        # 신뢰도 계산
        inlier_matches = []
        for i, is_inlier in enumerate(mask.ravel()):
            if is_inlier:
                if i < len(orb_matches):
                    inlier_matches.append(orb_matches[i])
                else:
                    inlier_matches.append(sift_matches[i - len(orb_matches)])
        
        confidence = compute_homography_confidence(
            inlier_matches, all_src_pts[mask.ravel() == 1], 
            all_dst_pts[mask.ravel() == 1], H
        )
        
        return H, confidence, inlier_matches
    
    def _project_hazard_zone(self, zone, H, current_shape):
        """위험구역을 현재 이미지로 투영"""
        try:
            # 위험구역 폴리곤 투영
            zone_poly = np.array(zone['polygon'], dtype=np.float32).reshape(-1, 1, 2)
            projected_poly = cv2.perspectiveTransform(zone_poly, H)
            projected_poly = projected_poly.reshape(-1, 2)
            
            # 투영된 폴리곤이 이미지 범위 내에 있는지 확인
            h, w = current_shape[:2]
            valid_projection = True
            
            for pt in projected_poly:
                if pt[0] < -w*0.5 or pt[0] > w*1.5 or pt[1] < -h*0.5 or pt[1] > h*1.5:
                    valid_projection = False
                    break
            
            if not valid_projection:
                return None
            
            # 투영된 영역 마스크 생성
            projected_mask = np.zeros(current_shape[:2], dtype=np.uint8)
            projected_poly_int = projected_poly.astype(np.int32)
            cv2.fillPoly(projected_mask, [projected_poly_int], 255)
            
            return {
                'polygon': projected_poly,
                'mask': projected_mask,
                'area': cv2.contourArea(projected_poly_int)
            }
            
        except Exception as e:
            print(f"투영 오류: {e}")
            return None
    
    def _verify_detection(self, zone, current_image, projection, current_features):
        """검출 결과 검증"""
        verification_score = 1.0
        
        # 1. 영역 크기 검증 (너무 크거나 작으면 오탐지 가능성)
        original_area = cv2.contourArea(np.array(zone['polygon'], dtype=np.int32))
        projected_area = projection['area']
        
        if projected_area > 0:
            scale_ratio = projected_area / original_area
            if scale_ratio < 0.1 or scale_ratio > 10:  # 10배 이상 차이나면 의심
                verification_score *= 0.5
        
        # 2. 위험구역 내부 특징 검증
        if len(zone['zone_orb_desc']) > 0:
            zone_matches = match_features_robust(
                zone['zone_orb_desc'],
                current_features['orb_desc']
            )
            
            # 투영된 영역 내에 얼마나 많은 특징점이 매칭되는지 확인
            matches_in_zone = 0
            for match in zone_matches:
                dst_pt = current_features['orb_kp'][match.trainIdx].pt
                if cv2.pointPolygonTest(projection['polygon'], dst_pt, False) >= 0:
                    matches_in_zone += 1
            
            if len(zone_matches) > 0:
                zone_match_ratio = matches_in_zone / len(zone_matches)
                verification_score *= (0.5 + 0.5 * zone_match_ratio)
        
        # 3. 컨텍스트 검증 (주변 랜드마크)
        if len(zone['context_orb_desc']) > 0:
            context_matches = match_features_robust(
                zone['context_orb_desc'],
                current_features['orb_desc']
            )
            
            if len(context_matches) > 5:  # 충분한 컨텍스트 매칭
                verification_score *= 1.2
            elif len(context_matches) < 2:  # 컨텍스트 부족
                verification_score *= 0.7
        
        return min(1.0, verification_score)
    
    def save_database(self, filepath):
        """데이터베이스 저장"""
        try:
            data = {
                'zones': [],
                'zone_counter': self.zone_counter
            }
            
            for zone in self.zones:
                zone_data = {
                    'id': zone['id'],
                    'name': zone['name'],
                    'description': zone['description'],
                    'polygon': zone['polygon'],
                    'confidence_threshold': zone['confidence_threshold'],
                    'detection_count': zone['detection_count'],
                    'false_positive_count': zone['false_positive_count']
                }
                data['zones'].append(zone_data)
                
                # 이미지와 특징 데이터는 별도 파일로 저장
                zone_dir = os.path.join(os.path.dirname(filepath), f"zone_{zone['id']}")
                os.makedirs(zone_dir, exist_ok=True)
                
                cv2.imwrite(os.path.join(zone_dir, "reference.jpg"), zone['reference_image'])
                cv2.imwrite(os.path.join(zone_dir, "mask.png"), zone['mask'])
                
                # 특징점 데이터 저장
                features_file = os.path.join(zone_dir, "features.npz")
                np.savez_compressed(features_file,
                    orb_desc=zone['env_features']['orb_desc'],
                    sift_desc=zone['env_features']['sift_desc'],
                    zone_orb_desc=zone['zone_orb_desc'],
                    context_orb_desc=zone['context_orb_desc']
                )
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"저장 오류: {e}")
            return False
    
    def load_database(self, filepath):
        """데이터베이스 로드"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.zones.clear()
            self.zone_counter = data.get('zone_counter', 0)
            
            for zone_data in data['zones']:
                zone_dir = os.path.join(os.path.dirname(filepath), f"zone_{zone_data['id']}")
                
                # 이미지 로드
                reference_image = cv2.imread(os.path.join(zone_dir, "reference.jpg"))
                mask = cv2.imread(os.path.join(zone_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
                
                # 특징점 데이터 로드
                features_file = os.path.join(zone_dir, "features.npz")
                if os.path.exists(features_file):
                    features_data = np.load(features_file, allow_pickle=True)
                    
                    zone = {
                        'id': zone_data['id'],
                        'name': zone_data['name'],
                        'description': zone_data['description'],
                        'polygon': zone_data['polygon'],
                        'reference_image': reference_image,
                        'mask': mask,
                        'confidence_threshold': zone_data.get('confidence_threshold', 0.3),
                        'detection_count': zone_data.get('detection_count', 0),
                        'false_positive_count': zone_data.get('false_positive_count', 0),
                        'env_features': {
                            'orb_desc': features_data['orb_desc'],
                            'sift_desc': features_data['sift_desc']
                        },
                        'zone_orb_desc': features_data['zone_orb_desc'],
                        'context_orb_desc': features_data['context_orb_desc']
                    }
                    
                    self.zones.append(zone)
            
            return True
        except Exception as e:
            print(f"로드 오류: {e}")
            return False

# --------- 메인 애플리케이션 ---------
class RobotHazardDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로봇 감시 시스템 - 위험구역 인식")
        self.resize(1600, 1000)
        
        # 핵심 컴포넌트
        self.hazard_db = HazardZoneDatabase()
        self.current_frame = None
        self.current_detections = []
        
        # UI 상태
        self.frames = []
        self.current_frame_index = 0
        self.region_drawing_mode = False
        self.current_region_points = []
        
        # 설정
        self.detection_confidence = 0.3
        self.auto_detection = False
        
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QHBoxLayout()
        
        # 왼쪽 제어 패널
        left_panel = self.create_control_panel()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(400)
        
        # 오른쪽 표시 패널
        right_panel = self.create_display_panel()
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)
        
        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        
    def create_control_panel(self):
        layout = QVBoxLayout()
        
        # 탭 위젯 생성
        tab_widget = QTabWidget()
        
        # 1. 데이터베이스 관리 탭
        db_tab = QWidget()
        db_layout = QVBoxLayout()
        
        # 파일 관리
        file_group = QGroupBox("데이터베이스 관리")
        file_layout = QVBoxLayout()
        
        self.btn_load_video = QPushButton("테스트 영상 불러오기")
        self.btn_load_video.clicked.connect(self.load_video)
        file_layout.addWidget(self.btn_load_video)
        
        self.btn_save_db = QPushButton("위험구역 DB 저장")
        self.btn_save_db.clicked.connect(self.save_database)
        file_layout.addWidget(self.btn_save_db)
        
        self.btn_load_db = QPushButton("위험구역 DB 불러오기")
        self.btn_load_db.clicked.connect(self.load_database)
        file_layout.addWidget(self.btn_load_db)
        
        file_group.setLayout(file_layout)
        db_layout.addWidget(file_group)
        
        # 위험구역 관리
        zone_group = QGroupBox("위험구역 관리")
        zone_layout = QVBoxLayout()
        
        self.btn_add_zone = QPushButton("위험구역 추가")
        self.btn_add_zone.clicked.connect(self.toggle_zone_drawing)
        zone_layout.addWidget(self.btn_add_zone)
        
        self.zone_list = QListWidget()
        self.zone_list.setMaximumHeight(200)
        zone_layout.addWidget(self.zone_list)
        
        self.btn_delete_zone = QPushButton("선택된 구역 삭제")
        self.btn_delete_zone.clicked.connect(self.delete_selected_zone)
        zone_layout.addWidget(self.btn_delete_zone)
        
        zone_group.setLayout(zone_layout)
        db_layout.addWidget(zone_group)
        
        db_tab.setLayout(db_layout)
        tab_widget.addTab(db_tab, "데이터베이스")
        
        # 2. 감지 설정 탭
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()
        
        # 감지 설정
        detection_group = QGroupBox("감지 설정")
        detection_grid = QGridLayout()
        
        detection_grid.addWidget(QLabel("신뢰도 임계값:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(30)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        detection_grid.addWidget(self.confidence_slider, 0, 1)
        
        self.confidence_label = QLabel("0.30")
        detection_grid.addWidget(self.confidence_label, 0, 2)
        
        self.auto_detect_check = QCheckBox("실시간 자동 감지")
        self.auto_detect_check.toggled.connect(self.toggle_auto_detection)
        detection_grid.addWidget(self.auto_detect_check, 1, 0, 1, 3)
        
        detection_group.setLayout(detection_grid)
        detection_layout.addWidget(detection_group)
        
        # 수동 감지 버튼
        self.btn_detect_now = QPushButton("현재 프레임 감지")
        self.btn_detect_now.clicked.connect(self.detect_current_frame)
        detection_layout.addWidget(self.btn_detect_now)
        
        # 전체 영상 분석
        self.btn_analyze_video = QPushButton("전체 영상 분석")
        self.btn_analyze_video.clicked.connect(self.analyze_full_video)
        detection_layout.addWidget(self.btn_analyze_video)
        
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        detection_layout.addWidget(self.analysis_progress)
        
        detection_tab.setLayout(detection_layout)
        tab_widget.addTab(detection_tab, "감지 설정")
        
        # 3. 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(300)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(QLabel("감지 로그:"))
        log_layout.addWidget(self.log_text)
        
        self.btn_clear_log = QPushButton("로그 지우기")
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        log_layout.addWidget(self.btn_clear_log)
        
        log_tab.setLayout(log_layout)
        tab_widget.addTab(log_tab, "로그")
        
        layout.addWidget(tab_widget)
        
        # 재생 컨트롤
        playback_group = QGroupBox("재생 제어")
        playback_layout = QVBoxLayout()
        
        self.btn_play = QPushButton("재생/일시정지")
        self.btn_play.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.btn_play)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        playback_layout.addWidget(self.frame_slider)
        
        self.frame_info = QLabel("프레임: 0/0")
        playback_layout.addWidget(self.frame_info)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        return layout
    
    def create_display_panel(self):
        layout = QVBoxLayout()
        
        # 메인 이미지 표시
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.mouseMoveEvent = self.on_image_drag
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
        
        # 상태 정보
        self.status_label = QLabel("시스템 준비")
        self.status_label.setStyleSheet("background-color: #e0e0e0; padding: 10px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        return layout
    
    def load_video(self):
        """테스트 영상 로딩"""
        fname, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "Videos (*.mp4 *.avi *.mov)")
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
            self.update_status(f"영상 로딩 완료: {len(self.frames)}개 프레임")
            self.log(f"영상 파일 로딩: {os.path.basename(fname)} ({len(self.frames)} 프레임)")
    
    def show_current_frame(self):
        """현재 프레임 표시"""
        if not self.frames:
            return
            
        self.current_frame = self.frames[self.current_frame_index].copy()
        display_frame = self.current_frame.copy()
        
        # 등록된 위험구역 표시 (반투명)
        for zone in self.hazard_db.zones:
            pts = np.array(zone['polygon'], dtype=np.int32)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))  # 빨간색
            cv2.addWeighted(display_frame, 0.8, overlay, 0.2, 0, display_frame)
            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
            
            # 구역 이름 표시
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(display_frame, zone['name'], tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 현재 감지된 위험구역 표시
        for detection in self.current_detections:
            pts = np.array(detection['projected_polygon'], dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)  # 초록색 두꺼운 선
            
            # 감지 정보 표시
            centroid = np.mean(pts, axis=0).astype(int)
            text = f"[감지] {detection['zone_name']} ({detection['confidence']:.2f})"
            cv2.putText(display_frame, text, tuple(centroid - [0, 20]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 현재 그리고 있는 구역 표시
        if self.current_region_points:
            for i, point in enumerate(self.current_region_points):
                cv2.circle(display_frame, tuple(map(int, point)), 4, (255, 255, 0), -1)
                if i > 0:
                    cv2.line(display_frame, tuple(map(int, self.current_region_points[i-1])), 
                            tuple(map(int, point)), (255, 255, 0), 2)
        
        # Qt 표시
        pixmap = cv_to_qpixmap(display_frame)
        self.image_label.setPixmap(pixmap)
        
        # 프레임 정보 업데이트
        self.frame_info.setText(f"프레임: {self.current_frame_index + 1}/{len(self.frames)}")
        self.frame_slider.setValue(self.current_frame_index)
        
        # 자동 감지 활성화 시 감지 수행
        if self.auto_detection:
            self.detect_current_frame()
    
    def toggle_zone_drawing(self):
        """위험구역 그리기 모드 토글"""
        self.region_drawing_mode = not self.region_drawing_mode
        
        if self.region_drawing_mode:
            self.btn_add_zone.setText("구역 추가 완료")
            self.current_region_points = []
            self.update_status("위험구역을 클릭하여 지정하세요 (최소 3개 점)")
        else:
            self.btn_add_zone.setText("위험구역 추가")
            
            if len(self.current_region_points) >= 3:
                # 구역 추가 다이얼로그
                zone_name, ok = self.get_zone_info()
                if ok:
                    zone_id = self.hazard_db.add_hazard_zone(
                        self.current_frame, 
                        self.current_region_points,
                        zone_name,
                        f"프레임 {self.current_frame_index}에서 생성"
                    )
                    self.update_zone_list()
                    self.update_status(f"위험구역 '{zone_name}' 추가됨 (ID: {zone_id})")
                    self.log(f"위험구역 추가: {zone_name} (ID: {zone_id})")
            else:
                self.update_status("최소 3개의 점이 필요합니다")
            
            self.current_region_points = []
        
        self.show_current_frame()
    
    def get_zone_info(self):
        """위험구역 정보 입력 다이얼로그"""
        from PyQt5.QtWidgets import QInputDialog
        
        zone_name, ok = QInputDialog.getText(
            self, '위험구역 이름', '위험구역 이름을 입력하세요:',
            text=f'위험구역_{len(self.hazard_db.zones) + 1}'
        )
        
        return zone_name, ok
    
    def on_image_click(self, event):
        """이미지 클릭 이벤트"""
        if self.region_drawing_mode and self.current_frame is not None:
            x, y = event.pos().x(), event.pos().y()
            
            # 이미지 좌표로 변환 (스케일링 고려)
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_size = self.image_label.size()
                pixmap_size = pixmap.size()
                
                scale_x = self.current_frame.shape[1] / pixmap_size.width()
                scale_y = self.current_frame.shape[0] / pixmap_size.height()
                
                # 이미지가 라벨 중앙에 표시되므로 오프셋 계산
                offset_x = (label_size.width() - pixmap_size.width()) / 2
                offset_y = (label_size.height() - pixmap_size.height()) / 2
                
                real_x = (x - offset_x) * scale_x
                real_y = (y - offset_y) * scale_y
                
                if 0 <= real_x < self.current_frame.shape[1] and 0 <= real_y < self.current_frame.shape[0]:
                    self.current_region_points.append([real_x, real_y])
                    self.show_current_frame()
    
    def on_image_drag(self, event):
        """이미지 드래그 이벤트"""
        pass
    
    def detect_current_frame(self):
        """현재 프레임에서 위험구역 감지"""
        if self.current_frame is None or not self.hazard_db.zones:
            return
        
        try:
            detections = self.hazard_db.detect_zones_in_image(
                self.current_frame, 
                self.detection_confidence
            )
            
            self.current_detections = detections
            
            if detections:
                detection_info = []
                for det in detections:
                    detection_info.append(f"{det['zone_name']} (신뢰도: {det['confidence']:.3f})")
                    self.log(f"[경고] 프레임 {self.current_frame_index}: {det['zone_name']} 감지됨 (신뢰도: {det['confidence']:.3f})")
                
                self.update_status(f"위험구역 감지: {', '.join(detection_info)}")
            else:
                self.current_detections = []
                
            self.show_current_frame()
            
        except Exception as e:
            self.update_status(f"감지 오류: {str(e)}")
            self.log(f"감지 오류: {str(e)}")
    
    def analyze_full_video(self):
        """전체 영상 분석"""
        if not self.frames or not self.hazard_db.zones:
            QMessageBox.warning(self, "경고", "영상과 위험구역이 모두 필요합니다.")
            return
        
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setMaximum(len(self.frames))
        
        detection_summary = defaultdict(list)
        
        for i, frame in enumerate(self.frames):
            self.analysis_progress.setValue(i)
            QApplication.processEvents()
            
            detections = self.hazard_db.detect_zones_in_image(frame, self.detection_confidence)
            
            if detections:
                for det in detections:
                    detection_summary[det['zone_name']].append({
                        'frame': i,
                        'confidence': det['confidence']
                    })
        
        self.analysis_progress.setVisible(False)
        
        # 결과 요약
        summary_text = "=== 전체 영상 분석 결과 ===\n"
        for zone_name, detections in detection_summary.items():
            summary_text += f"\n{zone_name}:\n"
            summary_text += f"  총 감지 횟수: {len(detections)}\n"
            summary_text += f"  평균 신뢰도: {np.mean([d['confidence'] for d in detections]):.3f}\n"
            summary_text += f"  감지 프레임: {[d['frame'] for d in detections[:10]]}{'...' if len(detections) > 10 else ''}\n"
        
        self.log(summary_text)
        QMessageBox.information(self, "분석 완료", f"분석이 완료되었습니다.\n총 {sum(len(d) for d in detection_summary.values())}건의 위험구역 감지")
    
    def update_confidence(self, value):
        """신뢰도 임계값 업데이트"""
        self.detection_confidence = value / 100.0
        self.confidence_label.setText(f"{self.detection_confidence:.2f}")
        
        # 자동 감지 중이면 즉시 재감지
        if self.auto_detection:
            self.detect_current_frame()
    
    def toggle_auto_detection(self, checked):
        """자동 감지 토글"""
        self.auto_detection = checked
        if checked:
            self.update_status("자동 감지 활성화")
            self.detect_current_frame()
        else:
            self.update_status("자동 감지 비활성화")
            self.current_detections = []
            self.show_current_frame()
    
    def update_zone_list(self):
        """위험구역 리스트 업데이트"""
        self.zone_list.clear()
        for zone in self.hazard_db.zones:
            item_text = f"[{zone['id']}] {zone['name']} (임계값: {zone['confidence_threshold']:.2f})"
            self.zone_list.addItem(item_text)
    
    def delete_selected_zone(self):
        """선택된 위험구역 삭제"""
        current_row = self.zone_list.currentRow()
        if current_row >= 0:
            zone = self.hazard_db.zones[current_row]
            reply = QMessageBox.question(
                self, '확인', f"위험구역 '{zone['name']}'을 삭제하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                deleted_zone = self.hazard_db.zones.pop(current_row)
                self.update_zone_list()
                self.show_current_frame()
                self.update_status(f"위험구역 '{deleted_zone['name']}' 삭제됨")
                self.log(f"위험구역 삭제: {deleted_zone['name']}")
    
    def save_database(self):
        """데이터베이스 저장"""
        if not self.hazard_db.zones:
            QMessageBox.warning(self, "경고", "저장할 위험구역이 없습니다.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "위험구역 DB 저장", "", "JSON Files (*.json)"
        )
        
        if filepath:
            if self.hazard_db.save_database(filepath):
                QMessageBox.information(self, "저장 완료", "위험구역 데이터베이스가 저장되었습니다.")
                self.log(f"데이터베이스 저장: {filepath}")
            else:
                QMessageBox.critical(self, "저장 실패", "데이터베이스 저장에 실패했습니다.")
    
    def load_database(self):
        """데이터베이스 로드"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "위험구역 DB 불러오기", "", "JSON Files (*.json)"
        )
        
        if filepath:
            if self.hazard_db.load_database(filepath):
                self.update_zone_list()
                self.show_current_frame()
                QMessageBox.information(self, "로드 완료", f"위험구역 {len(self.hazard_db.zones)}개가 로드되었습니다.")
                self.log(f"데이터베이스 로드: {filepath} ({len(self.hazard_db.zones)}개 구역)")
            else:
                QMessageBox.critical(self, "로드 실패", "데이터베이스 로드에 실패했습니다.")
    
    def on_frame_change(self, value):
        """프레임 변경"""
        self.current_frame_index = value
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
            self.show_current_frame()
        else:
            self.toggle_playback()
    
    def update_status(self, message):
        """상태 메시지 업데이트"""
        self.status_label.setText(message)
    
    def log(self, message):
        """로그 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RobotHazardDetectionApp()
    win.show()
    sys.exit(app.exec_())