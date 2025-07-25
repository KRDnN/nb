#!/usr/bin/env python3
"""
Enhanced Robot Hazard Detection System
=================================
논문 기반 최신 기술 통합:
- YOLO 실시간 객체 감지 (YOLOv8/v10)
- 딥러닝 기반 PPE 검출
- 다중 센서 융합 (RGB + Depth + IMU)
- SLAM 기반 공간 매핑
- 실시간 성능 최적화
- 안전 관리 시스템
"""

import sys
import cv2
import numpy as np
import os
import math
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QListWidget, QListWidgetItem, QMessageBox, QDialog, 
    QScrollArea, QCheckBox, QSpinBox, QComboBox, QGroupBox, QGridLayout,
    QTextEdit, QTabWidget, QProgressBar, QSplitter, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition

# Deep Learning imports (조건부 import)
try:
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO 라이브러리가 설치되지 않음 - 기본 CV 모드로 실행")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ================ 데이터 구조 ================

class AlertLevel(Enum):
    """경고 수준"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class DetectionResult:
    """감지 결과 데이터"""
    zone_id: int
    zone_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    polygon: List[Tuple[float, float]]
    timestamp: float
    alert_level: AlertLevel
    detection_method: str
    additional_info: Dict[str, Any]

@dataclass
class SensorData:
    """센서 데이터"""
    rgb_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    imu_data: Optional[Dict[str, float]] = None
    gps_data: Optional[Dict[str, float]] = None
    timestamp: float = 0.0

@dataclass
class PPEDetection:
    """개인보호장비 감지 결과"""
    person_bbox: Tuple[int, int, int, int]
    helmet_detected: bool
    vest_detected: bool
    gloves_detected: bool
    boots_detected: bool
    mask_detected: bool
    confidence_scores: Dict[str, float]

# ================ YOLO 기반 실시간 감지기 ================

class YOLODetector:
    """YOLO 기반 실시간 객체 감지"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
        if YOLO_AVAILABLE:
            try:
                # YOLOv8 또는 사용자 지정 모델 로드
                if model_path and os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # 기본 YOLOv8n 모델 사용
                    self.model = YOLO('yolov8n.pt')
                
                self.model_loaded = True
                print(f"YOLO 모델 로드 완료: {self.device}")
                
                # 클래스 이름 매핑
                self.class_names = self.model.names
                
            except Exception as e:
                print(f"YOLO 모델 로드 실패: {e}")
                self.model_loaded = False
    
    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """객체 감지 수행"""
        if not self.model_loaded:
            return []
        
        try:
            # YOLO 추론
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 바운딩 박스 정보 추출
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': float(conf),
                            'class_id': cls,
                            'class_name': self.class_names.get(cls, f'class_{cls}'),
                            'center': ((x1+x2)/2, (y1+y2)/2)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"YOLO 감지 오류: {e}")
            return []
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """사람 감지 (PPE 검출 준비)"""
        all_detections = self.detect_objects(frame, conf_threshold=0.3)
        # COCO 데이터셋에서 사람은 클래스 ID 0
        person_detections = [d for d in all_detections if d['class_id'] == 0]
        return person_detections

class PPEDetector:
    """개인보호장비(PPE) 감지기"""
    
    def __init__(self):
        self.yolo_detector = YOLODetector()
        
        # PPE 클래스 매핑 (사용자 정의 모델의 경우)
        self.ppe_classes = {
            'helmet': ['helmet', 'hard_hat', 'safety_helmet'],
            'vest': ['safety_vest', 'high_vis_vest', 'reflective_vest'],
            'gloves': ['safety_gloves', 'work_gloves'],
            'boots': ['safety_boots', 'steel_toe_boots'],
            'mask': ['face_mask', 'respirator', 'n95']
        }
        
        # 색상 기반 PPE 감지 (보조 방법)
        self.vest_color_ranges = {
            'orange': ([5, 50, 50], [25, 255, 255]),
            'yellow': ([25, 50, 50], [35, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255])
        }
    
    def detect_ppe(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> PPEDetection:
        """특정 사람의 PPE 감지"""
        x, y, w, h = person_bbox
        person_roi = frame[y:y+h, x:x+w]
        
        # YOLO 기반 PPE 감지
        ppe_detections = self.yolo_detector.detect_objects(person_roi, conf_threshold=0.3)
        
        # PPE 항목별 분석
        helmet_detected = self._check_ppe_item(ppe_detections, 'helmet')
        vest_detected = self._check_ppe_item(ppe_detections, 'vest') or self._check_vest_by_color(person_roi)
        gloves_detected = self._check_ppe_item(ppe_detections, 'gloves')
        boots_detected = self._check_ppe_item(ppe_detections, 'boots')
        mask_detected = self._check_ppe_item(ppe_detections, 'mask')
        
        # 신뢰도 점수 계산
        confidence_scores = {
            'helmet': self._get_ppe_confidence(ppe_detections, 'helmet'),
            'vest': self._get_ppe_confidence(ppe_detections, 'vest'),
            'gloves': self._get_ppe_confidence(ppe_detections, 'gloves'),
            'boots': self._get_ppe_confidence(ppe_detections, 'boots'),
            'mask': self._get_ppe_confidence(ppe_detections, 'mask')
        }
        
        return PPEDetection(
            person_bbox=person_bbox,
            helmet_detected=helmet_detected,
            vest_detected=vest_detected,
            gloves_detected=gloves_detected,
            boots_detected=boots_detected,
            mask_detected=mask_detected,
            confidence_scores=confidence_scores
        )
    
    def _check_ppe_item(self, detections: List[Dict], ppe_type: str) -> bool:
        """특정 PPE 항목 확인"""
        target_classes = self.ppe_classes.get(ppe_type, [])
        for detection in detections:
            if detection['class_name'].lower() in target_classes:
                return True
        return False
    
    def _get_ppe_confidence(self, detections: List[Dict], ppe_type: str) -> float:
        """PPE 항목 신뢰도 반환"""
        target_classes = self.ppe_classes.get(ppe_type, [])
        max_confidence = 0.0
        for detection in detections:
            if detection['class_name'].lower() in target_classes:
                max_confidence = max(max_confidence, detection['confidence'])
        return max_confidence
    
    def _check_vest_by_color(self, person_roi: np.ndarray) -> bool:
        """색상 기반 안전조끼 감지"""
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in self.vest_color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # 색상 영역이 충분히 크면 조끼로 판단
            color_ratio = np.sum(mask > 0) / mask.size
            if color_ratio > 0.05:  # 5% 이상
                return True
        
        return False

# ================ 고급 SLAM 기반 매핑 ================

class SLAMMapper:
    """SLAM 기반 환경 매핑"""
    
    def __init__(self):
        self.global_map = {}
        self.current_pose = np.eye(4)  # 4x4 변환 행렬
        self.landmark_database = {}
        self.pose_history = []
        
        # ORB 특징점 검출기
        self.orb = cv2.ORB_create(3000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 맵 업데이트 파라미터
        self.min_match_count = 15
        self.max_reprojection_error = 5.0
        
    def process_frame(self, sensor_data: SensorData) -> Dict[str, Any]:
        """새 프레임 처리 및 위치 추정"""
        frame = sensor_data.rgb_frame
        timestamp = sensor_data.timestamp
        
        # 특징점 추출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 50:
            return {'success': False, 'reason': 'insufficient_features'}
        
        # 이전 프레임과 매칭
        if hasattr(self, 'prev_descriptors') and self.prev_descriptors is not None:
            matches = self.bf_matcher.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) >= self.min_match_count:
                # 포즈 추정
                pose_result = self._estimate_pose(matches, self.prev_keypoints, keypoints)
                
                if pose_result['success']:
                    # 전역 좌표계로 변환
                    relative_transform = pose_result['transform']
                    self.current_pose = np.dot(self.current_pose, relative_transform)
                    
                    # 랜드마크 업데이트
                    self._update_landmarks(keypoints, descriptors, timestamp)
                    
                    # 포즈 히스토리 저장
                    self.pose_history.append({
                        'timestamp': timestamp,
                        'pose': self.current_pose.copy(),
                        'confidence': pose_result['confidence']
                    })
        
        # 현재 프레임 정보 저장
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = frame.copy()
        
        return {
            'success': True,
            'current_pose': self.current_pose,
            'landmark_count': len(self.landmark_database),
            'keypoint_count': len(keypoints)
        }
    
    def _estimate_pose(self, matches: List, prev_kp: List, curr_kp: List) -> Dict[str, Any]:
        """두 프레임 간 상대 포즈 추정"""
        
        if len(matches) < 8:
            return {'success': False}
        
        # 매칭된 점들 추출
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        
        # Essential Matrix 계산 (카메라 내부 파라미터 필요)
        # 여기서는 간단한 호모그래피 사용
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return {'success': False}
        
        # 인라이어 비율 계산
        inlier_ratio = np.sum(mask) / len(mask)
        confidence = min(1.0, inlier_ratio * 2.0)
        
        # 변환 행렬 생성 (간소화된 버전)
        transform = np.eye(4)
        # 실제로는 H에서 회전과 이동을 분해해야 함
        
        return {
            'success': True,
            'transform': transform,
            'confidence': confidence,
            'inlier_count': np.sum(mask)
        }
    
    def _update_landmarks(self, keypoints: List, descriptors: np.ndarray, timestamp: float):
        """랜드마크 데이터베이스 업데이트"""
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            # 랜드마크 ID 생성 (실제로는 더 정교한 방법 필요)
            landmark_id = f"{timestamp}_{i}"
            
            # 월드 좌표로 변환 (깊이 정보 필요)
            world_point = np.array([kp.pt[0], kp.pt[1], 0, 1])  # Z=0 가정
            world_point = np.dot(self.current_pose, world_point)
            
            self.landmark_database[landmark_id] = {
                'world_position': world_point[:3],
                'descriptor': desc,
                'observations': 1,
                'last_seen': timestamp
            }
    
    def get_hazard_zone_in_global_frame(self, local_polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """로컬 좌표의 위험구역을 전역 좌표로 변환"""
        global_polygon = []
        
        for point in local_polygon:
            local_point = np.array([point[0], point[1], 0, 1])
            global_point = np.dot(self.current_pose, local_point)
            global_polygon.append((global_point[0], global_point[1]))
        
        return global_polygon

# ================ 통합 위험구역 감지 시스템 ================

class EnhancedHazardDetectionSystem:
    """통합 위험구역 감지 시스템"""
    
    def __init__(self):
        # 서브시스템 초기화
        self.yolo_detector = YOLODetector()
        self.ppe_detector = PPEDetector()
        self.slam_mapper = SLAMMapper()
        
        # 위험구역 데이터베이스
        self.hazard_zones = []
        self.detection_history = deque(maxlen=1000)
        
        # 실시간 모니터링
        self.current_alerts = []
        self.safety_violations = []
        
        # 성능 메트릭
        self.performance_stats = {
            'fps': 0.0,
            'detection_time': 0.0,
            'processing_time': 0.0,
            'frame_count': 0
        }
        
        # 쓰레딩 제어
        self.processing_lock = threading.Lock()
        self.stop_processing = False
    
    def add_hazard_zone(self, frame: np.ndarray, polygon: List[Tuple[float, float]], 
                       zone_info: Dict[str, Any]) -> int:
        """위험구역 추가"""
        
        # 전역 좌표로 변환
        global_polygon = self.slam_mapper.get_hazard_zone_in_global_frame(polygon)
        
        hazard_zone = {
            'id': len(self.hazard_zones),
            'name': zone_info.get('name', f'위험구역_{len(self.hazard_zones)}'),
            'type': zone_info.get('type', 'general'),
            'local_polygon': polygon,
            'global_polygon': global_polygon,
            'reference_frame': frame.copy(),
            'alert_level': AlertLevel(zone_info.get('alert_level', 2)),
            'ppe_required': zone_info.get('ppe_required', []),
            'creation_pose': self.slam_mapper.current_pose.copy(),
            'active': True,
            'metadata': zone_info
        }
        
        # 특징점 추출 및 저장
        hazard_zone.update(self._extract_zone_features(frame, polygon))
        
        self.hazard_zones.append(hazard_zone)
        return hazard_zone['id']
    
    def process_sensor_data(self, sensor_data: SensorData) -> List[DetectionResult]:
        """센서 데이터 처리 및 위험구역 감지"""
        start_time = time.time()
        
        with self.processing_lock:
            results = []
            
            # 1. SLAM 위치 추정
            slam_result = self.slam_mapper.process_frame(sensor_data)
            
            # 2. YOLO 객체 감지
            object_detections = self.yolo_detector.detect_objects(
                sensor_data.rgb_frame, conf_threshold=0.5
            )
            
            # 3. 사람 감지 및 PPE 검사
            person_detections = self.yolo_detector.detect_persons(sensor_data.rgb_frame)
            ppe_results = []
            
            for person in person_detections:
                ppe_result = self.ppe_detector.detect_ppe(
                    sensor_data.rgb_frame, person['bbox']
                )
                ppe_results.append(ppe_result)
            
            # 4. 위험구역 감지
            hazard_detections = self._detect_hazard_zones(
                sensor_data, object_detections, ppe_results
            )
            
            # 5. 안전 규칙 검증
            safety_violations = self._check_safety_violations(
                hazard_detections, ppe_results, person_detections
            )
            
            # 6. 결과 통합
            all_results = hazard_detections + safety_violations
            
            # 7. 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            return all_results
    
    def _detect_hazard_zones(self, sensor_data: SensorData, 
                            object_detections: List[Dict],
                            ppe_results: List[PPEDetection]) -> List[DetectionResult]:
        """위험구역 감지 수행"""
        
        detections = []
        frame = sensor_data.rgb_frame
        current_pose = self.slam_mapper.current_pose
        
        for zone in self.hazard_zones:
            if not zone['active']:
                continue
            
            # 포즈 기반 위험구역 투영
            projected_polygon = self._project_zone_to_current_view(zone, current_pose)
            
            if projected_polygon is None:
                continue
            
            # 위험구역 내 객체 확인
            objects_in_zone = self._find_objects_in_zone(object_detections, projected_polygon)
            people_in_zone = self._find_people_in_zone(ppe_results, projected_polygon)
            
            if objects_in_zone or people_in_zone:
                # 위험도 계산
                risk_score = self._calculate_risk_score(zone, objects_in_zone, people_in_zone)
                
                # 경고 레벨 결정
                alert_level = self._determine_alert_level(zone, risk_score, people_in_zone)
                
                detection = DetectionResult(
                    zone_id=zone['id'],
                    zone_name=zone['name'],
                    confidence=risk_score,
                    bbox=self._polygon_to_bbox(projected_polygon),
                    polygon=projected_polygon,
                    timestamp=sensor_data.timestamp,
                    alert_level=alert_level,
                    detection_method='slam_projection',
                    additional_info={
                        'objects_in_zone': len(objects_in_zone),
                        'people_in_zone': len(people_in_zone),
                        'ppe_compliance': self._check_ppe_compliance(people_in_zone, zone)
                    }
                )
                
                detections.append(detection)
        
        return detections
    
    def _check_safety_violations(self, hazard_detections: List[DetectionResult],
                               ppe_results: List[PPEDetection],
                               person_detections: List[Dict]) -> List[DetectionResult]:
        """안전 규칙 위반 검사"""
        
        violations = []
        
        # PPE 미착용 검사
        for ppe in ppe_results:
            violation_items = []
            
            if not ppe.helmet_detected:
                violation_items.append('helmet')
            if not ppe.vest_detected:
                violation_items.append('vest')
            if not ppe.gloves_detected:
                violation_items.append('gloves')
            
            if violation_items:
                violation = DetectionResult(
                    zone_id=-1,  # 특별 ID
                    zone_name="PPE 위반",
                    confidence=1.0 - min(ppe.confidence_scores.values()),
                    bbox=ppe.person_bbox,
                    polygon=[],
                    timestamp=time.time(),
                    alert_level=AlertLevel.HIGH,
                    detection_method='ppe_violation',
                    additional_info={
                        'missing_items': violation_items,
                        'confidence_scores': ppe.confidence_scores
                    }
                )
                violations.append(violation)
        
        return violations
    
    def _extract_zone_features(self, frame: np.ndarray, polygon: List[Tuple[float, float]]) -> Dict:
        """위험구역 특징 추출"""
        
        # 마스크 생성
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # ORB 특징점 추출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.slam_mapper.orb.detectAndCompute(gray, mask)
        
        # 색상 히스토그램
        hist = cv2.calcHist([frame], [0, 1, 2], mask, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'color_histogram': hist,
            'mask': mask
        }
    
    def _project_zone_to_current_view(self, zone: Dict, current_pose: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """위험구역을 현재 뷰로 투영"""
        
        try:
            # 포즈 차이 계산
            zone_pose = zone['creation_pose']
            relative_transform = np.linalg.inv(zone_pose) @ current_pose
            
            # 폴리곤 점들을 현재 뷰로 변환
            projected_points = []
            for point in zone['local_polygon']:
                # 로컬 좌표를 호모지니어스 좌표로 변환
                local_point = np.array([point[0], point[1], 0, 1])
                
                # 현재 뷰로 변환
                transformed_point = np.linalg.inv(relative_transform) @ local_point
                
                # 화면 좌표로 투영 (간단한 투영 모델)
                x = transformed_point[0]
                y = transformed_point[1]
                
                projected_points.append((x, y))
            
            return projected_points
            
        except Exception as e:
            print(f"투영 오류: {e}")
            return None
    
    def _find_objects_in_zone(self, object_detections: List[Dict], 
                             zone_polygon: List[Tuple[float, float]]) -> List[Dict]:
        """위험구역 내 객체 찾기"""
        objects_in_zone = []
        
        for obj in object_detections:
            center_x, center_y = obj['center']
            
            # 점이 폴리곤 내부에 있는지 확인
            if cv2.pointPolygonTest(np.array(zone_polygon, dtype=np.float32), 
                                   (center_x, center_y), False) >= 0:
                objects_in_zone.append(obj)
        
        return objects_in_zone
    
    def _find_people_in_zone(self, ppe_results: List[PPEDetection],
                            zone_polygon: List[Tuple[float, float]]) -> List[PPEDetection]:
        """위험구역 내 사람 찾기"""
        people_in_zone = []
        
        for ppe in ppe_results:
            x, y, w, h = ppe.person_bbox
            center_x, center_y = x + w/2, y + h/2
            
            if cv2.pointPolygonTest(np.array(zone_polygon, dtype=np.float32),
                                   (center_x, center_y), False) >= 0:
                people_in_zone.append(ppe)
        
        return people_in_zone
    
    def _calculate_risk_score(self, zone: Dict, objects_in_zone: List[Dict],
                             people_in_zone: List[PPEDetection]) -> float:
        """위험도 점수 계산"""
        
        base_risk = 0.5  # 기본 위험도
        
        # 사람 수에 따른 위험도 증가
        people_factor = min(len(people_in_zone) * 0.2, 0.5)
        
        # PPE 미착용에 따른 위험도 증가
        ppe_violation_factor = 0.0
        for ppe in people_in_zone:
            violations = sum([
                not ppe.helmet_detected,
                not ppe.vest_detected,
                not ppe.gloves_detected
            ])
            ppe_violation_factor += violations * 0.1
        
        # 위험 객체에 따른 위험도
        hazardous_objects = ['truck', 'forklift', 'crane', 'excavator']
        object_factor = 0.0
        for obj in objects_in_zone:
            if obj['class_name'] in hazardous_objects:
                object_factor += 0.3
        
        total_risk = min(base_risk + people_factor + ppe_violation_factor + object_factor, 1.0)
        return total_risk
    
    def _determine_alert_level(self, zone: Dict, risk_score: float,
                              people_in_zone: List[PPEDetection]) -> AlertLevel:
        """경고 레벨 결정"""
        
        if risk_score < 0.3:
            return AlertLevel.LOW
        elif risk_score < 0.6:
            return AlertLevel.MEDIUM
        elif risk_score < 0.8:
            return AlertLevel.HIGH
        else:
            return AlertLevel.CRITICAL
    
    def _check_ppe_compliance(self, people_in_zone: List[PPEDetection], zone: Dict) -> Dict[str, Any]:
        """PPE 준수 상황 확인"""
        if not people_in_zone:
            return {'compliant': True, 'violations': []}
        
        required_ppe = zone.get('ppe_required', [])
        violations = []
        
        for ppe in people_in_zone:
            person_violations = []
            
            if 'helmet' in required_ppe and not ppe.helmet_detected:
                person_violations.append('helmet')
            if 'vest' in required_ppe and not ppe.vest_detected:
                person_violations.append('vest')
            if 'gloves' in required_ppe and not ppe.gloves_detected:
                person_violations.append('gloves')
            
            if person_violations:
                violations.append({
                    'person_bbox': ppe.person_bbox,
                    'missing_items': person_violations
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'compliance_rate': 1.0 - (len(violations) / len(people_in_zone))
        }
    
    def _polygon_to_bbox(self, polygon: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        """폴리곤을 바운딩 박스로 변환"""
        if not polygon:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _update_performance_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        self.performance_stats['frame_count'] += 1
        self.performance_stats['processing_time'] = processing_time
        
        # FPS 계산 (이동 평균)
        if hasattr(self, '_fps_history'):
            self._fps_history.append(1.0 / processing_time if processing_time > 0 else 0)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self.performance_stats['fps'] = np.mean(self._fps_history)
        else:
            self._fps_history = [1.0 / processing_time if processing_time > 0 else 0]
            self.performance_stats['fps'] = self._fps_history[0]

# ================ UI 컴포넌트 ================

def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """OpenCV 이미지를 QPixmap으로 변환"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(cv_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class PerformanceMonitor(QWidget):
    """성능 모니터링 위젯"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 성능 지표 표시
        self.fps_label = QLabel("FPS: 0.0")
        self.processing_time_label = QLabel("처리시간: 0.0ms")
        self.detection_count_label = QLabel("감지 수: 0")
        
        layout.addWidget(QLabel("성능 모니터링"))
        layout.addWidget(self.fps_label)
        layout.addWidget(self.processing_time_label)
        layout.addWidget(self.detection_count_label)
        
        self.setLayout(layout)
    
    def update_stats(self, stats: Dict[str, Any]):
        """성능 통계 업데이트"""
        self.fps_label.setText(f"FPS: {stats.get('fps', 0):.1f}")
        self.processing_time_label.setText(f"처리시간: {stats.get('processing_time', 0)*1000:.1f}ms")

class SafetyDashboard(QWidget):
    """안전 대시보드 위젯"""
    
    def __init__(self):
        super().__init__()
        self.current_alerts = []
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 현재 경고 상태
        self.alert_status = QLabel("시스템 정상")
        self.alert_status.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold;")
        layout.addWidget(self.alert_status)
        
        # 경고 목록
        self.alert_list = QListWidget()
        self.alert_list.setMaximumHeight(200)
        layout.addWidget(QLabel("활성 경고:"))
        layout.addWidget(self.alert_list)
        
        # 통계 정보
        stats_group = QGroupBox("안전 통계")
        stats_layout = QGridLayout()
        
        self.total_detections_label = QLabel("총 감지: 0")
        self.violations_label = QLabel("위반 사항: 0")
        self.compliance_rate_label = QLabel("준수율: 100%")
        
        stats_layout.addWidget(self.total_detections_label, 0, 0)
        stats_layout.addWidget(self.violations_label, 0, 1)
        stats_layout.addWidget(self.compliance_rate_label, 1, 0)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self.setLayout(layout)
    
    def update_alerts(self, detections: List[DetectionResult]):
        """경고 상태 업데이트"""
        
        # 현재 경고 상태 결정
        max_alert_level = AlertLevel.SAFE
        active_alerts = []
        
        for detection in detections:
            if detection.alert_level.value > max_alert_level.value:
                max_alert_level = detection.alert_level
            
            alert_text = f"[{detection.alert_level.name}] {detection.zone_name} (신뢰도: {detection.confidence:.2f})"
            active_alerts.append(alert_text)
        
        # 상태 표시 업데이트
        status_colors = {
            AlertLevel.SAFE: ("시스템 정상", "green"),
            AlertLevel.LOW: ("낮은 위험", "yellow"),
            AlertLevel.MEDIUM: ("중간 위험", "orange"),
            AlertLevel.HIGH: ("높은 위험", "red"),
            AlertLevel.CRITICAL: ("치명적 위험", "darkred")
        }
        
        status_text, color = status_colors[max_alert_level]
        self.alert_status.setText(status_text)
        self.alert_status.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; font-weight: bold;")
        
        # 경고 목록 업데이트
        self.alert_list.clear()
        for alert in active_alerts:
            self.alert_list.addItem(alert)

# ================ 메인 애플리케이션 ================

class EnhancedRobotSafetyApp(QWidget):
    """향상된 로봇 안전 감시 시스템"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Robot Safety Surveillance System")
        self.resize(1800, 1200)
        
        # 핵심 시스템 초기화
        self.detection_system = EnhancedHazardDetectionSystem()
        
        # UI 상태
        self.frames = []
        self.current_frame_index = 0
        self.current_sensor_data = None
        self.region_drawing_mode = False
        self.current_region_points = []
        
        # 실시간 처리
        self.real_time_mode = False
        self.processing_thread = None
        
        self.setup_ui()
        self.setup_timers()
    
    def setup_ui(self):
        """UI 설정"""
        main_layout = QHBoxLayout()
        
        # 좌측 제어 패널
        left_panel = self.create_control_panel()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(450)
        
        # 중앙 비디오 표시
        center_panel = self.create_video_panel()
        
        # 우측 대시보드
        right_panel = self.create_dashboard_panel()
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(400)
        
        # 스플리터로 구성
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def create_control_panel(self):
        """제어 패널 생성"""
        layout = QVBoxLayout()
        
        # 탭 위젯
        tab_widget = QTabWidget()
        
        # 1. 시스템 제어 탭
        system_tab = QWidget()
        system_layout = QVBoxLayout()
        
        # 파일 관리
        file_group = QGroupBox("파일 관리")
        file_layout = QVBoxLayout()
        
        self.btn_load_video = QPushButton("테스트 영상 로딩")
        self.btn_load_video.clicked.connect(self.load_video)
        file_layout.addWidget(self.btn_load_video)
        
        self.btn_load_model = QPushButton("YOLO 모델 로딩")
        self.btn_load_model.clicked.connect(self.load_yolo_model)
        file_layout.addWidget(self.btn_load_model)
        
        self.btn_save_config = QPushButton("설정 저장")
        self.btn_save_config.clicked.connect(self.save_configuration)
        file_layout.addWidget(self.btn_save_config)
        
        file_group.setLayout(file_layout)
        system_layout.addWidget(file_group)
        
        # 실시간 제어
        realtime_group = QGroupBox("실시간 제어")
        realtime_layout = QVBoxLayout()
        
        self.btn_start_realtime = QPushButton("실시간 감시 시작")
        self.btn_start_realtime.clicked.connect(self.toggle_realtime_mode)
        realtime_layout.addWidget(self.btn_start_realtime)
        
        self.realtime_status = QLabel("대기 중")
        realtime_layout.addWidget(self.realtime_status)
        
        realtime_group.setLayout(realtime_layout)
        system_layout.addWidget(realtime_group)
        
        system_tab.setLayout(system_layout)
        tab_widget.addTab(system_tab, "시스템")
        
        # 2. 위험구역 관리 탭
        zone_tab = QWidget()
        zone_layout = QVBoxLayout()
        
        # 구역 추가
        add_zone_group = QGroupBox("위험구역 추가")
        add_zone_layout = QVBoxLayout()
        
        self.btn_add_zone = QPushButton("구역 그리기 시작")
        self.btn_add_zone.clicked.connect(self.toggle_zone_drawing)
        add_zone_layout.addWidget(self.btn_add_zone)
        
        # 구역 유형 선택
        zone_type_layout = QHBoxLayout()
        zone_type_layout.addWidget(QLabel("구역 유형:"))
        self.zone_type_combo = QComboBox()
        self.zone_type_combo.addItems(["일반 위험", "화재 위험", "독성 가스", "추락 위험", "기계 위험"])
        zone_type_layout.addWidget(self.zone_type_combo)
        add_zone_layout.addLayout(zone_type_layout)
        
        # 경고 레벨 선택
        alert_level_layout = QHBoxLayout()
        alert_level_layout.addWidget(QLabel("경고 레벨:"))
        self.alert_level_combo = QComboBox()
        self.alert_level_combo.addItems(["낮음", "중간", "높음", "치명적"])
        alert_level_layout.addWidget(self.alert_level_combo)
        add_zone_layout.addLayout(alert_level_layout)
        
        # PPE 요구사항
        ppe_group = QGroupBox("필수 PPE")
        ppe_layout = QVBoxLayout()
        
        self.helmet_check = QCheckBox("안전모")
        self.vest_check = QCheckBox("안전조끼")
        self.gloves_check = QCheckBox("안전장갑")
        self.boots_check = QCheckBox("안전화")
        self.mask_check = QCheckBox("마스크")
        
        ppe_layout.addWidget(self.helmet_check)
        ppe_layout.addWidget(self.vest_check)
        ppe_layout.addWidget(self.gloves_check)
        ppe_layout.addWidget(self.boots_check)
        ppe_layout.addWidget(self.mask_check)
        
        ppe_group.setLayout(ppe_layout)
        add_zone_layout.addWidget(ppe_group)
        
        add_zone_group.setLayout(add_zone_layout)
        zone_layout.addWidget(add_zone_group)
        
        # 기존 구역 목록
        zone_list_group = QGroupBox("등록된 위험구역")
        zone_list_layout = QVBoxLayout()
        
        self.zone_list = QListWidget()
        self.zone_list.setMaximumHeight(200)
        zone_list_layout.addWidget(self.zone_list)
        
        zone_buttons_layout = QHBoxLayout()
        self.btn_edit_zone = QPushButton("편집")
        self.btn_delete_zone = QPushButton("삭제")
        zone_buttons_layout.addWidget(self.btn_edit_zone)
        zone_buttons_layout.addWidget(self.btn_delete_zone)
        zone_list_layout.addLayout(zone_buttons_layout)
        
        zone_list_group.setLayout(zone_list_layout)
        zone_layout.addWidget(zone_list_group)
        
        zone_tab.setLayout(zone_layout)
        tab_widget.addTab(zone_tab, "위험구역")
        
        # 3. 감지 설정 탭
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()
        
        # YOLO 설정
        yolo_group = QGroupBox("YOLO 설정")
        yolo_layout = QGridLayout()
        
        yolo_layout.addWidget(QLabel("신뢰도 임계값:"), 0, 0)
        self.yolo_confidence_slider = QSlider(Qt.Horizontal)
        self.yolo_confidence_slider.setRange(10, 95)
        self.yolo_confidence_slider.setValue(50)
        yolo_layout.addWidget(self.yolo_confidence_slider, 0, 1)
        self.yolo_conf_label = QLabel("0.50")
        yolo_layout.addWidget(self.yolo_conf_label, 0, 2)
        
        self.yolo_confidence_slider.valueChanged.connect(
            lambda v: self.yolo_conf_label.setText(f"{v/100:.2f}")
        )
        
        yolo_group.setLayout(yolo_layout)
        detection_layout.addWidget(yolo_group)
        
        # PPE 설정
        ppe_detection_group = QGroupBox("PPE 감지 설정")
        ppe_detection_layout = QVBoxLayout()
        
        self.enable_ppe_detection = QCheckBox("PPE 감지 활성화")
        self.enable_ppe_detection.setChecked(True)
        ppe_detection_layout.addWidget(self.enable_ppe_detection)
        
        self.strict_ppe_mode = QCheckBox("엄격한 PPE 모드")
        ppe_detection_layout.addWidget(self.strict_ppe_mode)
        
        ppe_detection_group.setLayout(ppe_detection_layout)
        detection_layout.addWidget(ppe_detection_group)
        
        detection_tab.setLayout(detection_layout)
        tab_widget.addTab(detection_tab, "감지 설정")
        
        layout.addWidget(tab_widget)
        
        # 재생 제어 (하단)
        playback_group = QGroupBox("재생 제어")
        playback_layout = QVBoxLayout()
        
        playback_buttons = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_pause = QPushButton("⏸ 일시정지")
        self.btn_stop = QPushButton("⏹ 정지")
        
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        
        playback_buttons.addWidget(self.btn_play)
        playback_buttons.addWidget(self.btn_pause)
        playback_buttons.addWidget(self.btn_stop)
        playback_layout.addLayout(playback_buttons)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        playback_layout.addWidget(self.frame_slider)
        
        self.frame_info = QLabel("프레임: 0/0")
        playback_layout.addWidget(self.frame_info)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        return layout
    
    def create_video_panel(self):
        """비디오 패널 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 메인 비디오 표시
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("비디오를 로딩하세요")
        self.video_label.mousePressEvent = self.on_video_click
        
        # 스크롤 영역
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.video_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # 하단 정보 표시
        info_layout = QHBoxLayout()
        
        self.detection_info = QLabel("감지 정보: 대기 중")
        self.detection_info.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        info_layout.addWidget(self.detection_info)
        
        self.processing_info = QLabel("처리 시간: 0ms")
        self.processing_info.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        info_layout.addWidget(self.processing_info)
        
        layout.addLayout(info_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_dashboard_panel(self):
        """대시보드 패널 생성"""
        layout = QVBoxLayout()
        
        # 탭 위젯
        dashboard_tabs = QTabWidget()
        
        # 1. 안전 대시보드
        self.safety_dashboard = SafetyDashboard()
        dashboard_tabs.addTab(self.safety_dashboard, "안전 상태")
        
        # 2. 성능 모니터
        self.performance_monitor = PerformanceMonitor()
        dashboard_tabs.addTab(self.performance_monitor, "성능")
        
        # 3. 로그 뷰어
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(300)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(QLabel("시스템 로그:"))
        log_layout.addWidget(self.log_text)
        
        log_buttons = QHBoxLayout()
        self.btn_clear_log = QPushButton("로그 지우기")
        self.btn_export_log = QPushButton("로그 내보내기")
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        self.btn_export_log.clicked.connect(self.export_log)
        log_buttons.addWidget(self.btn_clear_log)
        log_buttons.addWidget(self.btn_export_log)
        log_layout.addLayout(log_buttons)
        
        log_tab.setLayout(log_layout)
        dashboard_tabs.addTab(log_tab, "로그")
        
        layout.addWidget(dashboard_tabs)
        
        return layout
    
    def setup_timers(self):
        """타이머 설정"""
        # 비디오 재생 타이머
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.next_frame)
        self.video_playing = False
        
        # 성능 업데이트 타이머
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_display)
        self.performance_timer.start(1000)  # 1초마다 업데이트
    
    def load_video(self):
        """비디오 로딩"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "비디오 선택", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if not fname:
            return
        
        try:
            cap = cv2.VideoCapture(fname)
            self.frames.clear()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 프로그레스 다이얼로그
            progress = QProgressBar(self)
            progress.setMaximum(total_frames)
            progress.show()
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frames.append(frame.copy())
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()
            
            cap.release()
            progress.close()
            
            if self.frames:
                self.frame_slider.setMaximum(len(self.frames) - 1)
                self.current_frame_index = 0
                self.show_current_frame()
                
                self.log(f"비디오 로딩 완료: {os.path.basename(fname)}")
                self.log(f"총 {len(self.frames)}프레임, {fps}FPS")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"비디오 로딩 실패: {str(e)}")
            self.log(f"비디오 로딩 오류: {str(e)}")
    
    def load_yolo_model(self):
        """YOLO 모델 로딩"""
        if not YOLO_AVAILABLE:
            QMessageBox.warning(self, "경고", "YOLO 라이브러리가 설치되지 않았습니다.")
            return
        
        fname, _ = QFileDialog.getOpenFileName(
            self, "YOLO 모델 선택", "", "Model Files (*.pt *.onnx)"
        )
        
        if fname:
            try:
                self.detection_system.yolo_detector = YOLODetector(fname)
                self.log(f"YOLO 모델 로딩 완료: {os.path.basename(fname)}")
                QMessageBox.information(self, "성공", "YOLO 모델이 성공적으로 로딩되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"모델 로딩 실패: {str(e)}")
                self.log(f"모델 로딩 오류: {str(e)}")
    
    def show_current_frame(self):
        """현재 프레임 표시"""
        if not self.frames:
            return
        
        frame = self.frames[self.current_frame_index].copy()
        
        # 센서 데이터 생성
        self.current_sensor_data = SensorData(
            rgb_frame=frame,
            timestamp=time.time()
        )
        
        # 위험구역 감지 수행
        if not self.real_time_mode:  # 실시간 모드가 아닐 때만
            detections = self.detection_system.process_sensor_data(self.current_sensor_data)
            self.display_detections(frame, detections)
            self.safety_dashboard.update_alerts(detections)
        
        # 등록된 위험구역 표시
        self.draw_registered_zones(frame)
        
        # 현재 그리고 있는 구역 표시
        self.draw_current_drawing(frame)
        
        # 화면에 표시
        pixmap = cv_to_qpixmap(frame)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # 프레임 정보 업데이트
        self.frame_info.setText(f"프레임: {self.current_frame_index + 1}/{len(self.frames)}")
        self.frame_slider.setValue(self.current_frame_index)
    
    def display_detections(self, frame: np.ndarray, detections: List[DetectionResult]):
        """감지 결과 표시"""
        detection_count = 0
        
        for detection in detections:
            detection_count += 1
            
            # 경고 레벨에 따른 색상 설정
            colors = {
                AlertLevel.SAFE: (0, 255, 0),
                AlertLevel.LOW: (0, 255, 255),
                AlertLevel.MEDIUM: (0, 165, 255),
                AlertLevel.HIGH: (0, 0, 255),
                AlertLevel.CRITICAL: (0, 0, 139)
            }
            
            color = colors.get(detection.alert_level, (255, 255, 255))
            
            # 바운딩 박스 그리기
            if detection.bbox != (0, 0, 0, 0):
                x, y, w, h = detection.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # 라벨 텍스트
                label = f"{detection.zone_name} ({detection.confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 라벨 배경
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), color, -1)
                
                # 라벨 텍스트
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 폴리곤 그리기 (있는 경우)
            if detection.polygon:
                pts = np.array(detection.polygon, dtype=np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
        
        # 감지 정보 업데이트
        self.detection_info.setText(f"감지된 위험: {detection_count}개")
    
    def draw_registered_zones(self, frame: np.ndarray):
        """등록된 위험구역 그리기"""
        for i, zone in enumerate(self.detection_system.hazard_zones):
            if zone['active']:
                # 투영된 폴리곤 가져오기
                projected_polygon = self.detection_system.slam_mapper.get_hazard_zone_in_global_frame(
                    zone['local_polygon']
                )
                
                if projected_polygon:
                    pts = np.array(projected_polygon, dtype=np.int32)
                    
                    # 반투명 채우기
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))
                    cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
                    
                    # 경계선
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    
                    # 구역 이름
                    centroid = np.mean(pts, axis=0).astype(int)
                    cv2.putText(frame, zone['name'], tuple(centroid), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_current_drawing(self, frame: np.ndarray):
        """현재 그리고 있는 구역 표시"""
        if self.current_region_points:
            # 점들 연결
            for i, point in enumerate(self.current_region_points):
                cv2.circle(frame, tuple(map(int, point)), 5, (255, 255, 0), -1)
                if i > 0:
                    cv2.line(frame, tuple(map(int, self.current_region_points[i-1])), 
                            tuple(map(int, point)), (255, 255, 0), 2)
            
            # 첫 점과 마지막 점 연결 (3개 이상일 때)
            if len(self.current_region_points) >= 3:
                cv2.line(frame, tuple(map(int, self.current_region_points[-1])), 
                        tuple(map(int, self.current_region_points[0])), (255, 255, 0), 2)
    
    def toggle_zone_drawing(self):
        """위험구역 그리기 모드 토글"""
        self.region_drawing_mode = not self.region_drawing_mode
        
        if self.region_drawing_mode:
            self.btn_add_zone.setText("구역 추가 완료")
            self.current_region_points = []
            self.log("위험구역 그리기 모드 시작")
        else:
            self.btn_add_zone.setText("구역 그리기 시작")
            
            if len(self.current_region_points) >= 3:
                self.complete_zone_drawing()
            
            self.current_region_points = []
            self.log("위험구역 그리기 모드 종료")
    
    def complete_zone_drawing(self):
        """위험구역 그리기 완료"""
        if not self.current_sensor_data:
            QMessageBox.warning(self, "경고", "현재 프레임이 없습니다.")
            return
        
        # 구역 정보 수집
        zone_info = {
            'name': f"위험구역_{len(self.detection_system.hazard_zones) + 1}",
            'type': self.zone_type_combo.currentText(),
            'alert_level': self.alert_level_combo.currentIndex() + 1,
            'ppe_required': []
        }
        
        # PPE 요구사항 수집
        if self.helmet_check.isChecked():
            zone_info['ppe_required'].append('helmet')
        if self.vest_check.isChecked():
            zone_info['ppe_required'].append('vest')
        if self.gloves_check.isChecked():
            zone_info['ppe_required'].append('gloves')
        if self.boots_check.isChecked():
            zone_info['ppe_required'].append('boots')
        if self.mask_check.isChecked():
            zone_info['ppe_required'].append('mask')
        
        # 위험구역 추가
        zone_id = self.detection_system.add_hazard_zone(
            self.current_sensor_data.rgb_frame,
            self.current_region_points,
            zone_info
        )
        
        # UI 업데이트
        self.update_zone_list()
        self.log(f"위험구역 추가 완료: {zone_info['name']} (ID: {zone_id})")
        
        QMessageBox.information(self, "완료", f"위험구역 '{zone_info['name']}'이 추가되었습니다.")
    
    def on_video_click(self, event):
        """비디오 클릭 이벤트"""
        if self.region_drawing_mode and self.current_sensor_data is not None:
            # 클릭 위치를 이미지 좌표로 변환
            x, y = event.pos().x(), event.pos().y()
            
            # 스케일링 고려
            pixmap = self.video_label.pixmap()
            if pixmap:
                label_size = self.video_label.size()
                pixmap_size = pixmap.size()
                
                scale_x = self.current_sensor_data.rgb_frame.shape[1] / pixmap_size.width()
                scale_y = self.current_sensor_data.rgb_frame.shape[0] / pixmap_size.height()
                
                offset_x = (label_size.width() - pixmap_size.width()) / 2
                offset_y = (label_size.height() - pixmap_size.height()) / 2
                
                real_x = (x - offset_x) * scale_x
                real_y = (y - offset_y) * scale_y
                
                if (0 <= real_x < self.current_sensor_data.rgb_frame.shape[1] and 
                    0 <= real_y < self.current_sensor_data.rgb_frame.shape[0]):
                    
                    self.current_region_points.append([real_x, real_y])
                    self.show_current_frame()
                    self.log(f"점 추가: ({real_x:.0f}, {real_y:.0f})")
    
    def update_zone_list(self):
        """위험구역 목록 업데이트"""
        self.zone_list.clear()
        
        for zone in self.detection_system.hazard_zones:
            item_text = f"[{zone['id']}] {zone['name']} ({zone['type']})"
            self.zone_list.addItem(item_text)
    
    def toggle_realtime_mode(self):
        """실시간 모드 토글"""
        self.real_time_mode = not self.real_time_mode
        
        if self.real_time_mode:
            self.btn_start_realtime.setText("실시간 감시 중지")
            self.realtime_status.setText("실시간 감시 중")
            self.start_realtime_processing()
        else:
            self.btn_start_realtime.setText("실시간 감시 시작")
            self.realtime_status.setText("대기 중")
            self.stop_realtime_processing()
    
    def start_realtime_processing(self):
        """실시간 처리 시작"""
        if self.frames:
            self.video_timer.start(100)  # 10 FPS
            self.log("실시간 처리 시작")
    
    def stop_realtime_processing(self):
        """실시간 처리 중지"""
        self.video_timer.stop()
        self.log("실시간 처리 중지")
    
    def play_video(self):
        """비디오 재생"""
        if not self.video_playing and self.frames:
            self.video_timer.start(33)  # ~30 FPS
            self.video_playing = True
            self.log("비디오 재생 시작")
    
    def pause_video(self):
        """비디오 일시정지"""
        self.video_timer.stop()
        self.video_playing = False
        self.log("비디오 일시정지")
    
    def stop_video(self):
        """비디오 정지"""
        self.video_timer.stop()
        self.video_playing = False
        self.current_frame_index = 0
        self.show_current_frame()
        self.log("비디오 정지")
    
    def next_frame(self):
        """다음 프레임으로 이동"""
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.show_current_frame()
        elif self.video_playing:
            # 루프 재생
            self.current_frame_index = 0
            self.show_current_frame()
    
    def on_frame_change(self, value):
        """프레임 슬라이더 변경"""
        self.current_frame_index = value
        self.show_current_frame()
    
    def update_performance_display(self):
        """성능 표시 업데이트"""
        stats = self.detection_system.performance_stats
        self.performance_monitor.update_stats(stats)
        
        processing_time_ms = stats.get('processing_time', 0) * 1000
        self.processing_info.setText(f"처리 시간: {processing_time_ms:.1f}ms")
    
    def save_configuration(self):
        """설정 저장"""
        config = {
            'zones': [
                {
                    'id': zone['id'],
                    'name': zone['name'],
                    'type': zone['type'],
                    'polygon': zone['local_polygon'],
                    'alert_level': zone['alert_level'].value,
                    'ppe_required': zone['ppe_required']
                }
                for zone in self.detection_system.hazard_zones
            ],
            'detection_settings': {
                'yolo_confidence': self.yolo_confidence_slider.value() / 100,
                'ppe_detection_enabled': self.enable_ppe_detection.isChecked(),
                'strict_ppe_mode': self.strict_ppe_mode.isChecked()
            }
        }
        
        fname, _ = QFileDialog.getSaveFileName(
            self, "설정 저장", "", "JSON Files (*.json)"
        )
        
        if fname:
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.log(f"설정 저장 완료: {fname}")
                QMessageBox.information(self, "성공", "설정이 저장되었습니다.")
                
            except Exception as e:
                self.log(f"설정 저장 오류: {str(e)}")
                QMessageBox.critical(self, "오류", f"설정 저장 실패: {str(e)}")
    
    def export_log(self):
        """로그 내보내기"""
        fname, _ = QFileDialog.getSaveFileName(
            self, "로그 내보내기", "", "Text Files (*.txt)"
        )
        
        if fname:
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                
                QMessageBox.information(self, "성공", "로그가 내보내졌습니다.")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"로그 내보내기 실패: {str(e)}")
    
    def log(self, message: str):
        """로그 메시지 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_text.append(log_entry)
        
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 콘솔에도 출력
        print(log_entry)
    
    def closeEvent(self, event):
        """애플리케이션 종료 이벤트"""
        self.stop_realtime_processing()
        event.accept()

# ================ 메인 실행 ================

if __name__ == "__main__":
    # 고해상도 디스플레이 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName("Enhanced Robot Safety System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Robot Safety Solutions")
    
    # 메인 윈도우 생성 및 표시
    main_window = EnhancedRobotSafetyApp()
    main_window.show()
    
    # 시작 로그
    main_window.log("=== Enhanced Robot Safety System 시작 ===")
    main_window.log(f"YOLO 지원: {'활성화' if YOLO_AVAILABLE else '비활성화'}")
    main_window.log(f"TensorFlow 지원: {'활성화' if TF_AVAILABLE else '비활성화'}")
    main_window.log("시스템 준비 완료")
    
    sys.exit(app.exec_())