import cv2
import time
import json
import torch
import paho.mqtt.client as mqtt

# Monkey patch torch.load to handle PyTorch 2.6 weights_only issue
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    """Patched torch.load that sets weights_only=False for YOLO compatibility"""
    try:
        # Try with weights_only=False first for YOLO models
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **pickle_load_args)
    except Exception as e:
        # If that fails, try the original call
        print(f"[WARNING] Patched load failed, trying original: {e}")
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)

# Apply the patch
torch.load = patched_torch_load

# Now import ultralytics after patching
from ultralytics import YOLO

# RTSP stream URL
rtsp_url = "rtsp://admin:leelab01!!@166.104.248.212:8554/profile2/media.smp"

# YOLO helmet detection 모델
try:
    print("[INFO] Loading YOLO model...")
    model = YOLO("/app/helmet.pt")
    print("[INFO] YOLO model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    print("[INFO] Trying with torch serialization safe globals...")
    
    # Add safe globals as backup
    import torch.serialization
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.models.yolo.detect.DetectionModel',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.SiLU',
        'collections.OrderedDict'
    ])
    
    try:
        model = YOLO("/app/helmet.pt")
        print("[INFO] YOLO model loaded with safe globals")
    except Exception as e2:
        print(f"[FATAL] Could not load YOLO model: {e2}")
        exit(1)

# CUDA 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# GPU 정보 출력
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# MQTT 설정
broker_address = "192.168.0.53"
broker_port = 1883
mqtt_user = "qawsed123"
mqtt_pass = "qawsed123"
topic_pub = "robot_cmd"

# 바운딩 박스 MQTT 전송
def send_helmet_boxes(client, results):
    try:
        payload = json.dumps(results)
        client.publish(topic_pub, payload)
        print(f"[MQTT Sent] {payload}")
    except Exception as e:
        print(f"[ERROR] MQTT send failed: {e}")

def main():
    # MQTT 클라이언트 설정
    try:
        client = mqtt.Client()
        client.username_pw_set(mqtt_user, mqtt_pass)
        client.connect(broker_address, broker_port, 60)
        client.loop_start()
        print("[INFO] MQTT client connected successfully")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MQTT broker: {e}")
        print("[WARNING] Continuing without MQTT...")
        client = None

    # RTSP 스트림 연결
    print(f"[INFO] Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    # 연결 재시도 로직
    retry_count = 0
    max_retries = 5
    while not cap.isOpened() and retry_count < max_retries:
        retry_count += 1
        print(f"[WARNING] RTSP connection failed, retrying ({retry_count}/{max_retries})...")
        time.sleep(2)
        cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream after retries")
        return

    print("[INFO] RTSP stream opened successfully")
    print("[INFO] Starting helmet detection...")

    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame, attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            frame_count += 1
            
            try:
                # YOLO 추론
                results = model(frame, device=device, verbose=False)

                boxes = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names[cls_id].lower()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if conf < 0.6:
                                continue

                            # 헬멧이면 초록, 아니면 unsafe로 빨강
                            label = cls_name
                            color = [0, 255, 0]

                            if cls_name != "helmet":
                                label = "unsafe"
                                color = [255, 0, 0]

                            boxes.append({
                                "class": label,
                                "confidence": round(conf, 2),
                                "bbox": [x1, y1, x2, y2],
                                "color": color
                            })

                if boxes:
                    detection_count += 1
                    if client:
                        send_helmet_boxes(client, boxes)
                    print(f"[INFO] Frame {frame_count}: Detected {len(boxes)} objects")
                
                # 통계 출력 (매 100프레임마다)
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[STATS] Processed {frame_count} frames, {detection_count} detections, {fps:.1f} FPS")

            except Exception as e:
                print(f"[ERROR] Error processing frame {frame_count}: {e}")
                continue

            time.sleep(1)  # 1초 대기

    except KeyboardInterrupt:
        print("[INFO] Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        # 정리
        cap.release()
        if client:
            client.loop_stop()
            client.disconnect()
        print("[INFO] Application terminated cleanly")

if __name__ == "__main__":
    main()