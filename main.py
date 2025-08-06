import cv2
import time
import json
import torch
import torch.serialization
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# Add safe globals for YOLO model loading (PyTorch 2.6 compatibility)
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.head.Detect',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.block.SPPF',
    'ultralytics.nn.modules.head.Segment',
    'ultralytics.nn.modules.head.Pose',
    'ultralytics.nn.modules.head.Classify',
    'ultralytics.models.yolo.detect.DetectionModel',
    'torch.nn.modules.conv.Conv2d',
    'torch.nn.modules.batchnorm.BatchNorm2d',
    'torch.nn.modules.activation.SiLU',
    'torch.nn.modules.pooling.MaxPool2d',
    'torch.nn.modules.upsampling.Upsample',
    'torch.nn.modules.container.Sequential',
    'torch.nn.modules.container.ModuleList',
    'collections.OrderedDict'
])

# RTSP stream URL
rtsp_url = "rtsp://admin:leelab01!!@166.104.248.212:8554/profile2/media.smp"

# YOLO helmet detection 모델
try:
    model = YOLO("/app/helmet.pt")
    print("[INFO] YOLO model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    # Try alternative loading method
    try:
        import torch
        # Force weights_only=False for legacy compatibility
        torch._C._set_weights_only_load_mode(False)
        model = YOLO("/app/helmet.pt")
        print("[INFO] YOLO model loaded with legacy mode")
    except Exception as e2:
        print(f"[ERROR] Failed to load model even with legacy mode: {e2}")
        exit(1)

# CUDA 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# MQTT 설정
broker_address = "192.168.0.53"
broker_port = 1883
mqtt_user = "qawsed123"
mqtt_pass = "qawsed123"
topic_pub = "robot_cmd"

# 바운딩 박스 MQTT 전송
def send_helmet_boxes(client, results):
    payload = json.dumps(results)
    client.publish(topic_pub, payload)
    print(f"[MQTT Sent] {payload}")

def main():
    client = mqtt.Client()
    client.username_pw_set(mqtt_user, mqtt_pass)
    
    try:
        client.connect(broker_address, broker_port, 60)
        client.loop_start()
        print("[INFO] MQTT client connected successfully")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MQTT broker: {e}")
        return

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream")
        return

    print("[INFO] RTSP stream opened successfully")
    print("[INFO] Starting helmet detection...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        frame_count += 1
        
        try:
            results = model(frame, device=device)

            boxes = []
            for r in results:
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
                send_helmet_boxes(client, boxes)
                print(f"[INFO] Frame {frame_count}: Detected {len(boxes)} objects")
            else:
                print(f"[INFO] Frame {frame_count}: No objects detected")

        except Exception as e:
            print(f"[ERROR] Error processing frame {frame_count}: {e}")
            continue

        time.sleep(1)

    cap.release()
    client.loop_stop()
    print("[INFO] Application terminated")

if __name__ == "__main__":
    main()