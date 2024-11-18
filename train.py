from ultralytics import YOLO
import os

# 현재 파일의 절대 경로를 구함
current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, 'data.yaml')

# 모델 생성
model = YOLO('yolov8n.pt')

# 학습 시작 - 데이터셋 경로를 직접 지정
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    verbose=True,
    project=current_dir,  # 프로젝트 경로 지정
    name='train'  # 실험 이름
)
    