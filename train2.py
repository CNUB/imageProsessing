from ultralytics import YOLO
import os

def main():
    # 현재 파일의 절대 경로를 구함
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터셋 YAML 파일 경로 설정
    yaml_path = os.path.join(current_dir, 'data.yaml')  # 데이터 경로 수정

    # YOLOv8n 모델 생성 (전이 학습 대상)
    model = YOLO('yolov8n.pt')  

    # YOLOv8m 모델의 훈련된 가중치 로드
    pretrained_model_path = os.path.join(current_dir, 'train7', 'weights', 'best.pt')  # YOLOv8m 가중치 경로

    # 가중치 로드
    model.load(pretrained_model_path)

    # 전이 학습 시작
    results = model.train(
        data=yaml_path,     # 데이터 경로 (yaml 파일)
        epochs=50,          # 학습 반복 횟수 (필요에 따라 조정)
        imgsz=640,          # 이미지 크기
        batch=16,           # 배치 크기
        patience=20,        # 얼리 스톱 기준
        verbose=True,       # 학습 상태 출력
        project=current_dir,  # 프로젝트 경로 지정
        name='transfer_train',  # 전이 학습 이름
        device='0'          # GPU 사용
    )

if __name__ == '__main__':
    main()