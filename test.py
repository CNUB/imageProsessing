import cv2
from ultralytics import YOLO

def save_bounding_boxes(image_path, output_path):
    # YOLOv8 모델 로드
    model = YOLO(r'D:\imageProcessing\bicycle\transfer_train2\weights\best.pt')  # 모델 파일 경로를 지정하세요

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 객체 감지
    results = model(img)

    # 바운딩 박스 그리기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0]  # (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 신뢰도와 클래스 ID 가져오기
            confidence = box.conf[0]  # 신뢰도
            class_id = int(box.cls[0])  # 클래스 ID

            # 클래스 이름 가져오기 (모델에 설정된 클래스 이름 사용)
            class_name = model.names[class_id]

            # 바운딩 박스와 클래스 이름, 신뢰도 그리기
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 바운딩 박스
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 텍스트

    # 결과 이미지 저장
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")

# 사용 예시
save_bounding_boxes(r'D:\test.jpg', r'D:\output_image.jpg')