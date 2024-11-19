import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    # TensorFlow Lite 모델 로드
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_objects(interpreter, image):
    # 입력 텐서와 출력 텐서 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 디버깅: 출력 텐서 정보 확인
    print("Output Details:", output_details)

    # 이미지 전처리
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

    # 모델 실행
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 결과 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])  # [1, 28, 8400]
    print("Output Data Shape:", output_data.shape)

    # 데이터 분리
    output_data = output_data[0]  # [28, 8400]
    boxes = output_data[:4, :].T  # 바운딩 박스 (x, y, w, h), [8400, 4]
    class_scores = output_data[4:, :].T  # 클래스 신뢰도, [8400, 24]

    # 최고 클래스와 신뢰도 추출
    class_ids = np.argmax(class_scores, axis=1)  # 각 앵커별 최고 클래스
    confidences = np.max(class_scores, axis=1)  # 최고 클래스 신뢰도

    return boxes, class_ids, confidences

def save_bounding_boxes(image_path, output_path):
    # 모델 로드
    model_path = r"D:\imageProcessing\bicycle\transfer_train2\weights\best_float16.tflite"
    interpreter = load_model(model_path)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 객체 감지
    boxes, class_ids, confidences = detect_objects(interpreter, img)

    # 디버깅: 출력된 값 확인
    print("Boxes:", boxes)
    print("Class IDs:", class_ids)
    print("Confidences:", confidences)

    # 신뢰도 임계값
    confidence_threshold = 0.8  # 신뢰도 80% 이상만 표시
    h, w, _ = img.shape

    # 바운딩 박스 그리기
    for i in range(len(boxes)):
        if confidences[i] > confidence_threshold:  # 신뢰도 기준 필터링
            x, y, box_w, box_h = boxes[i]
            x1 = int((x - box_w / 2) * w)
            y1 = int((y - box_h / 2) * h)
            x2 = int((x + box_w / 2) * w)
            y2 = int((y + box_h / 2) * h)

            label = f"Class {class_ids[i]}: {confidences[i]:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 바운딩 박스
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 결과 이미지 저장
    cv2.imwrite(output_path, img)
save_bounding_boxes(r"D:\test.jpg", r"D:\output_image.jpg")
