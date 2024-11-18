import os
from pathlib import Path

# 클래스 매핑 정의
class_mapping = {
    'B01': 0,
    'B05': 1,
    'A01': 2,
    'C01': 3
}

def convert_label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:  # 빈 줄 건너뛰기
            class_name = parts[0]
            if class_name in class_mapping:
                # 클래스 이름을 숫자로 변환하고 나머지 좌표는 그대로 유지
                converted_line = f"{class_mapping[class_name]} {' '.join(parts[1:])}"
                converted_lines.append(converted_line)

    # 변환된 내용을 같은 파일에 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines))

def convert_directory(directory):
    # .txt 파일 찾기
    label_files = Path(directory).rglob('*.txt')
    
    for label_file in label_files:
        print(f"Converting {label_file}")
        convert_label_file(str(label_file))

# train과 val 디렉토리의 라벨 변환
base_path = "dataset"  # 실제 경로에 맞게 수정하세요
for split in ['train', 'val']:
    labels_dir = os.path.join(base_path, split, 'labels')
    if os.path.exists(labels_dir):
        print(f"\nProcessing {split} labels...")
        convert_directory(labels_dir)
        print(f"Completed {split} conversion")

print("\nLabel conversion completed!")