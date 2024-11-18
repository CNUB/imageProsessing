import glob, os
import json

def get_all_categories(json_dir):
    categories = set()
    # 모든 JSON 파일을 순회하면서 카테고리 수집
    for json_file in glob.glob(os.path.join(json_dir, "*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for ann in data['annotations']:
                category = ann['category_id'][:3]  # 앞 3글자만 사용
                categories.add(category)
    
    # 정렬된 카테고리 목록으로 변환하고 번호 매핑
    categories = sorted(list(categories))
    class_mapping = {cat: idx for idx, cat in enumerate(categories)}
    
    print("발견된 클래스:")
    for cat, idx in class_mapping.items():
        print(f"{cat}: {idx}")
    
    return class_mapping

def convert_json_to_yolo(json_file, class_mapping):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    width = float(data['images']['width'])
    height = float(data['images']['height'])
    
    yolo_lines = []
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        
        x_center = (x + w/2) / width
        y_center = (y + h/2) / height
        w = w / width
        h = h / height
        
        category = ann['category_id'][:3]
        class_id = class_mapping[category]
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines

def create_yaml(class_mapping, yaml_path):
    yaml_content = "path: C:/Users/김종순/Desktop/MICT/bicycle/dataset\n"
    yaml_content += "train: train/images\n"
    yaml_content += "val: val/images\n\n"
    yaml_content += "names:\n"
    for cat, idx in class_mapping.items():
        yaml_content += f"  {idx}: {cat}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\ndata.yaml 파일이 생성되었습니다: {yaml_path}")

# 메인 실행 코드
json_dir = "./bicycle/json/1"
output_dir = "./bicycle/dataset/train/labels/"
yaml_path = "./bicycle/data.yaml"

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 모든 카테고리 찾기 및 매핑 생성
class_mapping = get_all_categories(json_dir)

# YAML 파일 생성
create_yaml(class_mapping, yaml_path)

# JSON 파일 변환
for json_file in glob.glob(os.path.join(json_dir, "*.json")):
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    
    try:
        yolo_lines = convert_json_to_yolo(json_file, class_mapping)
        
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_file, 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')
        print(f"변환 완료: {base_name}")
    except Exception as e:
        print(f"에러 발생 ({base_name}): {str(e)}")

print("\n모든 변환이 완료되었습니다!")