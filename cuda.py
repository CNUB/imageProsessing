import torch
print(torch.cuda.is_available())  # True가 나와야 GPU 사용 가능
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수
print(torch.cuda.get_device_name(0))  # GPU 이름