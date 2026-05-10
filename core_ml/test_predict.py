import json
import math
import os # Thêm thư viện os

def predict_sleep_health(time_in_bed_minutes, heart_rate):
    # Dùng os để lấy chính xác thư mục chứa file test_predict.py hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Nối đường dẫn an toàn: current_dir + sleep_model.json
    model_path = os.path.join(current_dir, 'sleep_model.json')
    
    # 1. Mở và đọc bộ não AI bằng đường dẫn tuyệt đối vừa tạo
    with open(model_path, 'r') as f:
        model = json.load(f)
        
    weights = model['weights']
    min_max = model['min_max']
    
    # ... (giữ nguyên phần tính toán phía dưới của bạn)
    
    time_norm = (time_in_bed_minutes - min_max[0][0]) / (min_max[0][1] - min_max[0][0])
    hr_norm = (heart_rate - min_max[1][0]) / (min_max[1][1] - min_max[1][0])
    z = weights[0] + (weights[1] * time_norm) + (weights[2] * hr_norm)
    z = max(min(z, 250), -250)
    probability = 1.0 / (1.0 + math.exp(-z))
    
    return probability * 100

# ==========================================
# CHẠY THỬ NGHIỆM VỚI NGƯỜI DÙNG GIẢ LẬP
# ==========================================
if __name__ == "__main__":
    khach_1 = predict_sleep_health(480, 60)
    print(f"Khách 1 (Ngủ 8h, tim 60): Sức khỏe dự đoán đạt {khach_1:.1f}%")
    
    khach_2 = predict_sleep_health(240, 85)
    print(f"Khách 2 (Ngủ 4h, tim 85): Sức khỏe dự đoán đạt {khach_2:.1f}%")