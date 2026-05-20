import csv
import math
import os
from django.conf import settings

# =========================================================
# CÁC HÀM TOÁN HỌC TỰ VIẾT (THAY THẾ SCIKIT-LEARN)
# =========================================================

def calculate_mean(data):
    """Tính giá trị trung bình (Mean)"""
    return sum(data) / len(data)

def calculate_std_dev(data, mean_val):
    """Tính độ lệch chuẩn (Standard Deviation)"""
    variance = sum((x - mean_val) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def euclidean_distance(row1, row2):
    """Tính khoảng cách Euclidean giữa 2 điểm dữ liệu trong không gian n-chiều"""
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# =========================================================
# HÀM XỬ LÝ CHÍNH
# =========================================================

def predict_sleep_quality_knn(user_inputs):
    """
    Thuật toán KNN tự lập trình (Code chay 100%)
    """
    try:
        csv_path = os.path.join(settings.BASE_DIR, 'Sleep_health_and_lifestyle_dataset.csv')
        k_neighbors = 5
        
        dataset = []
        
        # 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU TỪ CSV (Thay thế Pandas)
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Xử lý nhãn (Label)
                label = row.get('Sleep Disorder', '').strip()
                if not label or label.lower() == 'none':
                    label = 'None'
                
                # Tách huyết áp (VD: "120/80" -> 120.0 và 80.0)
                bp_str = row.get('Blood Pressure', '120/80')
                try:
                    sys, dia = map(float, bp_str.split('/'))
                except:
                    sys, dia = 120.0, 80.0
                
                # Mã hóa BMI
                bmi_str = row.get('BMI Category', 'Normal Weight').strip()
                if bmi_str == 'Normal Weight': bmi_str = 'Normal'
                bmi_mapping = {'Normal': 0.0, 'Overweight': 1.0, 'Obese': 2.0}
                bmi_val = bmi_mapping.get(bmi_str, 0.0)
                
                # Gộp đặc trưng (Features)
                try:
                    features = [
                        float(row['Age']),
                        float(row['Sleep Duration']),
                        float(row['Quality of Sleep']),
                        float(row['Physical Activity Level']),
                        float(row['Stress Level']),
                        float(row['Heart Rate']),
                        float(row['Daily Steps']),
                        sys,       # Tâm thu
                        dia,       # Tâm trương
                        bmi_val    # BMI mã hóa
                    ]
                    dataset.append((features, label))
                except ValueError:
                    continue # Bỏ qua dòng bị lỗi dữ liệu

        if not dataset:
            return None

        # 2. CHUẨN HÓA DỮ LIỆU Z-SCORE (Thay thế StandardScaler)
        # Công thức: z = (x - mean) / std_dev
        num_features = len(dataset[0][0])
        means = []
        stds = []
        
        # Tính mean và std cho từng cột (10 cột)
        for i in range(num_features):
            col_data = [row[0][i] for row in dataset]
            mean_val = calculate_mean(col_data)
            std_val = calculate_std_dev(col_data, mean_val)
            if std_val == 0: std_val = 1e-9 # Tránh lỗi chia cho 0
            
            means.append(mean_val)
            stds.append(std_val)
            
        # Chuẩn hóa tập dữ liệu train
        scaled_dataset = []
        for features, label in dataset:
            scaled_features = [(features[i] - means[i]) / stds[i] for i in range(num_features)]
            scaled_dataset.append((scaled_features, label))


        # 3. CHUẨN BỊ DỮ LIỆU NGƯỜI DÙNG NHẬP (Test Data)
        user_bp = user_inputs.get('blood_pressure', '120/80')
        try:
            u_sys, u_dia = map(float, user_bp.split('/'))
        except:
            u_sys, u_dia = 120.0, 80.0

        user_bmi = user_inputs.get('bmi_category', 'Normal')
        u_bmi_val = bmi_mapping.get(user_bmi, 0.0)

        user_features = [
            float(user_inputs.get('age', 30)),
            float(user_inputs.get('sleep_duration', 7.0)),
            float(user_inputs.get('quality_of_sleep', 7)),
            float(user_inputs.get('physical_activity', 60)),
            float(user_inputs.get('stress_level', 5)),
            float(user_inputs.get('heart_rate', 70)),
            float(user_inputs.get('daily_steps', 8000)),
            u_sys,
            u_dia,
            u_bmi_val
        ]

        # Chuẩn hóa Test Data dựa trên Mean và Std của Train Data
        scaled_user_features = [(user_features[i] - means[i]) / stds[i] for i in range(num_features)]

        # 4. TÍNH KHOẢNG CÁCH TỪ USER ĐẾN TẤT CẢ CÁC ĐIỂM TRONG DATASET
        distances = []
        for train_features, label in scaled_dataset:
            dist = euclidean_distance(scaled_user_features, train_features)
            
            # Tính trọng số (Khoảng cách càng gần, trọng số càng lớn)
            weight = 1.0 / (dist + 1e-5) 
            distances.append((dist, weight, label))

        # 5. SẮP XẾP VÀ TÌM K HÀNG XÓM GẦN NHẤT
        distances.sort(key=lambda x: x[0]) # Sắp xếp tăng dần theo khoảng cách
        neighbors = distances[:k_neighbors] # Lấy K hàng xóm đầu tiên

        votes = {"Insomnia": 0, "Sleep Apnea": 0, "None": 0}
        
        for _, _, label in neighbors:
            if label in votes:
                votes[label] += 1
            else:
                # Trường hợp có nhãn khác trong file CSV thì gộp vào nhóm liên quan hoặc None
                votes["None"] += 1

        # Tính % xác suất cho từng loại (Công thức: số phiếu / K * 100)
        # Với K=5, mỗi hàng xóm tương ứng 20%
        insomnia_prob = int((votes["Insomnia"] / k_neighbors) * 100)
        apnea_prob = int((votes["Sleep Apnea"] / k_neighbors) * 100)
        none_prob = int((votes["None"] / k_neighbors) * 100)

        # Tìm nhãn có phiếu cao nhất để làm kết luận cuối
        predicted_label = max(votes, key=votes.get)

        return {
            "label": predicted_label,
            "insomnia_prob": insomnia_prob,
            "apnea_prob": apnea_prob,
            "none_prob": none_prob,
            "score": int((votes[predicted_label] / k_neighbors) * 100) # Độ tự tin chung
        }

    except Exception as e:
        print(f"Lỗi khi chạy thuật toán KNN tự viết: {e}")
        return None