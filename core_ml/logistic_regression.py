import csv
import math
import json
import os

class LogisticRegressionChay:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []  # Lưu trọng số [bias, w_time_in_bed, w_heart_rate]
        self.min_max = []  # Lưu giá trị Min, Max để chuẩn hóa dữ liệu mới sau này

    def _sigmoid(self, z):
        """Hàm Sigmoid ép giá trị về khoảng từ 0 đến 1"""
        # Giới hạn z để tránh lỗi tràn số (overflow) trong hàm math.exp()
        z = max(min(z, 250), -250)
        return 1.0 / (1.0 + math.exp(-z))

    def _normalize_data(self, dataset):
        """Chuẩn hóa dữ liệu (Min-Max Scaling) đưa về khoảng 0-1 để model học hiệu quả"""
        m = len(dataset)
        n_features = len(dataset[0])
        
        # Tìm Min và Max cho từng cột feature
        self.min_max = [[min(column), max(column)] for column in zip(*dataset)]
        
        normalized_dataset = []
        for row in dataset:
            norm_row = []
            for i in range(n_features):
                # Công thức: (X - Min) / (Max - Min)
                denominator = self.min_max[i][1] - self.min_max[i][0]
                if denominator == 0:
                    norm_row.append(0.0)
                else:
                    norm_row.append((row[i] - self.min_max[i][0]) / denominator)
            normalized_dataset.append(norm_row)
        return normalized_dataset

    def fit(self, X, y):
        """Huấn luyện mô hình bằng Gradient Descent"""
        # Chuẩn hóa dữ liệu đầu vào
        X_norm = self._normalize_data(X)
        
        n_features = len(X_norm[0])
        m = len(X_norm)
        
        # Khởi tạo trọng số bằng 0 (1 bias + n weights)
        self.weights = [0.0] * (n_features + 1)

        print(f"Bắt đầu huấn luyện với {m} dòng dữ liệu...")
        for epoch in range(self.epochs):
            # Khởi tạo mảng lưu độ dốc (gradient)
            gradients = [0.0] * (n_features + 1)
            
            # Tính toán lỗi cho toàn bộ tập dữ liệu (Batch Gradient Descent)
            for i in range(m):
                row = X_norm[i]
                # z = bias(w0) + w1*x1 + w2*x2
                z = self.weights[0]
                for j in range(n_features):
                    z += self.weights[j + 1] * row[j]
                
                prediction = self._sigmoid(z)
                error = prediction - y[i]
                
                # Tính Gradient
                gradients[0] += error  # Gradient cho bias
                for j in range(n_features):
                    gradients[j + 1] += error * row[j] # Gradient cho weights
            
            # Cập nhật trọng số sau mỗi Epoch
            self.weights[0] -= self.learning_rate * (gradients[0] / m)
            for j in range(n_features):
                self.weights[j + 1] -= self.learning_rate * (gradients[j + 1] / m)
                
            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{self.epochs} hoàn tất.")
        
        print("Huấn luyện thành công!")

    def save_model(self, filepath='saved_model.json'):
        """Lưu model ra file JSON (Vì không dùng thư viện joblib/pickle)"""
        model_data = {
            'weights': self.weights,
            'min_max': self.min_max
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Đã lưu mô hình tại: {filepath}")

# ==========================================
# PHẦN 2: XỬ LÝ DỮ LIỆU TỪ FILE CSV CỦA BẠN
# ==========================================
def parse_time_to_minutes(time_str):
    """Chuyển đổi '8:32' thành 512 phút"""
    if not time_str or ':' not in time_str:
        return 0
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def load_and_prep_data(csv_path):
    X = [] # Chứa các Feature (Time in bed, Heart rate)
    y = [] # Chứa nhãn Label (1: Tốt, 0: Kém)
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        # THÊM delimiter=';' Ở DÒNG DƯỚI ĐÂY
        reader = csv.DictReader(file, delimiter=';') 
        
        for row in reader:
            try:
                # 1. Trích xuất Label (Sleep quality)
                quality_str = row['Sleep quality'].replace('%', '').strip()
                if not quality_str: continue
                quality = float(quality_str)
                label = 1 if quality >= 75 else 0
                
                # 2. Trích xuất Features
                time_in_bed_mins = parse_time_to_minutes(row['Time in bed'])
                heart_rate_str = row['Heart rate'].strip()
                # Bỏ qua các dòng không có dữ liệu nhịp tim
                if not heart_rate_str: continue 
                heart_rate = float(heart_rate_str)
                
                X.append([time_in_bed_mins, heart_rate])
                y.append(label)
            except Exception as e:
                # In ra lỗi để xem dòng nào bị vấn đề (nếu có)
                # print(f"Lỗi dòng: {row} - Chi tiết: {e}") 
                continue 
                
    return X, y

# ==========================================
# PHẦN 3: KỊCH BẢN CHẠY CODE TRỰC TIẾP
# ==========================================
if __name__ == "__main__":
    # Đặt đường dẫn tới file CSV của bạn
    csv_file_path = 'sleepdata.csv' # Cập nhật lại đường dẫn thực tế nếu cần
    
    if not os.path.exists(csv_file_path):
        print("Lưu ý: Không tìm thấy file CSV, bạn cần đưa file sleepdata.csv vào đúng thư mục.")
    else:
        # 1. Đọc dữ liệu
        X_train, y_train = load_and_prep_data(csv_file_path)
        
        # 2. Khởi tạo và Huấn luyện
        # Learning rate 0.5 là phù hợp vì dữ liệu đã được chuẩn hóa Min-Max
        model = LogisticRegressionChay(learning_rate=0.5, epochs=3000)
        model.fit(X_train, y_train)
        
        # 3. Lưu Model (Lưu dưới dạng file JSON cơ bản)
        model.save_model('core_ml/sleep_model.json')