import csv
import random
import math
import pickle

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU THỦ CÔNG
# ==========================================
def load_and_process_csv(filename):
    dataset = []
    occupations_set = set()
    bmi_set = set()
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Tách huyết áp
            bp = row['Blood Pressure'].split('/')
            systolic = int(bp[0])
            diastolic = int(bp[1])
            
            # Chuẩn hóa BMI
            bmi = row['BMI Category']
            if bmi == 'Normal Weight': bmi = 'Normal'
            
            processed_row = {
                'Age': float(row['Age']),
                'Occupation': row['Occupation'],
                'Sleep Duration': float(row['Sleep Duration']),
                'Quality of Sleep': float(row['Quality of Sleep']),
                'Physical Activity Level': float(row['Physical Activity Level']),
                'Stress Level': float(row['Stress Level']),
                'BMI': bmi,
                'Systolic': systolic,
                'Diastolic': diastolic,
                'Sleep Disorder': row['Sleep Disorder']
            }
            dataset.append(processed_row)
            occupations_set.add(row['Occupation'])
            bmi_set.add(bmi)
            
    # Tạo từ điển mã hóa (Label Encoding)
    occ_map = {name: i for i, name in enumerate(occupations_set)}
    bmi_map = {name: i for i, name in enumerate(bmi_set)}
    
    # Gắn mã số vào dataset
    for row in dataset:
        row['Occ_Num'] = occ_map[row['Occupation']]
        row['BMI_Num'] = bmi_map[row['BMI']]
        
    return dataset, occ_map, bmi_map

# ==========================================
# 2. XÂY DỰNG CÂY QUYẾT ĐỊNH (DECISION TREE)
# ==========================================
def test_split(feature, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[feature] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def calculate_gini(groups, classes, target_col):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0: continue
        score = 0.0
        for class_val in classes:
            p = [row[target_col] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_best_split(dataset, features, target_col):
    class_values = list(set([row[target_col] for row in dataset]))
    b_index, b_value, b_score, b_groups = None, None, 999, None
    for feature in features:
        for row in dataset:
            groups = test_split(feature, row[feature], dataset)
            gini = calculate_gini(groups, class_values, target_col)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = feature, row[feature], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group, target_col, mode='classification'):
    outcomes = [row[target_col] for row in group]
    if mode == 'classification':
        return max(set(outcomes), key=outcomes.count) # Phân loại: Lấy số đông
    else:
        return sum(outcomes) / len(outcomes) # Hồi quy: Lấy trung bình cộng

def split(node, max_depth, min_size, depth, features, target_col, mode):
    left, right = node['groups']
    del(node['groups'])
    
    # Nếu 1 trong 2 nhánh trống
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right, target_col, mode)
        return
        
    # Đạt độ sâu tối đa
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, target_col, mode), to_terminal(right, target_col, mode)
        return
        
    # Xử lý nhánh trái
    if len(left) <= min_size:
        node['left'] = to_terminal(left, target_col, mode)
    else:
        node['left'] = get_best_split(left, features, target_col)
        split(node['left'], max_depth, min_size, depth+1, features, target_col, mode)
        
    # Xử lý nhánh phải
    if len(right) <= min_size:
        node['right'] = to_terminal(right, target_col, mode)
    else:
        node['right'] = get_best_split(right, features, target_col)
        split(node['right'], max_depth, min_size, depth+1, features, target_col, mode)

def build_tree(train, max_depth, min_size, features, target_col, mode):
    root = get_best_split(train, features, target_col)
    split(root, max_depth, min_size, 1, features, target_col, mode)
    return root

# ==========================================
# 3. THUẬT TOÁN RỪNG NGẪU NHIÊN (RANDOM FOREST)
# ==========================================
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def random_forest(train, max_depth, min_size, sample_size, n_trees, features, target_col, mode):
    trees = list()
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        # Random subset of features (Kỹ thuật Bagging)
        tree_features = random.sample(features, k=max(1, int(math.sqrt(len(features)))))
        tree = build_tree(sample, max_depth, min_size, tree_features, target_col, mode)
        trees.append(tree)
    return trees

# ==========================================
# 4. CHƯƠNG TRÌNH HUẤN LUYỆN VÀ LƯU MÔ HÌNH
# ==========================================
def train_and_save():
    print("="*60)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH AI (TỰ CODE)")
    print("="*60)
    
    # 1. Tải dữ liệu
    print("[1/3] Đang tải và xử lý dữ liệu gốc từ CSV...")
    try:
        dataset, occ_map, bmi_map = load_and_process_csv('Sleep_health_and_lifestyle_dataset.csv')
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'Sleep_health_and_lifestyle_dataset.csv'!")
        return
    
    # Các đặc trưng (features) dùng để dự đoán
    features_disorder = ['Age', 'Occ_Num', 'BMI_Num', 'Systolic', 'Diastolic', 'Stress Level']
    features_quality = ['Physical Activity Level', 'Stress Level']
    
    # 2. Huấn luyện mô hình
    print(f"[2/3] Đang huấn luyện Rừng ngẫu nhiên (Vui lòng đợi vài giây)...")
    
    # Mô hình 1: Dự đoán phân loại Bệnh lý (Classification)
    rf_disorder = random_forest(
        train=dataset, 
        max_depth=4, 
        min_size=2, 
        sample_size=0.8, 
        n_trees=5, 
        features=features_disorder, 
        target_col='Sleep Disorder', 
        mode='classification'
    )
    
    # Mô hình 2: Dự đoán điểm số Chất lượng giấc ngủ (Regression)
    rf_quality = random_forest(
        train=dataset, 
        max_depth=4, 
        min_size=2, 
        sample_size=0.8, 
        n_trees=5, 
        features=features_quality, 
        target_col='Quality of Sleep', 
        mode='regression'
    )

    # 3. Đóng gói và lưu
    print("[3/3] Đang đóng gói và xuất mô hình...")
    model_data = {
        'rf_disorder': rf_disorder,
        'rf_quality': rf_quality,
        'occ_map': occ_map,  # Lưu lại từ điển nghề nghiệp
        'bmi_map': bmi_map   # Lưu lại từ điển cân nặng
    }
    
    with open('sleep_model_brain.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("-" * 60)
    print("THÀNH CÔNG! Đã lưu toàn bộ 'Não bộ AI' vào file: sleep_model_brain.pkl")
    print("Bây giờ bạn có thể chạy file predict để dự đoán cá nhân.")
    print("="*60)

if __name__ == '__main__':
    # Fix seed để kết quả ổn định mỗi lần chạy (tùy chọn)
    random.seed(42) 
    train_and_save()