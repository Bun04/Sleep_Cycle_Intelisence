import csv
import random
import math
import pickle

#Xu ly du lieu
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
            
    # Tạo từ điển mã hóa 
    occ_map = {name: i for i, name in enumerate(occupations_set)}
    bmi_map = {name: i for i, name in enumerate(bmi_set)}
    
    # Gắn mã số vào dataset
    for row in dataset:
        row['Occ_Num'] = occ_map[row['Occupation']]
        row['BMI_Num'] = bmi_map[row['BMI']]
        
    return dataset, occ_map, bmi_map

# Cay quyet dinh
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
        return max(set(outcomes), key=outcomes.count) # Phân loại: Lấy nhãn xuất hiện nhiều nhất
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

## Random Forest
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
        # Random subset of features
        tree_features = random.sample(features, k=max(1, int(math.sqrt(len(features)))))
        tree = build_tree(sample, max_depth, min_size, tree_features, target_col, mode)
        trees.append(tree)
    return trees

# Predict one sample with a single tree
def predict_tree(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_tree(node['left'], row)
        return node['left']
    if isinstance(node['right'], dict):
        return predict_tree(node['right'], row)
    return node['right']

# Predict with a forest
def predict_rf(trees, row, mode):
    predictions = [predict_tree(tree, row) for tree in trees]
    if mode == 'classification':
        return max(set(predictions), key=predictions.count)
    return sum(predictions) / len(predictions)

# Split dataset into train/test portions
def train_test_split(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    cutoff = int(len(dataset_copy) * (1 - test_ratio))
    return dataset_copy[:cutoff], dataset_copy[cutoff:]

# Classification metrics
def evaluate_classification(actual, predicted):
    accuracy = sum(1 for a, p in zip(actual, predicted) if a == p) / len(actual)
    labels = sorted(set(actual))
    precision_by_label = {}
    recall_by_label = {}
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        precision_by_label[label] = tp / (tp + fp) if tp + fp else 0.0
        recall_by_label[label] = tp / (tp + fn) if tp + fn else 0.0
    avg_precision = sum(precision_by_label.values()) / len(labels)
    avg_recall = sum(recall_by_label.values()) / len(labels)
    return {
        'accuracy': accuracy,
        'precision_by_label': precision_by_label,
        'recall_by_label': recall_by_label,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall
    }

# Regression metrics
def evaluate_regression(actual, predicted):
    n = len(actual)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(mse)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }



# =========================================================
# RANDOM FOREST SYSTEM
# =========================================================
class SleepCycleSystem:
    def __init__(self):
        self.occ_map = {}
        self.bmi_map = {}
        self.rf_disorder = None
        self.rf_quality = None
        self.max_depth = 4
        self.min_size = 2
        self.sample_size = 0.8
        self.n_trees = 5

    def _load_csv_data(self, filepath):
        return load_and_process_csv(filepath)

    def train_from_csv(self, csv_path):
        print("Loading data from CSV...")
        dataset, self.occ_map, self.bmi_map = self._load_csv_data(csv_path)
        
        print("Splitting data...")
        train_set, test_set = train_test_split(dataset, test_ratio=0.2, seed=42)
        
        # Features for different models
        features_disorder = ['Age', 'Occ_Num', 'BMI_Num', 'Systolic', 'Diastolic', 'Stress Level']
        features_quality = ['Physical Activity Level', 'Stress Level']
        
        print("Start training Random Forest...")
        
        # Train disorder classification model
        self.rf_disorder = random_forest(
            train=train_set,
            max_depth=self.max_depth,
            min_size=self.min_size,
            sample_size=self.sample_size,
            n_trees=self.n_trees,
            features=features_disorder,
            target_col='Sleep Disorder',
            mode='classification'
        )
        
        # Train quality regression model
        self.rf_quality = random_forest(
            train=train_set,
            max_depth=self.max_depth,
            min_size=self.min_size,
            sample_size=self.sample_size,
            n_trees=self.n_trees,
            features=features_quality,
            target_col='Quality of Sleep',
            mode='regression'
        )

        print("Evaluating model...")
        
        # Evaluate disorder model
        disorder_actual = [row['Sleep Disorder'] for row in test_set]
        disorder_predicted = [predict_rf(self.rf_disorder, row, mode='classification') for row in test_set]
        disorder_metrics = evaluate_classification(disorder_actual, disorder_predicted)
        
        # Evaluate quality model
        quality_actual = [row['Quality of Sleep'] for row in test_set]
        quality_predicted = [predict_rf(self.rf_quality, row, mode='regression') for row in test_set]
        quality_metrics = evaluate_regression(quality_actual, quality_predicted)
        
        acc = disorder_metrics['accuracy']
        print(f"Accuracy: {acc:.4f}")
        return acc

    def predict_from_form(self, form_data):
        # Process input data
        bp = form_data['Blood Pressure'].split('/')
        systolic = int(bp[0])
        diastolic = int(bp[1])
        
        bmi = form_data['BMI Category']
        if bmi == 'Normal Weight': 
            bmi = 'Normal'
        
        processed_input = {
            'Age': float(form_data['Age']),
            'Occ_Num': self.occ_map.get(form_data['Occupation'], 0),
            'BMI_Num': self.bmi_map.get(bmi, 0),
            'Systolic': systolic,
            'Diastolic': diastolic,
            'Stress Level': float(form_data['Stress Level']),
            'Physical Activity Level': float(form_data['Physical Activity Level'])
        }

        # Make predictions
        pred_disorder = predict_rf(self.rf_disorder, processed_input, mode='classification')
        pred_quality = predict_rf(self.rf_quality, processed_input, mode='regression')

        return {
            "prediction": pred_disorder,
            "quality_score": round(pred_quality, 2)
        }

    def save_model(self, filepath):
        model_data = {
            'rf_disorder': self.rf_disorder,
            'rf_quality': self.rf_quality,
            'occ_map': self.occ_map,
            'bmi_map': self.bmi_map
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        system = SleepCycleSystem()
        system.rf_disorder = model_data['rf_disorder']
        system.rf_quality = model_data['rf_quality']
        system.occ_map = model_data['occ_map']
        system.bmi_map = model_data['bmi_map']
        return system


# =========================================================
# EXAMPLE USAGE
# =========================================================
if __name__ == "__main__":
    system = SleepCycleSystem()

    # Train model
    accuracy = system.train_from_csv(
        "/home/nauq-anh/django_project/Sleep_Cycle/Sleep_health_and_lifestyle_data.csv"
    )

    # Save model
    system.save_model("models/random_forest_model.pkl")

    # Predict from form
    sample_user = {
        "Gender": "Male",
        "Age": 25,
        "Occupation": "Engineer",
        "Sleep Duration": 6.0,
        "Quality of Sleep": 6,
        "Physical Activity Level": 35,
        "Stress Level": 8,
        "BMI Category": "Overweight",
        "Blood Pressure": "130/85",
        "Heart Rate": 78,
        "Daily Steps": 4000
    }

    result = system.predict_from_form(sample_user)

    print("Prediction:", result["prediction"])
    print("Quality Score:", result["quality_score"])