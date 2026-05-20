import os
import pickle

import numpy as np
from core_ml.random_forest import load_model

MODEL_FILENAME = "sleep_model.pkl"


def _normalize_text(value):
    return str(value).strip().lower()


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_sleep_model():
    import sys
    import core_ml.random_forest as rf
    import core_ml.knn as knn_module
    sys.modules['__main__'].RandomForest = rf.RandomForest

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, MODEL_FILENAME)

    if not os.path.exists(model_path):
        return None

    try:
        return load_model(model_path)
    except Exception:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                # If KNN model not present, try to build a lightweight KNN
                # from the original dataset file so classification can use KNN.
                if 'knn_disorder' not in model:
                    try:
                        project_root = os.path.dirname(current_dir)
                        dataset_file = os.path.join(project_root, 'Sleep_health_and_lifestyle_dataset.csv')
                        if os.path.exists(dataset_file):
                            dataset, occ_map, bmi_map = rf.load_and_process_csv(dataset_file)
                            # Build a KNN for disorder classification
                            features_disorder = [
                                'Age', 'Occ_Num', 'BMI_Num', 'Systolic', 'Diastolic', 'Stress Level'
                            ]
                            knn = knn_module.KNN(k=7)
                            knn.fit(dataset, features_disorder, 'Sleep Disorder')
                            model['knn_disorder'] = knn
                    except Exception:
                        pass

                return model
        except Exception:
            return None


def _parse_blood_pressure(row_lower):
    for key in ['blood pressure', 'bp', 'huyết áp', 'ha']:
        if key in row_lower:
            value = row_lower[key]
            if isinstance(value, str) and '/' in value:
                parts = value.split('/')
                if len(parts) >= 2:
                    return _to_int(parts[0].strip()), _to_int(parts[1].strip())
    if 'systolic' in row_lower and 'diastolic' in row_lower:
        return _to_int(row_lower['systolic']), _to_int(row_lower['diastolic'])
    return None, None


def _binary_prediction_label(value, positive='Cao', negative='Thấp'):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'cao', 'high', 'positive', 'pos'):
        return positive
    if normalized in ('0', 'false', 'no', 'thấp', 'low', 'negative', 'neg'):
        return negative
    return str(value)


def _map_row_to_model_features(row, model_data):
    if model_data is None:
        return None

    row_lower = {k.strip().lower(): v for k, v in row.items()}

    age = None
    for key in ['age', 'tuổi', 'age_years', 'năm']:
        if key in row_lower:
            age = _to_float(row_lower[key], None)
            break

    occupation = None
    for key in ['occupation', 'job', 'nghề nghiệp', 'profession']:
        if key in row_lower:
            occupation = str(row_lower[key]).strip()
            break

    bmi_value = None
    for key in ['bmi category', 'bmi', 'bmi_category']:
        if key in row_lower:
            bmi_value = str(row_lower[key]).strip()
            break

    systolic, diastolic = _parse_blood_pressure(row_lower)

    stress = None
    for key in ['stress level', 'stress', 'stress_score', 'stresslevel']:
        if key in row_lower:
            stress = _to_float(row_lower[key], None)
            break

    activity = None
    for key in ['physical activity level', 'physical activity', 'activity level', 'physical_activity_level', 'physicalactivitylevel']:
        if key in row_lower:
            activity = _to_float(row_lower[key], None)
            break

    heart_rate = None
    for key in ['heart rate', 'hr', 'rhr', 'resting heart rate', 'nhịp tim']:
        if key in row_lower:
            heart_rate = _to_float(row_lower[key], None)
            break

    deep_sleep = None
    for key in ['deep sleep', 'deep_sleep', 'deep_sleep_percentage', 'n3', 'phút ngủ sâu', 'deep sleep percentage']:
        if key in row_lower:
            deep_sleep = _to_float(row_lower[key], None)
            break

    if age is None or stress is None or activity is None or heart_rate is None or systolic is None or diastolic is None:
        return None

    occ_num = 0
    if occupation:
        occ_num = model_data.get('occ_map', {}).get(occupation, 0)

    bmi_num = 0
    if bmi_value:
        bmi_num = model_data.get('bmi_map', {}).get(bmi_value, 0)

    smoker = 0
    for key in ['smoker', 'hút thuốc', 'smokes']:
        if key in row_lower:
            val = str(row_lower[key]).strip().lower()
            if val in ['yes', 'y', '1', 'true', 'có', 'hút']:
                smoker = 1
            elif val in ['no', 'n', '0', 'false', 'không']:
                smoker = 0
            else:
                smoker = _to_int(row_lower[key], 0)
            break

    stroke = 0
    for key in ['stroke', 'tai biến', 'stroke_history', 'stroke history']:
        if key in row_lower:
            val = str(row_lower[key]).strip().lower()
            if val in ['yes', 'y', '1', 'true', 'có', 'bị']:
                stroke = 1
            elif val in ['no', 'n', '0', 'false', 'không']:
                stroke = 0
            else:
                stroke = _to_int(row_lower[key], 0)
            break
        
        
    # history_family_alzheimer = 0
    # for key in ['family_alzheimer','Người nhà bệnh','family_history_alzheimer']:
    #     if key in row_lower:
    #         val = str(row_lower[key]).strip().lower()
    #         if val in ['yes','y','1','có', 'bị','tre']:
    #               history_family_alzheimer = 1
    #         elif val in ['no', 'n', '0', 'false', 'không']:
    #               history_family_alzheimer = 0
    #         else:
    #             history_family_alzheimer = _to_int(row_lower[key],0)
    #         break

    return {
        'Age': age,
        'Occ_Num': occ_num,
        'BMI_Num': bmi_num,
        'Systolic': systolic,
        'Diastolic': diastolic,
        'Stress Level': stress,
        'Physical Activity Level': activity,
        'Heart Rate': heart_rate,
        'Deep_Sleep': deep_sleep if deep_sleep is not None else 0.0,
        'Smoker': smoker,
        'Stroke': stroke,
    }


def predict_sleep_model_from_csv_row(row):
    model_data = _load_sleep_model()
    if model_data is None:
        return None

    mapped_row = _map_row_to_model_features(row, model_data)
    if mapped_row is None:
        return None

    rf_disorder = model_data.get('rf_disorder')
    knn_disorder = model_data.get('knn_disorder')
    rf_quality = model_data.get('rf_quality')
    rf_heart = model_data.get('rf_heart')
    rf_alzheimer = model_data.get('rf_alzheimer')

    if rf_disorder is None or rf_quality is None or rf_heart is None:
        return None

    # Prefer KNN for the evaluation/classification step if available
    if knn_disorder is not None:
        raw_sleep_disorder = knn_disorder.predict_row(mapped_row)
    else:
        raw_sleep_disorder = rf_disorder.predict_row(mapped_row)
    raw_quality_score = rf_quality.predict_row(mapped_row)
    raw_heart = rf_heart.predict_row(mapped_row)
    raw_alzheimer = None
    if rf_alzheimer is not None:
        raw_alzheimer = rf_alzheimer.predict_row(mapped_row)

    result = {
        'sleep_disorder': raw_sleep_disorder,
        'quality_of_sleep': round(raw_quality_score, 1),
        'heart_disease': _binary_prediction_label(raw_heart),
        'alzheimer_risk': _binary_prediction_label(raw_alzheimer) if raw_alzheimer is not None else None,
        'model_file': MODEL_FILENAME,
    }

    return result


def predict_sleep_cycles(inputs):
    """
    Predict number of sleep cycles based on inputs using a simple formula.
    In a real model, this would be trained on data.
    """
    base_cycles = inputs['sleep_hours'] / 1.5
    adjustment = (inputs['deep_sleep_minutes'] - 75) * 0.01 + (inputs['rem_sleep_minutes'] - 90) * 0.005 - inputs['awakenings'] * 0.1
    predicted_cycles = base_cycles + adjustment
    return round(max(1, min(6, predicted_cycles)), 1)