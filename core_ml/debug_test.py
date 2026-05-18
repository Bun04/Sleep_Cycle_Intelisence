import sys
sys.stdout.reconfigure(encoding='utf-8')

row = {'Name': 'Nguyen Van A', 'Gender': 'Male', 'Age': '30', 'Occupation': 'Software Engineer', 'Sleep Duration': '6.5', 'Quality of Sleep': '6', 'Physical Activity Level': '40', 'Stress Level': '7', 'BMI Category': 'Overweight', 'Blood Pressure': '135/85', 'Heart Rate': '75', 'Daily Steps': '5000', 'Sleep Disorder': 'None', 'Smoker': '1', 'Stroke_History': '0', 'Deep_Sleep_Percentage': '0.14', 'Heart_Disease': 'No'}

row_lower = {k.strip().lower(): v for k, v in row.items()}
print('Keys in row_lower:', list(row_lower.keys()))

# Check Deep_Sleep lookup - the code only checks these keys:
search_keys = ['deep sleep', 'deep_sleep', 'n3', 'phút ngủ sâu', 'deep sleep percentage']
for key in search_keys:
    found = key in row_lower
    print(f'  Key "{key}" in row_lower: {found} -> {row_lower.get(key, "NOT FOUND")}')

# The actual CSV column after lowering
print()
print('Actual key "deep_sleep_percentage" in row_lower:', 'deep_sleep_percentage' in row_lower)
print('Value:', row_lower.get('deep_sleep_percentage'))
print()
print('BUG: The code searches for "deep sleep percentage" (with spaces)')
print('     but the CSV column is "Deep_Sleep_Percentage" -> "deep_sleep_percentage" (with underscores)')
print('     So Deep_Sleep defaults to 0.0!')

# Also test the upload flow
sys.path.insert(0, '.')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sleep_project.settings')
from core_ml.sleep_dept_logic import predict_sleep_model_from_csv_row, _map_row_to_model_features, _load_sleep_model

model_data = _load_sleep_model()
mapped = _map_row_to_model_features(row, model_data)
print('\nMapped row:', mapped)
print('\nDeep_Sleep value in mapped:', mapped.get('Deep_Sleep') if mapped else 'N/A')
print('Expected: 0.14, Got:', mapped.get('Deep_Sleep') if mapped else 'N/A')

# Now test the full prediction
result = predict_sleep_model_from_csv_row(row)
print('\nPrediction result:', result)

# Test with Sleep Duration column mapping  
print('\n--- Checking Sleep Duration mapping ---')
sleep_keys = ['sleep hours', 'sleep_hours', 'hours', 'total_sleep_hrs', 'sleep duration', 'sleep duration (hrs)']
for key in sleep_keys:
    found = key in row_lower
    print(f'  Key "{key}" in row_lower: {found} -> {row_lower.get(key, "NOT FOUND")}')
