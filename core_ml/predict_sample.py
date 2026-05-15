import csv
import pickle

def predict_tree(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict): return predict_tree(node['left'], row)
        else: return node['left']
    else:
        if isinstance(node['right'], dict): return predict_tree(node['right'], row)
        else: return node['right']

def predict_rf(trees, row, mode):
    predictions = [predict_tree(tree, row) for tree in trees]
    if mode == 'classification':
        return max(set(predictions), key=predictions.count)
    else:
        return sum(predictions) / len(predictions)

def run_dynamic_prediction():
    # 1. LOAD MÔ HÌNH (Không cần train lại)
    try:
        with open('sleep_model_brain.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'sleep_model_brain.pkl'.")
        return

    # 2. ĐỌC FILE DỮ LIỆU CÁ NHÂN (Mọi kích cỡ)
    days_data = []
    try:
        with open('my_tracking_data.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                days_data.append(row)
    except FileNotFoundError:
        print("Lỗi: Hãy tạo file 'my_tracking_data.csv' trước.")
        return

    num_days = len(days_data)
    if num_days == 0:
        print("File dữ liệu trống!")
        return

    # 3. TÍNH TOÁN ĐỘNG (DYNAMIC AGGREGATION)
    total_sleep = total_physical = total_stress = total_sys = total_dia = 0

    for day in days_data:
        total_sleep += float(day['Sleep Duration'])
        total_physical += float(day['Physical Activity Level'])
        total_stress += float(day['Stress Level'])
        bp_split = day['Blood Pressure'].split('/')
        total_sys += int(bp_split[0])
        total_dia += int(bp_split[1])

    avg_sleep = total_sleep / num_days
    avg_physical = total_physical / num_days
    avg_stress = total_stress / num_days
    avg_sys = int(total_sys / num_days)
    avg_dia = int(total_dia / num_days)

    user_info = days_data[-1]
    occ_num = model_data['occ_map'].get(user_info['Occupation'], 0)
    bmi = user_info['BMI Category']
    if bmi == 'Normal Weight': bmi = 'Normal'
    bmi_num = model_data['bmi_map'].get(bmi, 0)

    processed_user = {
        'Age': float(user_info['Age']),
        'Occ_Num': occ_num,
        'BMI_Num': bmi_num,
        'Systolic': avg_sys,
        'Diastolic': avg_dia,
        'Stress Level': avg_stress,
        'Physical Activity Level': avg_physical
    }

    # 4. IN BÁO CÁO LINH HOẠT THEO THỜI GIAN
    print("="*75)
    print(f"BÁO CÁO GIẤC NGỦ CHU KỲ {num_days} NGÀY - DÀNH CHO: {user_info['Name'].upper()}")
    print("="*75)
    
    # Xác định loại chu kỳ
    period_type = "Ngắn hạn" if num_days <= 7 else "Trung hạn" if num_days <= 14 else "Dài hạn"
    print(f"Phân tích {period_type} dựa trên {num_days} bản ghi dữ liệu.\n")
    
    print(f"[Chỉ số trung bình]: HA: {avg_sys}/{avg_dia} | Ngủ: {avg_sleep:.1f}h | Stress: {avg_stress:.1f} | Vận động: {avg_physical:.1f}")
    print("-" * 75)

    pred_disorder = predict_rf(model_data['rf_disorder'], processed_user, mode='classification')
    pred_quality = predict_rf(model_data['rf_quality'], processed_user, mode='regression')

    print("1. DỰ ĐOÁN NGUY CƠ BỆNH LÝ (AI CLASSIFICATION):")
    if pred_disorder != 'None':
        print(f"   [!] CẢNH BÁO: AI phát hiện nguy cơ cao mắc **{pred_disorder.upper()}**.")
        if num_days >= 14:
            print(f"   -> Lời khuyên ĐẶC BIỆT: Tình trạng này đã kéo dài liên tục {num_days} ngày. Bạn KHÔNG NÊN chủ quan, hãy đặt lịch khám bác sĩ trong tuần tới.")
        else:
            print(f"   -> Lời khuyên: Dấu hiệu đang xuất hiện. Hãy theo dõi thêm từ 1-2 tuần nữa để có kết luận chính xác.")
    else:
        print("   [v] Tuyệt vời: Không phát hiện nguy cơ Mất ngủ hay Ngưng thở khi ngủ.")

    print(f"\n2. ĐÁNH GIÁ CHẤT LƯỢNG (AI REGRESSION): **{pred_quality:.1f} / 10 Điểm**")
    
if __name__ == '__main__':
    run_dynamic_prediction()