import csv
import pickle

# test function 
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

def run_weekly_prediction():
    print("Đang nạp dữ liệu từ 'sleep_model.pkl'...")
    try:
        with open('sleep_model_brain.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file mô hình. Hãy chạy huấn luyện trước!")
        return

    rf_disorder = model_data['rf_disorder']
    rf_quality = model_data['rf_quality']
    occ_map = model_data['occ_map']
    bmi_map = model_data['bmi_map']

    print("Đang đọc và phân tích dữ liệu 7 ngày từ 'personal_test.csv'...\n")
    days_data = []
    try:
        with open('personal_test.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                days_data.append(row)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'personal_test.csv'")
        return

    if not days_data:
        print("File dữ liệu trống!")
        return

    num_days = len(days_data)
    
    #TÍNH TOÁN TRUNG BÌNh
    total_sleep = 0.0
    total_physical = 0.0
    total_stress = 0.0
    total_sys = 0
    total_dia = 0

    for day in days_data:
        total_sleep += float(day['Sleep Duration'])
        total_physical += float(day['Physical Activity Level'])
        total_stress += float(day['Stress Level'])
        
        bp_split = day['Blood Pressure'].split('/')
        total_sys += int(bp_split[0])
        total_dia += int(bp_split[1])

    # Chỉ số trung bình của tuần
    avg_sleep = total_sleep / num_days
    avg_physical = total_physical / num_days
    avg_stress = total_stress / num_days
    avg_sys = int(total_sys / num_days)
    avg_dia = int(total_dia / num_days)

    # Lấy thông tin cá nhân cơ bản từ dòng cuối cùng
    user_info = days_data[-1]
    bmi = user_info['BMI Category']
    if bmi == 'Normal Weight': bmi = 'Normal'
    
    occ_num = occ_map.get(user_info['Occupation'], 0)
    bmi_num = bmi_map.get(bmi, 0)

    # Đóng gói dữ liệu trung bình để gửi cho mô hình
    processed_user = {
        'Age': float(user_info['Age']),
        'Occ_Num': occ_num,
        'BMI_Num': bmi_num,
        'Systolic': avg_sys,
        'Diastolic': avg_dia,
        'Stress Level': avg_stress,
        'Physical Activity Level': avg_physical
    }

    #  HIỂN THỊ KẾT QUẢ RA TERMINAL
    print("="*70)
    print(f"BÁO CÁO PHÂN TÍCH GIẤC NGỦ (DỰA TRÊN {num_days} NGÀY) - DÀNH CHO: {user_info['Name'].upper()}")
    print("="*70)
    print(f"[Hồ sơ]: {user_info['Occupation']}, {user_info['Age']} tuổi | BMI: {bmi}")
    print(f"[Chỉ số trung bình {num_days} ngày]:")
    print(f"  - Thời gian ngủ : {avg_sleep:.1f} giờ/ngày")
    print(f"  - Huyết áp (HA) : {avg_sys}/{avg_dia}")
    print(f"  - Vận động      : {avg_physical:.1f}")
    print(f"  - Stress        : {avg_stress:.1f}/10")
    print("-" * 70)

    # Dự đoán
    pred_disorder = predict_rf(rf_disorder, processed_user, mode='classification')
    pred_quality = predict_rf(rf_quality, processed_user, mode='regression')

    print("1 DỰ ĐOÁN NGUY CƠ RỐI LOẠN GIẤC NGỦ:")
    print(f"   -> Kết luận từ Mô Hình: **{pred_disorder.upper()}**")
    if pred_disorder != 'None':
        print(f"   -> Đánh giá: Việc duy trì Huyết áp ở mức {avg_sys}/{avg_dia} và Stress {avg_stress:.1f} liên tục trong {num_days} ngày đang tạo áp lực lớn, dẫn tới rủi ro cao mắc {pred_disorder}.")
    else:
        print(f"   -> Đánh giá: Nhịp sinh học trong {num_days} ngày qua của bạn khá ổn định, không có dấu hiệu bệnh lý rõ ràng.")

    print("\n2. ĐÁNH GIÁ CHẤT LƯỢNG GIẤC NGỦ (DỰ KIẾN KẾT QUẢ TUẦN TỚI):")
    print(f"   --> Điểm số dự kiến: {pred_quality:.1f} / 10")
    
    
    print("\n[LỜI KHUYÊN TỪ HỆ THỐNG]:")
    if avg_stress >= 7:
        print("   ! Mức độ Stress trung bình của bạn đang khá cao. Hãy thử các bài tập giãn cơ hoặc nghe nhạc sóng Alpha trước khi ngủ.")
    if avg_sleep < 6.5:
        print(f"   ! Bạn đang ngủ hơi ít ({avg_sleep:.1f}h). Cố gắng ngủ sớm hơn 30 phút vào tuần tới.")
    if avg_sys >= 135:
        print(f"   ! Huyết áp tâm thu trung bình của bạn ({avg_sys}) đang ở mức cảnh báo. Cần chú ý chế độ ăn giảm muối và theo dõi thêm.")
    if avg_stress < 7 and avg_sleep >= 6.5 and avg_sys < 135:
         print("   ★ Tuyệt vời! Bạn đang duy trì lối sống rất lành mạnh. Hãy tiếp tục phát huy!")
    print("="*70)
    if pred_disorder != 'None' or pred_quality < 7:
        print(" Lưu ý: Dù kết quả dự đoán có thể không hoàn hảo, nhưng nếu bạn thấy có dấu hiệu bất thường hoặc cảm thấy không ổn, hãy cân nhắc đi khám chuyên khoa để được tư vấn chi tiết hơn.")
    print("Đây chỉ là dự đoán dựa trên dữ liệu đã cung cấp, không thay thế cho chẩn đoán y tế chính thức.")
    print("="*70)

if __name__ == '__main__':
    run_weekly_prediction()