from datetime import datetime, timedelta
import csv
import io

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages

from core_ml.sleep_dept_logic import predict_sleep_cycles
from core_ml.knn import predict_sleep_quality_knn  # IMPORT KNN
from .models import SleepRecord, CSVUploadHistory

# GỘP CHUNG DỮ LIỆU: Vừa có thông số Đồng hồ (cũ) vừa có thông số Y tế (mới)
DEFAULT_INPUT = {
    # Các biến của logic cũ (Random Forest / Wearable)
    "user_name": "Người dùng",
    "sleep_hours": 6.7,
    "deep_sleep_minutes": 82,
    "rem_sleep_minutes": 96,
    "awakenings": 4,
    "spo2_drop_events": 3,
    "hrv": 42,
    "rhr": 63,
    "consecutive_days": 5,
    "target_sleep_hours": 8.0,
    "wake_time": "06:30",
    "spo2_min": 89,
    
    # Các biến của logic mới (KNN / 13 trường)
    "person_id": 1,
    "gender": "Male",
    "age": 30,
    "occupation": "Software Engineer",
    "sleep_duration": 6.7,
    "quality_of_sleep": 7,
    "physical_activity": 60,
    "stress_level": 5,
    "bmi_category": "Normal",
    "blood_pressure": "120/80",
    "heart_rate": 70,
    "daily_steps": 8000,
    "sleep_disorder": "None"
}


def upload_view(request):
    """Xử lý cả file upload CSV"""
    context = {
        "default_input": DEFAULT_INPUT,
        "feature_groups": _build_feature_groups(),
    }
    
    if request.method == "POST" and "sleep_data_file" in request.FILES:
        try:
            csv_file = request.FILES["sleep_data_file"]
            
            if not csv_file.name.endswith('.csv'):
                context["upload_error"] = "Vui lòng tải file CSV hợp lệ."
                return render(request, "upload.html", context)
            
            stream = io.TextIOWrapper(csv_file.file, encoding='utf-8-sig')
            csv_reader = csv.DictReader(stream)
            
            if not csv_reader.fieldnames:
                context["upload_error"] = "File CSV không có header row."
                return render(request, "upload.html", context)
            
            rows = list(csv_reader)
            if not rows:
                context["upload_error"] = "File CSV không có dữ liệu."
                return render(request, "upload.html", context)
            
            upload_history = CSVUploadHistory.objects.create(
                file_name=csv_file.name,
                rows_count=len(rows),
                columns_count=len(csv_reader.fieldnames),
                upload_status='processing'
            )
            
            first_row = rows[0]
            extracted_input = _extract_csv_row(first_row)
            
            upload_history.processed_rows = len(rows)
            upload_history.upload_status = 'completed'
            upload_history.save()
            
            context["upload_meta"] = {
                "file_name": csv_file.name,
                "rows": len(rows),
                "columns": len(csv_reader.fieldnames)
            }
            context["data_preview"] = _build_csv_preview(rows[:5])
            context["uploaded_input"] = extracted_input
            context["missing_columns"] = _get_missing_columns(first_row)
            
        except Exception as e:
            context["upload_error"] = f"Lỗi xử lý file: {str(e)}"
    
    return render(request, "upload.html", context)


def dashboard_view(request):
    inputs = _extract_inputs(request)
    
    # 1. CHẠY LOGIC CŨ (RANDOM FOREST) CỦA BẠN - Giữ nguyên 100%
    analysis = _analyze_sleep(inputs)
    analysis['algorithm_used'] = "Random Forest & KNN"
    
    # 2. CHẠY SONG SONG LOGIC MỚI (KNN) - Chuyên chuẩn đoán bệnh
    knn_result = predict_sleep_quality_knn(inputs)
    
    # 🔥 BẢO VỆ GIAO DIỆN: Nếu KNN lỗi hoặc không chạy được, gán mặc định là 0%
    if not knn_result:
        knn_result = {
            "label": "None",
            "insomnia_prob": 0,
            "apnea_prob": 0,
            "none_prob": 0,
            "score": 0
        }
    
    if request.method == "POST":
        try:
            SleepRecord.objects.create(
                user_name=inputs.get('user_name', 'User'),
                sleep_hours=inputs.get('sleep_hours'),
                deep_sleep_minutes=inputs.get('deep_sleep_minutes'),
                rem_sleep_minutes=inputs.get('rem_sleep_minutes'),
                awakenings=inputs.get('awakenings'),
                spo2_drop_events=inputs.get('spo2_drop_events'),
                hrv=inputs.get('hrv'),
                rhr=inputs.get('rhr'),
                consecutive_days=inputs.get('consecutive_days', 1),
                target_sleep_hours=inputs.get('target_sleep_hours'),
                quality_label=analysis.get('quality_label'),
                recovery_score=analysis.get('recovery_score'),
                algorithm_used="Random Forest & KNN"
            )
        except Exception as e:
            print(f"Error saving sleep record: {e}")
    
    context = {
        "inputs": inputs,
        "analysis": analysis,
        "knn_result": knn_result, # Trả kết quả KNN ra Dashboard
        "feature_groups": _build_feature_groups(),
    }
    return render(request, "dashboard.html", context)


def _extract_inputs(request):
    """Lấy dữ liệu từ Request (Lấy đủ cả bộ cũ và bộ mới)"""
    if request.method != "POST":
        return DEFAULT_INPUT.copy()

    inputs = {
        "user_name": request.POST.get("user_name", "").strip() or "Người dùng",
        "sleep_hours": _to_float(request.POST.get("sleep_hours") or request.POST.get("sleep_duration"), DEFAULT_INPUT["sleep_hours"]),
        "deep_sleep_minutes": _to_int(request.POST.get("deep_sleep_minutes"), DEFAULT_INPUT["deep_sleep_minutes"]),
        "rem_sleep_minutes": _to_int(request.POST.get("rem_sleep_minutes"), DEFAULT_INPUT["rem_sleep_minutes"]),
        "awakenings": _to_int(request.POST.get("awakenings"), DEFAULT_INPUT["awakenings"]),
        "spo2_drop_events": _to_int(request.POST.get("spo2_drop_events"), DEFAULT_INPUT["spo2_drop_events"]),
        "hrv": _to_int(request.POST.get("hrv"), DEFAULT_INPUT["hrv"]),
        "rhr": _to_int(request.POST.get("rhr"), DEFAULT_INPUT["rhr"]),
        "consecutive_days": _to_int(request.POST.get("consecutive_days"), DEFAULT_INPUT["consecutive_days"]),
        "target_sleep_hours": _to_float(request.POST.get("target_sleep_hours"), DEFAULT_INPUT["target_sleep_hours"]),
        "wake_time": request.POST.get("wake_time", DEFAULT_INPUT["wake_time"]).strip() or DEFAULT_INPUT["wake_time"],
        
        # Các trường mới cho KNN
        "person_id": _to_int(request.POST.get("person_id"), DEFAULT_INPUT["person_id"]),
        "gender": request.POST.get("gender", DEFAULT_INPUT["gender"]),
        "age": _to_int(request.POST.get("age"), DEFAULT_INPUT["age"]),
        "occupation": request.POST.get("occupation", DEFAULT_INPUT["occupation"]),
        "sleep_duration": _to_float(request.POST.get("sleep_duration") or request.POST.get("sleep_hours"), DEFAULT_INPUT["sleep_duration"]),
        "quality_of_sleep": _to_int(request.POST.get("quality_of_sleep"), DEFAULT_INPUT["quality_of_sleep"]),
        "physical_activity": _to_int(request.POST.get("physical_activity"), DEFAULT_INPUT["physical_activity"]),
        "stress_level": _to_int(request.POST.get("stress_level"), DEFAULT_INPUT["stress_level"]),
        "bmi_category": request.POST.get("bmi_category", DEFAULT_INPUT["bmi_category"]),
        "blood_pressure": request.POST.get("blood_pressure", DEFAULT_INPUT["blood_pressure"]),
        "heart_rate": _to_int(request.POST.get("heart_rate"), DEFAULT_INPUT["heart_rate"]),
        "daily_steps": _to_int(request.POST.get("daily_steps"), DEFAULT_INPUT["daily_steps"]),
        "sleep_disorder": request.POST.get("sleep_disorder", DEFAULT_INPUT["sleep_disorder"]),
    }
    
    if request.method == "POST":
        # 1. Suy luận số lần thức giấc từ Điểm chất lượng (Chất lượng cao -> Ít thức giấc)
        inputs["awakenings"] = max(0, int((10 - inputs["quality_of_sleep"]) / 1.5))
        
        # 2. Suy luận hệ tim mạch (HRV, RHR) từ Mức độ căng thẳng (Stress Level)
        inputs["hrv"] = max(20, 85 - (inputs["stress_level"] * 6)) # Stress thấp = HRV cao (Tốt)
        inputs["rhr"] = min(95, 50 + (inputs["stress_level"] * 4)) # Stress thấp = RHR thấp (Tốt)
        
        # 3. Phân bổ chu kỳ ngủ (Deep/REM) dựa trên thời gian ngủ và chất lượng
        total_minutes = inputs["sleep_duration"] * 60
        # Chất lượng càng cao, tỷ lệ Deep/REM càng tiệm cận mức lý tưởng (20-25%)
        inputs["deep_sleep_minutes"] = int(total_minutes * (0.15 + (inputs["quality_of_sleep"] * 0.01)))
        inputs["rem_sleep_minutes"] = int(total_minutes * (0.18 + (inputs["quality_of_sleep"] * 0.005)))
        
        # 4. Đánh giá rủi ro ngạt thở (SpO2 drops) dựa trên Thể trạng (BMI)
        if inputs["bmi_category"] == "Obese":
            inputs["spo2_drop_events"] = 5
        elif inputs["bmi_category"] == "Overweight":
            inputs["spo2_drop_events"] = 2
        else:
            inputs["spo2_drop_events"] = 0

    inputs["spo2_min"] = max(75, 95 - inputs["spo2_drop_events"] * 2)
    return inputs


def _analyze_sleep(inputs):
    """LOGIC TÍNH TOÁN CŨ CỦA BẠN (CÒN NGUYÊN 100%)"""
    sleep_minutes = int(inputs["sleep_hours"] * 60)
    deep_ratio = inputs["deep_sleep_minutes"] / sleep_minutes if sleep_minutes else 0
    rem_ratio = inputs["rem_sleep_minutes"] / sleep_minutes if sleep_minutes else 0
    duration_gap = max(0, inputs["target_sleep_hours"] - inputs["sleep_hours"])

    recovery_score = 55
    recovery_score += min(18, (inputs["sleep_hours"] - 5) * 6)
    recovery_score += min(12, max(0, inputs["deep_sleep_minutes"] - 60) * 0.18)
    recovery_score += min(8, max(0, inputs["rem_sleep_minutes"] - 80) * 0.1)
    recovery_score += min(10, max(0, inputs["hrv"] - 35) * 0.5)
    recovery_score -= inputs["awakenings"] * 4
    recovery_score -= inputs["spo2_drop_events"] * 3
    recovery_score -= max(0, inputs["rhr"] - 58) * 0.8
    recovery_score -= duration_gap * 8
    recovery_score = round(_clamp(recovery_score, 18, 98))

    if recovery_score >= 78:
        quality_label = "Tốt"
        quality_tone = "good"
    elif recovery_score >= 60:
        quality_label = "Khá"
        quality_tone = "moderate"
    else:
        quality_label = "Tệ"
        quality_tone = "risk"

    apnea_risk_score = _clamp(inputs["spo2_drop_events"] * 14 + inputs["awakenings"] * 6, 5, 95)
    if inputs["spo2_drop_events"] >= 4 and inputs["awakenings"] >= 3:
        apnea_label = "Có nguy cơ"
        apnea_tone = "risk"
    else:
        apnea_label = "Không có nguy cơ"
        apnea_tone = "good"

    stress_score = _clamp((65 - inputs["hrv"]) * 1.2 + max(0, inputs["rhr"] - 60) * 2.1, 10, 96)
    if stress_score >= 68:
        stress_label = "Cao"
        stress_tone = "risk"
    elif stress_score >= 42:
        stress_label = "Trung bình"
        stress_tone = "moderate"
    else:
        stress_label = "Thấp"
        stress_tone = "good"

    energy_score = round(_clamp(recovery_score + deep_ratio * 100 * 0.18 + rem_ratio * 100 * 0.15 - stress_score * 0.15, 12, 97))
    if energy_score >= 78:
        energy_label = "Tỉnh táo"
        energy_tone = "good"
    elif energy_score >= 55:
        energy_label = "Uể oải"
        energy_tone = "moderate"
    else:
        energy_label = "Kiệt sức"
        energy_tone = "risk"

    sleep_debt_hours = round(duration_gap * inputs["consecutive_days"], 1)
    debt_progress = round(_clamp((sleep_debt_hours / 10) * 100, 0, 100))
    ideal_bedtime = _calculate_bedtime(inputs["wake_time"], inputs["target_sleep_hours"], sleep_debt_hours)

    num_cycles = predict_sleep_cycles(inputs)
    optimal_cycle_wake = round(_clamp(num_cycles, 4, 6))
    optimal_wake_time = _calculate_optimal_wake(ideal_bedtime, optimal_cycle_wake * 1.5)

    sleep_cycles = _build_sleep_cycles(ideal_bedtime, inputs["sleep_hours"])

    recommendations = _build_recommendations(
        recovery_score=recovery_score, apnea_label=apnea_label, stress_label=stress_label,
        energy_label=energy_label, sleep_debt_hours=sleep_debt_hours, inputs=inputs,
    )
    highlights = _build_highlights(inputs, recovery_score, stress_score, apnea_risk_score)
    prediction_cards = _build_prediction_cards(
        quality_label=quality_label, quality_tone=quality_tone, recovery_score=recovery_score,
        apnea_label=apnea_label, apnea_tone=apnea_tone, apnea_risk_score=apnea_risk_score,
        stress_label=stress_label, stress_tone=stress_tone, stress_score=stress_score,
        energy_label=energy_label, energy_tone=energy_tone, energy_score=energy_score,
        sleep_debt_hours=sleep_debt_hours, debt_progress=debt_progress, ideal_bedtime=ideal_bedtime,
        num_cycles=num_cycles, optimal_wake_time=optimal_wake_time,
    )

    return {
        "quality_label": quality_label, "quality_tone": quality_tone, "recovery_score": recovery_score,
        "apnea_label": apnea_label, "apnea_tone": apnea_tone, "apnea_risk_score": round(apnea_risk_score),
        "stress_label": stress_label, "stress_tone": stress_tone, "stress_score": round(stress_score),
        "energy_label": energy_label, "energy_tone": energy_tone, "energy_score": energy_score,
        "sleep_debt_hours": sleep_debt_hours, "debt_progress": debt_progress, "ideal_bedtime": ideal_bedtime,
        "deep_ratio": round(deep_ratio * 100), "rem_ratio": round(rem_ratio * 100),
        "num_cycles": num_cycles, "optimal_wake_time": optimal_wake_time,
        "sleep_cycles": sleep_cycles, "highlights": highlights,
        "prediction_cards": prediction_cards, "recommendations": recommendations,
    }


def _extract_csv_row(row):
    """Trích xuất file CSV, ghép nối chuẩn xác cả 2 bộ biến"""
    column_mappings = {
        'user_name': ['user name', 'name', 'tên', 'tên người dùng', 'person id', 'id'],
        'sleep_hours': ['sleep hours', 'hours', 'sleep duration', 'thời gian ngủ'],
        'deep_sleep_minutes': ['deep sleep', 'deep', 'n3'],
        'rem_sleep_minutes': ['rem sleep', 'rem'],
        'awakenings': ['awakenings'],
        'spo2_drop_events': ['spo2 drop', 'spo2 events'],
        'hrv': ['hrv'],
        'rhr': ['rhr', 'heart rate', 'nhịp tim'],
        'age': ['age', 'tuổi'],
        'bmi_category': ['bmi category', 'bmi'],
        'blood_pressure': ['blood pressure', 'huyết áp'],
        'stress_level': ['stress level', 'mức căng thẳng'],
    }
    
    row_lower = {k.lower().strip(): v for k, v in row.items()}
    extracted = DEFAULT_INPUT.copy()
    
    for field, aliases in column_mappings.items():
        for alias in aliases:
            if alias.lower() in row_lower:
                value = row_lower[alias.lower()]
                if value == '' or value is None: 
                    continue
                try:
                    if field == 'user_name':
                        if 'id' in alias.lower():
                            extracted[field] = f"User {str(value).strip()}"
                        else:
                            extracted[field] = str(value).strip()
                    elif field in ['sleep_hours', 'sleep_duration']:
                        extracted['sleep_hours'] = _to_float(value, DEFAULT_INPUT['sleep_hours'])
                        extracted['sleep_duration'] = _to_float(value, DEFAULT_INPUT['sleep_duration'])
                    elif field in ['blood_pressure', 'bmi_category', 'gender', 'occupation']:
                        extracted[field] = str(value).strip()
                    else:
                        extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                    break
                except: 
                    pass
                    
    extracted["spo2_min"] = max(75, 95 - extracted.get("spo2_drop_events", 0) * 2)
    return extracted


def _get_missing_columns(row):
    required = ['sleep hours', 'deep sleep', 'rem sleep']
    row_keys = [k.lower().strip() for k in row.keys()]
    return [col for col in required if not any(col in key or key in col for key in row_keys)]


def _build_csv_preview(rows):
    if not rows: return ""
    html = "<table class='csv-preview-table'><thead><tr>"
    for key in rows[0].keys(): html += f"<th>{key}</th>"
    html += "</tr></thead><tbody>"
    for row in rows[:5]:
        html += "<tr>"
        for value in row.values(): html += f"<td>{value}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


# ================= CÁC HÀM TIỆN ÍCH DƯỚI CÙNG GIỮ NGUYÊN =================

def _build_feature_groups():
    return [{"title": "Đánh giá & Phân loại chung", "eyebrow": "Nhóm 01", "description": "Cung cấp bức tranh tổng quan dễ hiểu.", "items": ["Phân loại chất lượng giấc ngủ tổng thể thành Tốt / Khá / Tệ.", "Tính Recovery Score từ 0 - 100 theo Deep Sleep, REM, thức giấc và tải phục hồi."]}, {"title": "Cảnh báo rủi ro sức khỏe", "eyebrow": "Nhóm 02", "description": "Biến dashboard thành công cụ cảnh báo sớm.", "items": ["Phát hiện nguy cơ Sleep Apnea bằng KNN.", "Dự đoán mức độ stress."]}, {"title": "Dự báo & Khuyến nghị", "eyebrow": "Nhóm 03", "description": "Dự báo trạng thái ngày mai.", "items": ["Dự đoán mức năng lượng ngày hôm sau.", "Tính Sleep Debt nhiều ngày."]}]

def _build_highlights(inputs, recovery_score, stress_score, apnea_risk_score):
    return [{"label": "Tổng thời gian ngủ", "value": f"{inputs['sleep_hours']:.1f} giờ", "detail": "So với mục tiêu sinh học đã chọn."}, {"label": "Deep Sleep", "value": f"{inputs['deep_sleep_minutes']} phút", "detail": "Phục hồi thể chất, cơ bắp và hormone."}, {"label": "REM Sleep", "value": f"{inputs['rem_sleep_minutes']} phút", "detail": "Hỗ trợ trí nhớ, cảm xúc và hiệu suất nhận thức."}, {"label": "Awakenings", "value": str(inputs["awakenings"]), "detail": "Số lần tỉnh giấc ngắt quãng trong đêm."}, {"label": "Stress Load", "value": f"{round(stress_score)}/100", "detail": f"HRV {inputs['hrv']} ms, RHR {inputs['rhr']} bpm."}, {"label": "Apnea Risk", "value": f"{round(apnea_risk_score)}%", "detail": f"{inputs['spo2_drop_events']} sự kiện giảm SpO2 được ghi nhận."}]

def _build_prediction_cards(*, quality_label, quality_tone, recovery_score, apnea_label, apnea_tone, apnea_risk_score, stress_label, stress_tone, stress_score, energy_label, energy_tone, energy_score, sleep_debt_hours, debt_progress, ideal_bedtime, num_cycles, optimal_wake_time):
    return [{"title": "Phân loại chất lượng giấc ngủ", "value": quality_label, "tone": quality_tone, "description": "Nhãn tổng quát.", "meta": f"Recovery nền: {recovery_score}/100"}, {"title": "Recovery Score", "value": f"{recovery_score}/100", "tone": quality_tone, "description": "Điểm phục hồi thể chất.", "meta": "Phân mảnh giấc ngủ làm giảm điểm."}, {"title": "Nguy cơ Sleep Apnea", "value": apnea_label, "tone": apnea_tone, "description": "Cảnh báo từ SpO2.", "meta": f"Rủi ro ước tính: {round(apnea_risk_score)}%"}, {"title": "Stress Level", "value": stress_label, "tone": stress_tone, "description": "Phản ánh trạng thái thần kinh.", "meta": f"Stress index: {round(stress_score)}/100"}, {"title": "Năng lượng ngày mai", "value": energy_label, "tone": energy_tone, "description": "Dự báo mức năng lượng.", "meta": f"Forecast score: {energy_score}/100"}, {"title": "Sleep Debt", "value": f"{sleep_debt_hours} giờ", "tone": "moderate" if sleep_debt_hours > 0 else "good", "description": "Thiếu hụt giấc ngủ.", "meta": f"Bedtime: {ideal_bedtime}"}, {"title": "Sleep Cycles", "value": f"{num_cycles} cycles", "tone": "good" if num_cycles >= 4 else "moderate", "description": "Số chu kỳ hoàn chỉnh.", "meta": f"Wake tối ưu: {optimal_wake_time}"}]

def _build_recommendations(*, recovery_score, apnea_label, stress_label, energy_label, sleep_debt_hours, inputs):
    recommendations = []
    if recovery_score < 60: recommendations.append("Ưu tiên thêm 45 - 60 phút ngủ trong 2 đêm tới để phục hồi nền.")
    if apnea_label == "Có nguy cơ": recommendations.append("Theo dõi thêm SpO2 và cân nhắc tư vấn y tế.")
    if stress_label == "Cao": recommendations.append("Giảm caffeine và thêm 10 phút thở chậm trước khi ngủ.")
    if energy_label != "Tỉnh táo": recommendations.append("Sắp xếp lịch nhẹ nhàng hơn vào sáng mai.")
    if sleep_debt_hours > 0: recommendations.append("Ngủ sớm hơn 30 phút mỗi đêm để giảm nợ ngủ.")
    return recommendations[:5]

def _build_sleep_cycles(bedtime_str, sleep_hours):
    try: bedtime = datetime.strptime(bedtime_str, "%H:%M")
    except: bedtime = datetime.strptime(DEFAULT_INPUT["wake_time"], "%H:%M") - timedelta(hours=DEFAULT_INPUT["target_sleep_hours"])
    cycles, current_time, total_minutes, cycle_length = [], bedtime, int(sleep_hours * 60), 90
    for i in range(total_minutes // cycle_length):
        cycle_end = current_time + timedelta(minutes=cycle_length)
        cycles.append({"time": f"{current_time.strftime('%H:%M')} - {cycle_end.strftime('%H:%M')}", "stages": [{"name": "N1", "class": "n1", "duration": 5}, {"name": "N2", "class": "n2", "duration": 20}, {"name": "N3", "class": "n3", "duration": 25}, {"name": "REM", "class": "rem", "duration": 40}]})
        current_time = cycle_end
    return cycles

def _calculate_bedtime(wake_time, target_sleep_hours, sleep_debt_hours):
    try: wake_dt = datetime.strptime(wake_time, "%H:%M")
    except: wake_dt = datetime.strptime(DEFAULT_INPUT["wake_time"], "%H:%M")
    return (wake_dt - timedelta(hours=target_sleep_hours + (min(1.0, sleep_debt_hours / 7) if sleep_debt_hours else 0))).strftime("%H:%M")

def _calculate_optimal_wake(bedtime, sleep_hours):
    try: bed_dt = datetime.strptime(bedtime, "%H:%M")
    except: bed_dt = datetime.strptime("22:30", "%H:%M")
    return (bed_dt + timedelta(hours=sleep_hours)).strftime("%H:%M")

def _to_float(value, fallback):
    try: return float(value)
    except: return fallback

def _to_int(value, fallback):
    try: return int(float(value))
    except: return fallback

def _clamp(value, min_value, max_value): return max(min_value, min(max_value, value))