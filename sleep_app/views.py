from datetime import datetime, timedelta
import csv
import io

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages

from core_ml.sleep_dept_logic import predict_sleep_cycles, predict_sleep_model_from_csv_row
from .models import SleepRecord, CSVUploadHistory


DEFAULT_INPUT = {
    "user_name": "Nguyen An",
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
    "spo2_min": 89,  # 95 - 3*2
    "algorithm_choice": "Auto",
}


def upload_view(request):
    """Xử lý cả file upload và form input"""
    context = {
        "default_input": DEFAULT_INPUT,
        "feature_groups": _build_feature_groups(),
    }
    
    # Xử lý upload file CSV
    if request.method == "POST" and "sleep_data_file" in request.FILES:
        try:
            csv_file = request.FILES["sleep_data_file"]
            
            # Kiểm tra loại file
            if not csv_file.name.endswith('.csv'):
                context["upload_error"] = "Vui lòng tải file CSV hợp lệ."
                return render(request, "upload.html", context)
            
            # Xử lý CSV
            stream = io.TextIOWrapper(csv_file.file, encoding='utf-8-sig')
            csv_reader = csv.DictReader(stream)
            
            if not csv_reader.fieldnames:
                context["upload_error"] = "File CSV không có header row."
                return render(request, "upload.html", context)
            
            rows = list(csv_reader)
            if not rows:
                context["upload_error"] = "File CSV không có dữ liệu."
                return render(request, "upload.html", context)
            
            # Lưu lịch sử upload
            upload_history = CSVUploadHistory.objects.create(
                file_name=csv_file.name,
                rows_count=len(rows),
                columns_count=len(csv_reader.fieldnames),
                upload_status='processing'
            )
            
            # Trích xuất dữ liệu từ dòng đầu tiên
            first_row = rows[0]
            extracted_input = _extract_csv_row(first_row)
            model_prediction = predict_sleep_model_from_csv_row(first_row)
            
            # Lưu dữ liệu vào database
            sleep_record = SleepRecord.objects.create(
                user_name=extracted_input.get('user_name', 'CSV Upload'),
                sleep_hours=extracted_input.get('sleep_hours', 7.0),
                deep_sleep_minutes=extracted_input.get('deep_sleep_minutes', 90),
                rem_sleep_minutes=extracted_input.get('rem_sleep_minutes', 100),
                awakenings=extracted_input.get('awakenings', 2),
                spo2_drop_events=extracted_input.get('spo2_drop_events', 0),
                hrv=extracted_input.get('hrv', 50),
                rhr=extracted_input.get('rhr', 60),
                consecutive_days=extracted_input.get('consecutive_days', 1),
                target_sleep_hours=extracted_input.get('target_sleep_hours', 8.0),
            )
            
            upload_history.processed_rows = len(rows)
            upload_history.upload_status = 'completed'
            upload_history.save()
            
            # Xây dựng preview dữ liệu
            context["upload_meta"] = {
                "file_name": csv_file.name,
                "rows": len(rows),
                "columns": len(csv_reader.fieldnames)
            }
            context["data_preview"] = _build_csv_preview(rows[:5])
            context["uploaded_input"] = extracted_input
            context["missing_columns"] = _get_missing_columns(first_row)
            context["model_predictions"] = model_prediction
            
        except Exception as e:
            context["upload_error"] = f"Lỗi xử lý file: {str(e)}"
            upload_history.upload_status = 'failed'
            upload_history.error_message = str(e)
            upload_history.save()
    
    return render(request, "upload.html", context)


def dashboard_view(request):
    inputs = _extract_inputs(request)
    
    # Chọn thuật toán từ người dùng hoặc tự động
    algorithm_choice = inputs.get("algorithm_choice", "Auto")
    if algorithm_choice.lower() == "auto":
        selected_algorithm = _select_algorithm(inputs)
    else:
        selected_algorithm = algorithm_choice
    
    analysis = _analyze_sleep(inputs)
    analysis['algorithm_used'] = selected_algorithm

    model_predictions = _extract_model_predictions(request)
    if model_predictions:
        analysis['sleep_disorder'] = model_predictions.get('sleep_disorder', analysis.get('sleep_disorder'))
        analysis['quality_of_sleep'] = model_predictions.get('quality_of_sleep', analysis.get('quality_of_sleep'))

        heart_label = model_predictions.get('heart_disease')
        if heart_label is not None:
            analysis['heart_disease_label'] = heart_label
            analysis['heart_disease_tone'] = 'risk' if heart_label.lower() in ['cao', 'high', 'yes', 'true', '1'] else 'good'
            analysis['heart_disease_score'] = 85 if analysis['heart_disease_tone'] == 'risk' else 15

        alzheimer_label = model_predictions.get('alzheimer_risk')
        if alzheimer_label is not None:
            analysis['alzheimer_risk_label'] = alzheimer_label
            analysis['alzheimer_risk_tone'] = 'risk' if alzheimer_label.lower() in ['cao', 'high', 'yes', 'true', '1'] else 'good'
            analysis['alzheimer_risk_score'] = 80 if analysis['alzheimer_risk_tone'] == 'risk' else 20
    
    # Lưu kết quả vào database nếu là request POST
    if request.method == "POST":
        try:
            sleep_record = SleepRecord.objects.create(
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
                quality_tone=analysis.get('quality_tone'),
                recovery_score=analysis.get('recovery_score'),
                algorithm_used=selected_algorithm
            )
        except Exception as e:
            print(f"Error saving sleep record: {e}")
    
    context = {
        "inputs": inputs,
        "analysis": analysis,
        "feature_groups": _build_feature_groups(),
    }
    return render(request, "dashboard.html", context)


def _extract_inputs(request):
    if request.method != "POST":
        return DEFAULT_INPUT.copy()

    inputs = {
        "user_name": request.POST.get("user_name", DEFAULT_INPUT["user_name"]).strip() or DEFAULT_INPUT["user_name"],
        "sleep_hours": _to_float(request.POST.get("sleep_hours"), DEFAULT_INPUT["sleep_hours"]),
        "deep_sleep_minutes": _to_int(request.POST.get("deep_sleep_minutes"), DEFAULT_INPUT["deep_sleep_minutes"]),
        "rem_sleep_minutes": _to_int(request.POST.get("rem_sleep_minutes"), DEFAULT_INPUT["rem_sleep_minutes"]),
        "awakenings": _to_int(request.POST.get("awakenings"), DEFAULT_INPUT["awakenings"]),
        "spo2_drop_events": _to_int(request.POST.get("spo2_drop_events"), DEFAULT_INPUT["spo2_drop_events"]),
        "hrv": _to_int(request.POST.get("hrv"), DEFAULT_INPUT["hrv"]),
        "rhr": _to_int(request.POST.get("rhr"), DEFAULT_INPUT["rhr"]),
        "consecutive_days": _to_int(request.POST.get("consecutive_days"), DEFAULT_INPUT["consecutive_days"]),
        "target_sleep_hours": _to_float(request.POST.get("target_sleep_hours"), DEFAULT_INPUT["target_sleep_hours"]),
        "wake_time": request.POST.get("wake_time", DEFAULT_INPUT["wake_time"]).strip() or DEFAULT_INPUT["wake_time"],
        "algorithm_choice": request.POST.get("algorithm_choice", DEFAULT_INPUT["algorithm_choice"]),
    }

    inputs["spo2_min"] = 95 - inputs["spo2_drop_events"] * 2
    return inputs


def _extract_model_predictions(request):
    if request.method != "POST":
        return None

    predictions = {}
    for key in ["sleep_disorder", "quality_of_sleep", "heart_disease", "alzheimer_risk"]:
        value = request.POST.get(f"model_{key}")
        if value is not None:
            normalized = str(value).strip()
            if normalized and normalized.lower() != 'none':
                predictions[key] = normalized

    return predictions if predictions else None


def _analyze_sleep(inputs):
    total_sleep_minutes = max(inputs["sleep_hours"] * 60.0, 1.0)
    deep_ratio = inputs["deep_sleep_minutes"] / total_sleep_minutes
    rem_ratio = inputs["rem_sleep_minutes"] / total_sleep_minutes
    duration_gap = max(0.0, inputs["target_sleep_hours"] - inputs["sleep_hours"])

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

    heart_disease_score = 0
    heart_disease_label = "Chưa xác định"
    heart_disease_tone = "moderate"

    alzheimer_risk_score = 0
    alzheimer_risk_label = "Chưa xác định"
    alzheimer_risk_tone = "moderate"

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
        recovery_score=recovery_score,
        apnea_label=apnea_label,
        stress_label=stress_label,
        energy_label=energy_label,
        sleep_debt_hours=sleep_debt_hours,
        inputs=inputs,
    )
    highlights = _build_highlights(inputs, recovery_score, stress_score, apnea_risk_score)
    prediction_cards = _build_prediction_cards(
        quality_label=quality_label,
        quality_tone=quality_tone,
        recovery_score=recovery_score,
        apnea_label=apnea_label,
        apnea_tone=apnea_tone,
        apnea_risk_score=apnea_risk_score,
        stress_label=stress_label,
        stress_tone=stress_tone,
        stress_score=stress_score,
        energy_label=energy_label,
        energy_tone=energy_tone,
        energy_score=energy_score,
        sleep_debt_hours=sleep_debt_hours,
        debt_progress=debt_progress,
        ideal_bedtime=ideal_bedtime,
        num_cycles=num_cycles,
        optimal_wake_time=optimal_wake_time,
    )

    return {
        "quality_label": quality_label,
        "quality_tone": quality_tone,
        "recovery_score": recovery_score,
        "apnea_label": apnea_label,
        "apnea_tone": apnea_tone,
        "apnea_risk_score": round(apnea_risk_score),
        "stress_label": stress_label,
        "stress_tone": stress_tone,
        "stress_score": round(stress_score),
        "heart_disease_label": heart_disease_label,
        "heart_disease_tone": heart_disease_tone,
        "heart_disease_score": heart_disease_score,
        "alzheimer_risk_label": alzheimer_risk_label,
        "alzheimer_risk_tone": alzheimer_risk_tone,
        "alzheimer_risk_score": alzheimer_risk_score,
        "energy_label": energy_label,
        "energy_tone": energy_tone,
        "energy_score": energy_score,
        "sleep_debt_hours": sleep_debt_hours,
        "debt_progress": debt_progress,
        "ideal_bedtime": ideal_bedtime,
        "deep_ratio": round(deep_ratio * 100),
        "rem_ratio": round(rem_ratio * 100),
        "num_cycles": num_cycles,
        "optimal_wake_time": optimal_wake_time,
        "sleep_cycles": sleep_cycles,
        "highlights": highlights,
        "prediction_cards": prediction_cards,
        "recommendations": recommendations,
    }


def _build_feature_groups():
    return [
        {
            "title": "Đánh giá & Phân loại chung",
            "eyebrow": "Nhóm 01",
            "description": "Cung cấp bức tranh tổng quan dễ hiểu cho người dùng phổ thông ngay sau một lần tải dữ liệu.",
            "items": [
                "Phân loại chất lượng giấc ngủ tổng thể thành Tốt / Khá / Tệ.",
                "Tính Recovery Score từ 0 - 100 theo Deep Sleep, REM, thức giấc và tải phục hồi.",
            ],
        },
        {
            "title": "Cảnh báo rủi ro sức khỏe",
            "eyebrow": "Nhóm 02",
            "description": "Biến dashboard thành công cụ cảnh báo sớm, hữu ích hơn cho theo dõi cá nhân và báo cáo sức khỏe.",
            "items": [
                "Phát hiện nguy cơ Sleep Apnea từ SpO2 giảm và thức giấc ngắn.",
                "Dự đoán mức độ stress dựa trên HRV và RHR trong khi ngủ.",
                "Ước tính nguy cơ bệnh tim dựa trên nhịp tim và các chỉ số ngủ.",
                "Dự báo khả năng Alzheimer từ dữ liệu sức khỏe giấc ngủ.",
            ],
        },
        {
            "title": "Dự báo & Khuyến nghị",
            "eyebrow": "Nhóm 03",
            "description": "Không chỉ mô tả đêm qua, hệ thống còn dự báo trạng thái ngày mai và đưa ra hành động nên làm tiếp theo.",
            "items": [
                "Dự đoán mức năng lượng ngày hôm sau từ REM và Deep Sleep.",
                "Tính Sleep Debt nhiều ngày và đề xuất giờ đi ngủ lý tưởng.",
            ],
        },
    ]


def _build_highlights(inputs, recovery_score, stress_score, apnea_risk_score):
    return [
        {
            "label": "Tổng thời gian ngủ",
            "value": f"{inputs['sleep_hours']:.1f} giờ",
            "detail": "So với mục tiêu sinh học đã chọn.",
        },
        {
            "label": "Deep Sleep",
            "value": f"{inputs['deep_sleep_minutes']} phút",
            "detail": "Phục hồi thể chất, cơ bắp và hormone.",
        },
        {
            "label": "REM Sleep",
            "value": f"{inputs['rem_sleep_minutes']} phút",
            "detail": "Hỗ trợ trí nhớ, cảm xúc và hiệu suất nhận thức.",
        },
        {
            "label": "Awakenings",
            "value": str(inputs["awakenings"]),
            "detail": "Số lần tỉnh giấc ngắt quãng trong đêm.",
        },
        {
            "label": "Stress Load",
            "value": f"{round(stress_score)}/100",
            "detail": f"HRV {inputs['hrv']} ms, RHR {inputs['rhr']} bpm.",
        },
        {
            "label": "Apnea Risk",
            "value": f"{round(apnea_risk_score)}%",
            "detail": f"{inputs['spo2_drop_events']} sự kiện giảm SpO2 được ghi nhận.",
        },
    ]


def _build_prediction_cards(
    *,
    quality_label,
    quality_tone,
    recovery_score,
    apnea_label,
    apnea_tone,
    apnea_risk_score,
    stress_label,
    stress_tone,
    stress_score,
    energy_label,
    energy_tone,
    energy_score,
    sleep_debt_hours,
    debt_progress,
    ideal_bedtime,
    num_cycles,
    optimal_wake_time,
):
    return [
        {
            "title": "Phân loại chất lượng giấc ngủ",
            "value": quality_label,
            "tone": quality_tone,
            "description": "Nhãn tổng quát để người dùng hiểu nhanh chất lượng đêm ngủ.",
            "meta": f"Recovery nền: {recovery_score}/100",
        },
        {
            "title": "Recovery Score",
            "value": f"{recovery_score}/100",
            "tone": quality_tone,
            "description": "Điểm số phục hồi thể chất sau khi cân đối thời lượng ngủ và mức phân mảnh.",
            "meta": "Deep Sleep thấp hoặc tỉnh giấc nhiều sẽ làm giảm điểm.",
        },
        {
            "title": "Nguy cơ Sleep Apnea",
            "value": apnea_label,
            "tone": apnea_tone,
            "description": "Cảnh báo từ dữ liệu SpO2 và các lần thức giấc ngắn.",
            "meta": f"Mức rủi ro ước tính: {round(apnea_risk_score)}%",
        },
        {
            "title": "Stress Level",
            "value": stress_label,
            "tone": stress_tone,
            "description": "Phản ánh trạng thái thần kinh tự chủ qua HRV và nhịp tim nghỉ.",
            "meta": f"Stress index: {round(stress_score)}/100",
        },
        {
            "title": "Năng lượng ngày mai",
            "value": energy_label,
            "tone": energy_tone,
            "description": "Dự báo cảm giác tỉnh táo, uể oải hoặc kiệt sức vào ngày tiếp theo.",
            "meta": f"Forecast score: {energy_score}/100",
        },
        {
            "title": "Sleep Debt",
            "value": f"{sleep_debt_hours} giờ",
            "tone": "moderate" if sleep_debt_hours > 0 else "good",
            "description": "Tổng thiếu hụt giấc ngủ tích lũy qua nhiều đêm liên tiếp.",
            "meta": f"Nên lên giường lúc {ideal_bedtime} để giảm nợ ngủ. Tiến độ hiện tại {debt_progress}%.",
        },
        {
            "title": "Sleep Cycles",
            "value": f"{num_cycles} cycles",
            "tone": "good" if num_cycles >= 4 else "moderate",
            "description": "Số chu kỳ ngủ hoàn chỉnh trong đêm, mỗi chu kỳ khoảng 90 phút.",
            "meta": f"Giờ thức dậy tối ưu: {optimal_wake_time} để tránh tỉnh giấc giữa chu kỳ.",
        },
    ]


def _build_recommendations(*, recovery_score, apnea_label, stress_label, energy_label, sleep_debt_hours, inputs):
    recommendations = []

    # Use RF/KNN model predictions when available to refine recommendations
    try:
        # build a minimal row for model prediction
        deep_pct = 0.0
        try:
            total_minutes = max(inputs.get("sleep_hours", 0) * 60.0, 1.0)
            deep_pct = inputs.get("deep_sleep_minutes", 0) / total_minutes
        except Exception:
            deep_pct = 0.0

        model_row = {
            'Age': inputs.get('age', 40),
            'Occupation': inputs.get('occupation', ''),
            'BMI Category': inputs.get('bmi_category', 'Normal'),
            'Blood Pressure': f"{inputs.get('systolic',120)}/{inputs.get('diastolic',80)}",
            'Stress Level': recovery_score if recovery_score is not None else inputs.get('hrv', 40),
            'Physical Activity Level': inputs.get('physical_activity_level', 1.0),
            'Heart Rate': inputs.get('rhr', 60),
            'Deep_Sleep': deep_pct,
        }
        model_pred = predict_sleep_model_from_csv_row(model_row)
    except Exception:
        model_pred = None

    if recovery_score < 60:
        recommendations.append("Ưu tiên thêm 45 - 60 phút ngủ trong 2 đêm tới để phục hồi nền.")
    if apnea_label == "Có nguy cơ":
        recommendations.append("Theo dõi thêm SpO2 nhiều đêm liên tục và cân nhắc tư vấn y tế nếu dấu hiệu lặp lại.")
    if stress_label == "Cao":
        recommendations.append("Giảm caffeine sau 14:00 và thêm 10 phút thở chậm trước khi ngủ để cải thiện HRV.")
    if energy_label != "Tỉnh táo":
        recommendations.append("Sắp xếp lịch sáng mai nhẹ hơn, tránh buổi tập cường độ cao khi chỉ số năng lượng thấp.")
    if sleep_debt_hours > 0:
        recommendations.append("Dịch giờ ngủ sớm hơn ít nhất 30 phút mỗi đêm cho tới khi nợ ngủ quay về gần 0.")
    if inputs["deep_sleep_minutes"] < 75:
        recommendations.append("Tăng Deep Sleep bằng cách giữ phòng ngủ mát và cố định giờ đi ngủ hằng ngày.")
    if inputs["rem_sleep_minutes"] < 90:
        recommendations.append("Tránh dùng rượu buổi tối vì có thể làm giảm REM và khiến hôm sau kém tỉnh táo.")

    # Add model-based recommendations (random forest outputs)
    if model_pred:
        try:
            sd = model_pred.get('sleep_disorder')
            q = model_pred.get('quality_of_sleep')
            hd = model_pred.get('heart_disease')
            if sd and str(sd).lower() != 'none':
                recommendations.insert(0, f"AI dự đoán nguy cơ: {sd}. Xem xét đánh giá y tế nếu tiếp diễn.")
            if q is not None and float(q) < 5.5:
                recommendations.append("Chất lượng giấc ngủ dự báo thấp — ưu tiên can thiệp hành vi trước (hạn chế thiết bị, ánh sáng, tập thở).")
            if hd is not None and str(hd).strip().lower() in ['cao', 'high', 'yes', '1', 'true']:
                recommendations.append("Mô hình ước tính rủi ro tim mạch cao — cân nhắc kiểm tra ECG/TSH/khám tim.")
        except Exception:
            pass

    return recommendations[:5]


def _build_sleep_cycles(bedtime_str, sleep_hours):
    try:
        bedtime = datetime.strptime(bedtime_str, "%H:%M")
    except ValueError:
        bedtime = datetime.strptime(DEFAULT_INPUT["wake_time"], "%H:%M") - timedelta(hours=DEFAULT_INPUT["target_sleep_hours"])
    
    cycles = []
    current_time = bedtime
    total_minutes = int(sleep_hours * 60)
    cycle_length = 90  # minutes
    num_full_cycles = total_minutes // cycle_length
    
    for i in range(num_full_cycles):
        cycle_start = current_time
        cycle_end = current_time + timedelta(minutes=cycle_length)
        stages = [
            {"name": "N1", "class": "n1", "duration": 5},
            {"name": "N2", "class": "n2", "duration": 20},
            {"name": "N3", "class": "n3", "duration": 25},
            {"name": "REM", "class": "rem", "duration": 40},
        ]
        cycles.append({
            "time": f"{cycle_start.strftime('%H:%M')} - {cycle_end.strftime('%H:%M')}",
            "stages": stages,
        })
        current_time = cycle_end
    
    # Remaining time
    remaining = total_minutes % cycle_length
    if remaining > 0:
        cycle_start = current_time
        cycle_end = current_time + timedelta(minutes=remaining)
        stages = []
        for stage in [{"name": "N1", "class": "n1", "duration": 5}, {"name": "N2", "class": "n2", "duration": 20}, {"name": "N3", "class": "n3", "duration": 25}, {"name": "REM", "class": "rem", "duration": 40}]:
            if remaining > 0:
                actual_duration = min(remaining, stage["duration"])
                stages.append({**stage, "duration": actual_duration})
                remaining -= actual_duration
        cycles.append({
            "time": f"{cycle_start.strftime('%H:%M')} - {cycle_end.strftime('%H:%M')}",
            "stages": stages,
        })
    
    return cycles


def _calculate_bedtime(wake_time, target_sleep_hours, sleep_debt_hours):
    try:
        wake_dt = datetime.strptime(wake_time, "%H:%M")
    except ValueError:
        wake_dt = datetime.strptime(DEFAULT_INPUT["wake_time"], "%H:%M")

    catch_up_hours = min(1.0, sleep_debt_hours / 7) if sleep_debt_hours else 0
    total_sleep = target_sleep_hours + catch_up_hours
    bedtime = wake_dt - timedelta(hours=total_sleep)
    return bedtime.strftime("%H:%M")


def _calculate_optimal_wake(bedtime, sleep_hours):
    try:
        bed_dt = datetime.strptime(bedtime, "%H:%M")
    except ValueError:
        bed_dt = datetime.strptime("22:30", "%H:%M")  # Default bedtime

    wake_time = bed_dt + timedelta(hours=sleep_hours)
    return wake_time.strftime("%H:%M")


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _to_int(value, fallback):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


# ============= CSV Processing Functions =============

def _extract_csv_row(row):
    """
    Trích xuất dữ liệu từ một dòng CSV
    Hỗ trợ các tên cột linh hoạt (tên tiếng Anh hoặc Việt)
    """
    column_mappings = {
        'user_name': ['name', 'user name', 'user_name', 'username', 'họ tên', 'họ và tên', 'tên'],
        'sleep_hours': ['sleep hours', 'sleep_hours', 'hours', 'total_sleep_hrs', 'sleep duration', 'sleep duration (hrs)'],
        'deep_sleep_minutes': ['deep sleep', 'deep_sleep', 'deep_sleep_pct', 'deep sleep pct', 'deep sleep percentage', 'deep_sleep_percentage', 'phút ngủ sâu'],
        'rem_sleep_minutes': ['rem sleep', 'rem_sleep', 'rem_sleep_pct', 'rem sleep pct', 'rem sleep percentage', 'rem_sleep_percentage', 'phút rem'],
        'awakenings': ['awakenings', 'awakenings count', 'awake_count', 'awake count', 'lần thức giấc'],
        'spo2_drop_events': ['spo2 drop', 'spo2_drop_events', 'spo2 events', 'spo2 avg', 'spo2_avg', 'sự kiện spo2'],
        'hrv': ['hrv', 'heart rate variability'],
        'rhr': ['rhr', 'resting heart rate', 'heart rate', 'nhịp tim'],
    }

    row_lower = {k.lower().strip(): v for k, v in row.items()}
    extracted = DEFAULT_INPUT.copy()
    sleep_hours_value = None

    if 'user_name' in column_mappings:
        for alias in column_mappings['user_name']:
            if alias in row_lower:
                extracted['user_name'] = str(row_lower[alias]).strip()
                break

    if 'sleep_hours' in column_mappings:
        for alias in column_mappings['sleep_hours']:
            if alias in row_lower:
                sleep_hours_value = _to_float(row_lower[alias], DEFAULT_INPUT['sleep_hours'])
                extracted['sleep_hours'] = sleep_hours_value
                break

    for field, aliases in column_mappings.items():
        if field in ['sleep_hours', 'user_name']:
            continue

        for alias in aliases:
            if alias in row_lower:
                value = row_lower[alias]
                try:
                    if field == 'deep_sleep_minutes':
                        if 'pct' in alias or 'percentage' in alias:
                            if sleep_hours_value is not None:
                                extracted[field] = round(sleep_hours_value * 60.0 * _to_float(value, 0) / 100)
                            else:
                                extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                        else:
                            extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                    elif field == 'rem_sleep_minutes':
                        if 'pct' in alias or 'percentage' in alias:
                            if sleep_hours_value is not None:
                                extracted[field] = round(sleep_hours_value * 60.0 * _to_float(value, 0) / 100)
                            else:
                                extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                        else:
                            extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                    elif field == 'spo2_drop_events':
                        if alias in ['spo2 avg', 'spo2_avg', 'spо2 average', 'sp02 avg']:
                            spo2_avg = _to_float(value, None)
                            if spo2_avg is not None:
                                extracted[field] = max(0, round(max(0, 95 - spo2_avg) / 2))
                            else:
                                extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                        else:
                            extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                    elif field in ['hrv', 'rhr']:
                        extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                    else:
                        extracted[field] = _to_int(value, DEFAULT_INPUT[field])
                except Exception:
                    pass
                break

    extracted["spo2_min"] = max(75, 95 - extracted["spo2_drop_events"] * 2)
    return extracted


def _get_missing_columns(row):
    """Tìm các cột bắt buộc bị thiếu"""
    required_aliases = {
        'sleep hours': ['sleep hours', 'sleep_hours', 'total_sleep_hrs', 'sleep duration'],
        'deep sleep': ['deep sleep', 'deep_sleep', 'deep_sleep_pct', 'deep sleep pct', 'deep sleep percentage', 'deep_sleep_percentage'],
        'rem sleep': ['rem sleep', 'rem_sleep', 'rem_sleep_pct', 'rem sleep pct', 'rem sleep percentage', 'rem_sleep_percentage'],
        'awakenings': ['awakenings', 'awakenings count', 'awake_count', 'awake count'],
    }

    row_keys = [k.lower().strip() for k in row.keys()]
    missing = []
    for label, aliases in required_aliases.items():
        if not any(alias in key for key in row_keys for alias in aliases):
            missing.append(label)
    return missing


def _build_csv_preview(rows):
    """Tạo HTML preview table cho CSV data"""
    if not rows:
        return ""
    
    html = "<table class='csv-preview-table'>"
    html += "<thead><tr>"
    
    # Headers
    for key in rows[0].keys():
        html += f"<th>{key}</th>"
    html += "</tr></thead>"
    
    html += "<tbody>"
    for row in rows[:5]:
        html += "<tr>"
        for value in row.values():
            html += f"<td>{value}</td>"
        html += "</tr>"
    html += "</tbody>"
    html += "</table>"
    
    return html


def _select_algorithm(inputs):
    """
    Tự động chọn thuật toán phù hợp dựa trên đặc điểm dataset
    
    Logic chọn:
    - Recovery Score cao & consistent: Linear Regression
    - Dữ liệu phức tạp (stress cao, apnea risk): Random Forest
    - Dataset nhỏ hoặc cần classification: KNN/Logistic Regression
    """
    recovery_score = inputs['recovery_score'] if 'recovery_score' in inputs else 0
    stress_level = (65 - inputs['hrv']) * 1.2 + max(0, inputs['rhr'] - 60) * 2.1
    apnea_risk = inputs['spo2_drop_events'] * 14 + inputs['awakenings'] * 6
    
    # Tính điểm complexity
    complexity = stress_level * 0.4 + (apnea_risk / 100) * 0.4 + (abs(inputs['sleep_hours'] - inputs['target_sleep_hours']) * 10) * 0.2
    
    # Chọn thuật toán
    if complexity < 20:
        algorithm = "Linear Regression"
    elif complexity < 50:
        algorithm = "Logistic Regression"
    elif apnea_risk > 50 or stress_level > 70:
        algorithm = "Random Forest"
    else:
        algorithm = "KNN"
    
    return algorithm
