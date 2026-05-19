import csv
import random
import math
import os

random.seed(42)

INPUT_FILE = "/home/nauq-anh/django_project/Sleep_Cycle/sample_test.csv"
OUTPUT_FILE = "Sleep_health_with_dataset.csv"



def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def generate_family_history():
    """
    20% dân số có tiền sử gia đình Alzheimer.
    Hoàn toàn độc lập với các cột khác — đây là yếu tố di truyền.
    """
    return "Yes" if random.random() < 0.20 else "No"


def generate_cognitive_score(age, stress, sleep_quality, sleep_duration):
    """
    Thang MMSE chuẩn: 0–30, bình thường >= 24, nghi ngờ < 24.

    Tương quan âm với:
    - Tuổi cao (não lão hóa)
    - Stress cao (cortisol làm hại vùng hippocampus)
    - Ngủ ít / kém chất lượng (thiếu dọn dẹp amyloid-beta)

    Base score 28 (người trẻ, khỏe mạnh), trừ dần theo risk factors.
    """
    base = 28.0

    # Tuổi: mỗi 10 năm trên 40 trừ ~1 điểm
    age_penalty = max(0, (age - 40) / 10.0)

    # Stress: mức 7-8 là cao, trừ mạnh hơn
    stress_penalty = max(0, (stress - 5) * 0.6)

    # Chất lượng ngủ thấp
    sleep_quality_penalty = max(0, (7 - sleep_quality) * 0.4)

    # Thời gian ngủ ngắn
    sleep_dur_penalty = max(0, (7.0 - sleep_duration) * 0.5)

    score = base - age_penalty - stress_penalty - sleep_quality_penalty - sleep_dur_penalty

    # Thêm nhiễu ngẫu nhiên ± 2 điểm (cá thể khác nhau)
    score += random.gauss(0, 1.5)

    return round(clamp(score, 0, 30), 1)


def generate_memory_test_score(cognitive_score):
    """
    Thang 0–10 (bài test nhớ danh sách từ chuẩn).
    Tương quan ~0.75 với Cognitive_Score + nhiễu độc lập.

    Cognitive 24–30 → Memory ~7–10
    Cognitive 18–24 → Memory ~4–7
    Cognitive < 18  → Memory ~0–4
    """
    # Scale cognitive (0–30) sang (0–10) với tương quan cao
    base = (cognitive_score / 30.0) * 10.0

    # Nhiễu ngẫu nhiên ± 1.5 điểm
    score = base + random.gauss(0, 1.2)

    return round(clamp(score, 0, 10), 1)


def generate_alzheimer_label(family_history, cognitive_score, memory_score, age):
    """
    Nhãn Alzheimer_Risk dựa trên 3 yếu tố chính:

    Nguy cơ CAO (label = 1) khi:
    - Có tiền sử gia đình VÀ cognitive thấp
    - Hoặc cognitive rất thấp (< 20) bất kể tiền sử
    - Hoặc memory rất thấp (< 4) + tuổi cao

    Thiết kế để đạt ~20–25% positive rate — balanced hơn nhiều.
    """
    risk_score = 0

    if family_history == "Yes":
        risk_score += 2

    if cognitive_score < 22:
        risk_score += 3
    elif cognitive_score < 25:
        risk_score += 1

    if memory_score < 4:
        risk_score += 2
    elif memory_score < 6:
        risk_score += 1

    if age > 50:
        risk_score += 1

    # Ngưỡng: risk >= 4 thì label = 1
    # Thêm xác suất nhỏ để tránh quá cứng nhắc
    if risk_score >= 4:
        return 1
    elif risk_score == 3:
        # 40% cơ hội label = 1 ở vùng biên
        return 1 if random.random() < 0.40 else 0
    else:
        return 0


# =========================================================
# ĐỌC CSV GỐC, SINH CỘT MỚI, GHI FILE MỚI
# =========================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(script_dir, INPUT_FILE)
output_path = os.path.join(script_dir, OUTPUT_FILE)

rows = list(csv.DictReader(open(input_path, encoding='utf-8')))

print(f"Đọc: {len(rows)} dòng từ {INPUT_FILE}")

for row in rows:
    age           = float(row['Age'])
    stress        = float(row['Stress Level'])
    sleep_quality = float(row['Quality of Sleep'])
    sleep_dur     = float(row['Sleep Duration'])

    fh  = generate_family_history()
    cog = generate_cognitive_score(age, stress, sleep_quality, sleep_dur)
    mem = generate_memory_test_score(cog)
    alz = generate_alzheimer_label(fh, cog, mem, age)

    row['Family_History_Alzheimer'] = fh
    row['Cognitive_Score']          = cog
    row['Memory_Test_Score']        = mem
    row['Alzheimer_Risk']           = alz

# Thống kê kiểm tra
total     = len(rows)
alz_pos   = sum(1 for r in rows if r['Alzheimer_Risk'] == 1)
fh_yes    = sum(1 for r in rows if r['Family_History_Alzheimer'] == 'Yes')
cog_vals  = [float(r['Cognitive_Score']) for r in rows]
mem_vals  = [float(r['Memory_Test_Score']) for r in rows]

print(f"\n=== THỐNG KÊ CỘT MỚI ===")
print(f"Family_History Yes : {fh_yes}/{total} ({fh_yes/total*100:.1f}%)")
print(f"Alzheimer_Risk = 1 : {alz_pos}/{total} ({alz_pos/total*100:.1f}%)")
print(f"Cognitive_Score    : min={min(cog_vals):.1f}  max={max(cog_vals):.1f}  avg={sum(cog_vals)/total:.1f}")
print(f"Memory_Test_Score  : min={min(mem_vals):.1f}  max={max(mem_vals):.1f}  avg={sum(mem_vals)/total:.1f}")

# Ghi file mới
new_fields = list(rows[0].keys())

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nĐã lưu: {OUTPUT_FILE}")
print(f"Tổng cột: {len(new_fields)} ({', '.join(new_fields[-4:])} được thêm mới)")