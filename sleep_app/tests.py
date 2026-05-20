from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(ALLOWED_HOSTS=["testserver", "localhost"])
class UploadViewTests(TestCase):
    def test_upload_page_shows_file_form(self):
        response = self.client.get(reverse("upload_page"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'enctype="multipart/form-data"', html=False)
        self.assertContains(response, 'type="file"', html=False)
        self.assertContains(response, 'name="sleep_data_file"', html=False)

    def test_upload_csv_shows_preview_and_analysis_form(self):
        csv_content = "\n".join(
            [
                "user_name,sleep_hours,deep_sleep_minutes,rem_sleep_minutes,awakenings,spo2_drop_events,hrv,rhr,consecutive_days,target_sleep_hours,wake_time",
                "Lan,7.4,95,110,2,1,56,58,3,8.0,06:15",
            ]
        )
        upload = SimpleUploadedFile("sleep.csv", csv_content.encode(), content_type="text/csv")

        response = self.client.post(reverse("upload_page"), {"sleep_data_file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dữ liệu sẵn sàng phân tích")
        self.assertContains(response, "<table", html=False)
        self.assertContains(response, 'name="sleep_hours" value="7.4"', html=False)
        self.assertContains(response, 'name="user_name" value="Lan"', html=False)
        self.assertContains(response, "Phân tích dòng đầu tiên từ CSV")

    def test_upload_empty_csv_shows_error(self):
        upload = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")

        response = self.client.post(reverse("upload_page"), {"sleep_data_file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload chưa thành công")
        self.assertContains(response, "Lỗi xử lý file")

    def test_dashboard_view_displays_charts_and_history(self):
        from sleep_app.models import SleepRecord
        record = SleepRecord.objects.create(
            user_name="Lan",
            sleep_hours=7.4,
            deep_sleep_minutes=95,
            rem_sleep_minutes=110,
            awakenings=2,
            spo2_drop_events=1,
            hrv=56,
            rhr=58,
            consecutive_days=3,
            target_sleep_hours=8.0,
            wake_time="06:15",
            quality_label="Tốt",
            quality_tone="good",
            recovery_score=85,
            algorithm_used="KNN"
        )
        
        response = self.client.get(reverse("dashboard_page"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Báo cáo của Lan")
        self.assertContains(response, "Lan")
        self.assertContains(response, "7.4 giờ")
        
        response_id = self.client.get(reverse("dashboard_page") + f"?id={record.id}")
        self.assertEqual(response_id.status_code, 200)
        self.assertContains(response_id, "Báo cáo của Lan")
