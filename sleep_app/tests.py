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
        self.assertContains(response, "Dữ liệu CSV đã sẵn sàng để phân tích.")
        self.assertContains(response, "<table", html=False)
        self.assertContains(response, 'name="sleep_hours" value="7.4"', html=False)
        self.assertContains(response, 'name="user_name" value="Lan"', html=False)
        self.assertContains(response, "Phân tích dòng đầu tiên từ CSV")

    def test_upload_empty_csv_shows_error(self):
        upload = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")

        response = self.client.post(reverse("upload_page"), {"sleep_data_file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload chưa thành công.")
        self.assertContains(response, "Có lỗi khi đọc file")
