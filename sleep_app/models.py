from django.db import models


class SleepRecord(models.Model):
    """Lưu trữ dữ liệu giấc ngủ của người dùng"""
    user_name = models.CharField(max_length=255, default="Anonymous")
    sleep_hours = models.FloatField()
    deep_sleep_minutes = models.IntegerField()
    rem_sleep_minutes = models.IntegerField()
    awakenings = models.IntegerField()
    spo2_drop_events = models.IntegerField()
    hrv = models.IntegerField()
    rhr = models.IntegerField()
    consecutive_days = models.IntegerField(default=1)
    target_sleep_hours = models.FloatField(default=8.0)
    wake_time = models.TimeField(default="06:30")
    
    # Kết quả phân tích
    quality_label = models.CharField(max_length=50, blank=True)
    quality_tone = models.CharField(max_length=50, blank=True)
    recovery_score = models.IntegerField(default=0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    algorithm_used = models.CharField(max_length=100, blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Sleep Records"
    
    def __str__(self):
        return f"{self.user_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class CSVUploadHistory(models.Model):
    """Lưu lịch sử upload file CSV"""
    file_name = models.CharField(max_length=255)
    rows_count = models.IntegerField()
    columns_count = models.IntegerField()
    processed_rows = models.IntegerField(default=0)
    upload_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Chờ xử lý'),
            ('processing', 'Đang xử lý'),
            ('completed', 'Hoàn thành'),
            ('failed', 'Lỗi'),
        ],
        default='pending'
    )
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.file_name} - {self.upload_status}"
