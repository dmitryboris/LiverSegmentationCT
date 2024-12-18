from django.db import models


class LiverImage(models.Model):
    id = models.AutoField(primary_key=True)

    title = models.CharField(max_length=255)
    original_url = models.ImageField(upload_to='result/original/', verbose_name='Оригинальное изображение')
    processed_url = models.ImageField(upload_to='result/processed/', verbose_name='Обработанное изображение')
    contours = models.JSONField(default=list)

    def __str__(self):
        return f"Image {self.id} - {self.title}"
