from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth import get_user_model
from django.conf import settings
import secrets
# Create your models here.

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    
    USERNAME_FIELD = ("email")
    REQUIRED_FIELDS = ["username"]
    
    def _str__(self):
        return self.email
    

class OtpToken(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="otps")
    otp_code = models.CharField(max_length=6, default=secrets.token_hex(3))
    tp_created_at = models.DateTimeField(auto_now_add=True)
    otp_expires_at = models.DateTimeField(blank=True, null=True)
    
    
    def __str__(self):
        return self.user.username


class FertilizerPrediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    temperature = models.FloatField()
    humidity = models.FloatField()
    moisture = models.FloatField()
    soil_type = models.IntegerField()
    crop_type = models.IntegerField()
    nitrogen = models.FloatField()
    potassium = models.FloatField()
    phosphorus = models.FloatField()
    recommended_fertilizer = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"FertilizerPrediction #{self.pk} by {self.user.username}"

class CropPrediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    recommended_crop = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CropPrediction #{self.pk} by {self.user.username}"

class PestClassificationRecord(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    image = models.ImageField(upload_to='pest_images/')
    prediction = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Record #{self.pk} - {self.prediction}"