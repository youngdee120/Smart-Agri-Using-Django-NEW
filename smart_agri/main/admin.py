import csv
from django.http import HttpResponse
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from import_export import resources
from django.utils.html import mark_safe
from import_export.admin import ImportExportModelAdmin
from .models import CustomUser, OtpToken, FertilizerPrediction, CropPrediction, PestClassificationRecord

# Define Import-Export Resource Classes
class FertilizerPredictionResource(resources.ModelResource):
    class Meta:
        model = FertilizerPrediction

class CropPredictionResource(resources.ModelResource):
    class Meta:
        model = CropPrediction

# Custom User Admin
class CustomUserAdmin(UserAdmin):
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "location",
        "date_joined",
        "is_active",
        "is_staff",
    )
    fieldsets = UserAdmin.fieldsets + (
        (None, {"fields": ("location",)}),
    )

class OtpTokenAdmin(admin.ModelAdmin):
    list_display = ("user", "otp_code")

# Register FertilizerPrediction with ImportExportModelAdmin
@admin.register(FertilizerPrediction)
class FertilizerPredictionAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_class = FertilizerPredictionResource
    list_display = (
        "id",
        "user",
        "temperature",
        "humidity",
        "moisture",
        "soil_type",
        "crop_type",
        "nitrogen",
        "potassium",
        "phosphorus",
        "recommended_fertilizer",
        "created_at",
    )

# Register CropPrediction with ImportExportModelAdmin
@admin.register(CropPrediction)
class CropPredictionAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_class = CropPredictionResource
    list_display = (
        "id",
        "user",
        "nitrogen",
        "phosphorus",
        "potassium",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "recommended_crop",
        "created_at",
    )

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(OtpToken, OtpTokenAdmin)

@admin.register(PestClassificationRecord)
class PestClassificationRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'image_thumbnail', 'prediction', 'created_at')

    def image_thumbnail(self, obj):
        """
        Returns an HTML <img> tag to display a small thumbnail of the image.
        """
        if obj.image:
            return mark_safe(f'<img src="{obj.image.url}" style="max-height: 80px;" />')
        return "No Image"

    image_thumbnail.short_description = 'Image'
