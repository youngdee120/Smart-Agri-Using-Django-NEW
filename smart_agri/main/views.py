import json
import os
import io
from django.http import HttpResponse
import joblib
import pickle
import numpy as np
import tensorflow as tf
from django.conf import settings
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from django.db.models import Count
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.contrib.auth import get_user_model, authenticate, login, logout
from django.contrib import messages
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.auth.tokens import default_token_generator

from .models import CropPrediction, FertilizerPrediction, OtpToken, PestClassificationRecord
from .forms import (
    RegisterForm,
    ForgotPasswordForm,
    ResetPasswordForm
)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

User = get_user_model()

# -----------------------------
#
# Load ML Models (Adjust Paths)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

# Example filenames -- adjust to match your actual files
crop_model = joblib.load(os.path.join(ML_MODELS_DIR, 'crop_model.pkl'))
stand_scaler = pickle.load(open(os.path.join(ML_MODELS_DIR, 'standscaler.pkl'), 'rb'))
minmax_scaler = pickle.load(open(os.path.join(ML_MODELS_DIR, 'minmaxscaler.pkl'), 'rb'))

fertilizer_model = pickle.load(open(os.path.join(ML_MODELS_DIR, 'classifier.pkl'), 'rb'))
ferti = pickle.load(open(os.path.join(ML_MODELS_DIR, 'fertilizer.pkl'), 'rb'))

# Load the pest classification model
MODEL_PATH = os.path.join(ML_MODELS_DIR, 'pests_classification_model.h5')
pest_model = tf.keras.models.load_model(MODEL_PATH)

# Manually define the label mapping (adjust if your training used a different index order)
pest_class_mapping = {
    0: "ants",
    1: "bees",
    2: "beetle",
    3: "catterpillar",
    4: "earthworms",
    5: "earwig",
    6: "grasshopper",
    7: "moth",
    8: "slug",
    9: "snail",
    10: "wasp",
    11: "weevil"
}


# -----------------------------
# Authentication + OTP Views
# -----------------------------
def home(request):
    """Renders the main login page."""
    return render(request, 'login.html')

def signup(request):
    """Sign up a new user with is_active=False, then send OTP."""
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            # Optionally create OTP here if not done via signals
            messages.success(request, "Account created successfully! An OTP was sent to your Email.")
            return redirect("verify-email", username=user.username)
    return render(request, "login.html", {"form": form})

def verify_email(request, username):
    """Verify OTP and activate the user if valid and not expired."""
    user = User.objects.get(username=username)
    user_otp = OtpToken.objects.filter(user=user).last()
    
    if request.method == 'POST':
        if user_otp and user_otp.otp_code == request.POST['otp_code']:
            if user_otp.otp_expires_at > timezone.now():
                user.is_active = True
                user.save()
                messages.success(request, "Account activated successfully! You can now log in.")
                return redirect("signin")
            else:
                messages.warning(request, "The OTP has expired, get a new OTP!")
                return redirect("verify-email", username=user.username)
        else:
            messages.warning(request, "Invalid OTP entered, enter a valid OTP!")
            return redirect("verify-email", username=user.username)
        
    return render(request, "verify_token.html")

def resend_otp(request):
    """Resend a new OTP to the user's email."""
    if request.method == 'POST':
        user_email = request.POST.get("otp_email")
        if User.objects.filter(email=user_email).exists():
            user = User.objects.get(email=user_email)
            otp = OtpToken.objects.create(
                user=user,
                otp_expires_at=timezone.now() + timezone.timedelta(minutes=5)
            )
            subject = "Email Verification"
            message = f"""
                Hi {user.username}, here is your OTP {otp.otp_code}.
                It expires in 5 minutes.
                Use this link to verify: http://127.0.0.1:8000/verify-email/{user.username}
            """
            sender = "deey440@gmail.com"
            receiver = [user.email]

            send_mail(subject, message, sender, receiver, fail_silently=False)
            messages.success(request, "A new OTP has been sent to your email address.")
            return redirect("verify-email", username=user.username)
        else:
            messages.warning(request, "This email doesn't exist in the database.")
            return redirect("resend-otp")
    
    return render(request, "resend_otp.html")

def signin(request):
    """Sign in an existing user, ensuring they are active (verified)."""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # 1. Attempt to find user by email
        try:
            user_obj = User.objects.get(email=email)
            if not user_obj.is_active:
                messages.warning(request, "Your account is not verified. Please verify your email.")
                return redirect("verify-email", username=user_obj.username)
        except User.DoesNotExist:
            pass  # We'll let authenticate handle the error below
        
        # 2. If user is active or doesn't exist, do normal authentication
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Hi {user.username}, you are now logged in.")
            return redirect("dashboard")
        else:
            messages.warning(request, "Invalid credentials.")
            return redirect("signin")
    return render(request, "login.html")

def dashboard(request):
    """Example post-login dashboard."""
    if not request.user.is_authenticated:
        return redirect('home')
    return render(request, 'landing.html', {'user': request.user})

def logout_view(request):
    """Logs out the user and redirects to home."""
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('home')

# -----------------------------
# Password Reset Views
# -----------------------------
def forgot_password(request):
    """Shows form for user email and sends a password-reset link."""
    if request.method == "POST":
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))

                reset_url = request.build_absolute_uri(f"/reset-password/{uid}/{token}/")
                subject = "Password Reset Request"
                message = f"""
                    Hi {user.username},
                    You requested a password reset. Click the link below to set a new password:
                    {reset_url}

                    If you did not request this, please ignore.
                """
                from_email = "noreply@yourdomain.com"
                recipient_list = [user.email]

                send_mail(subject, message, from_email, recipient_list, fail_silently=False)
                messages.success(request, "A password reset link has been sent to your email.")
                return redirect("signin")
            except User.DoesNotExist:
                messages.error(request, "No account found with that email.")
                return redirect("forgot_password")
    else:
        form = ForgotPasswordForm()
    return render(request, "forgot_password.html", {"form": form})

def reset_password(request, uidb64, token):
    """Validates the token and allows the user to set a new password."""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (ValueError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = ResetPasswordForm(request.POST)
            if form.is_valid():
                new_password = form.cleaned_data['password1']
                user.set_password(new_password)
                user.save()
                messages.success(request, "Your password has been reset successfully.")
                return redirect("signin")
        else:
            form = ResetPasswordForm()
        return render(request, "reset_password.html", {"form": form})
    else:
        messages.error(request, "This link is invalid or has expired.")
        return redirect("forgot_password")

# -----------------------------
# Additional Views
# -----------------------------


def model1_view(request):
    """Renders 'model1.html' (the fertilizer recommendation form)."""
    return render(request, 'model1.html')

def predict_view(request):
    """
    Handles POST data from 'model1.html' to predict fertilizer recommendation
    and saves the record to the database.
    """
    if request.method == 'POST':
        temp = request.POST.get('temp')
        humi = request.POST.get('humid')
        mois = request.POST.get('mois')
        soil = request.POST.get('soil')
        crop = request.POST.get('crop')
        nitro = request.POST.get('nitro')
        pota = request.POST.get('pota')
        phos = request.POST.get('phos')

        # Basic validation
        fields = [temp, humi, mois, soil, crop, nitro, pota, phos]
        if any(f is None or not f.isdigit() for f in fields):
            return render(request, 'model1.html', {'x': 'Invalid input. Please provide numeric values.'})

        input_list = list(map(int, fields))

        # ML model prediction
        pred_index = fertilizer_model.predict([input_list])[0]
        recommendation = ferti.classes_[pred_index]

        # Save record in the DB if user is authenticated
        if request.user.is_authenticated:
            from .models import FertilizerPrediction
            FertilizerPrediction.objects.create(
                user=request.user,
                temperature=int(temp),
                humidity=int(humi),
                moisture=int(mois),
                soil_type=int(soil),
                crop_type=int(crop),
                nitrogen=int(nitro),
                potassium=int(pota),
                phosphorus=int(phos),
                recommended_fertilizer=recommendation
            )

        return render(request, 'model1.html', {'x': recommendation})

    return redirect('model1')


def crops_view(request):
    """
    Renders 'crops.html' (the crop recommendation page).
    If POST, runs the crop model logic and saves the record.
    """
    if not request.user.is_authenticated:
        return redirect('home')

    result = None
    if request.method == 'POST':
        try:
            N = float(request.POST.get('Nitrogen'))
            P = float(request.POST.get('Phosphorus'))
            K = float(request.POST.get('Potassium'))
            temp = float(request.POST.get('Temperature'))
            humidity = float(request.POST.get('Humidity'))
            ph_val = float(request.POST.get('Ph'))
            rainfall = float(request.POST.get('Rainfall'))

            feature_list = [N, P, K, temp, humidity, ph_val, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale features
            scaled_features = minmax_scaler.transform(single_pred)
            final_features = stand_scaler.transform(scaled_features)

            prediction = crop_model.predict(final_features)

            # Map numeric label to crop name
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
                11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
                15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }
            crop_name = crop_dict.get(prediction[0], "Unknown")
            result = f"{crop_name} is the best crop to be cultivated right there."

            # Save record in the DB
            if request.user.is_authenticated:
                from .models import CropPrediction
                CropPrediction.objects.create(
                    user=request.user,
                    nitrogen=N,
                    phosphorus=P,
                    potassium=K,
                    temperature=temp,
                    humidity=humidity,
                    ph=ph_val,
                    rainfall=rainfall,
                    recommended_crop=crop_name
                )

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'crops.html', {'result': result})


def landing_view(request):
    """Renders the main landing page (e.g. 'landing.html')."""
    return render(request, 'landing.html')

@login_required
def profile_view(request):
    return render(request, 'profile.html')

@login_required
def change_password_view(request):
    if request.method == 'POST':
        # Implement logic for changing password, e.g. using Django's PasswordChangeForm
        # or your own custom logic
        messages.success(request, "Password updated successfully!")
        return redirect('profile_view')
    return render(request, 'change_password.html')

@login_required
def update_profile_view(request):
    if request.method == 'POST':
        user = request.user
        # Update user fields from the form input
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.location = request.POST.get('location', user.location)
        user.save()  # Save the updated user instance
        messages.success(request, "Profile updated successfully!")
        return redirect('profile_view')
    return render(request, 'update_profile.html')
def blog_view(request):
    # You can show a list of blog posts here
    return render(request, 'blog.html')

def pest_classification_view(request):
    prediction = None
    record = None

    if request.method == 'POST' and request.FILES.get('pest_image'):
        uploaded_image = request.FILES['pest_image']

        # Create a record to store the image and associate with user if logged in
        record = PestClassificationRecord.objects.create(
            user=request.user if request.user.is_authenticated else None,
            image=uploaded_image
        )

        # Now use record.image.path to load the file from disk
        img = tf.keras.preprocessing.image.load_img(
            record.image.path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        # Predict with the loaded model
        preds = pest_model.predict(img_array)
        pred_class_index = np.argmax(preds, axis=1)[0]

        # Map the predicted index to a pest label
        prediction = pest_class_mapping.get(pred_class_index, "Unknown")

        # Update the record with the prediction
        record.prediction = prediction
        record.save()

    # Render the same page; pass both record and prediction
    return render(request, 'pest_classification.html', {
        'prediction': prediction,
        'record': record
    })

@login_required
def reports_view(request):
    """
    Render the reports page with summary data for fertilizer, crop, and pest predictions.
    Staff users see aggregated data; regular users see only their own data.
    """
    if request.user.is_staff:
        fertilizer_counts = list(
            FertilizerPrediction.objects
            .values('recommended_fertilizer')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
        crop_counts = list(
            CropPrediction.objects
            .values('recommended_crop')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
        pest_counts = list(
            PestClassificationRecord.objects
            .values('prediction')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
    else:
        fertilizer_counts = list(
            FertilizerPrediction.objects
            .filter(user=request.user)
            .values('recommended_fertilizer')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
        crop_counts = list(
            CropPrediction.objects
            .filter(user=request.user)
            .values('recommended_crop')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
        pest_counts = list(
            PestClassificationRecord.objects
            .filter(user=request.user)
            .values('prediction')
            .annotate(total=Count('id'))
            .order_by('-total')
        )
    
    context = {
        'fertilizer_counts': fertilizer_counts,
        'crop_counts': crop_counts,
        'pest_counts': pest_counts,
    }
    return render(request, 'reports.html', context)


@login_required
def download_report(request):
    """
    Generate a PDF report using ReportLab.
    For staff users, show aggregated data from all users.
    For regular users, show only their own data.
    """
    if request.user.is_staff:
        fertilizer_counts = list(FertilizerPrediction.objects
                             .values('recommended_fertilizer')
                             .annotate(total=Count('id'))
                             .order_by('-total'))
        crop_counts = list(CropPrediction.objects
                       .values('recommended_crop')
                       .annotate(total=Count('id'))
                       .order_by('-total'))
        pest_counts = list(PestClassificationRecord.objects
                       .values('prediction')
                       .annotate(total=Count('id'))
                       .order_by('-total'))
    else:
        fertilizer_counts = list(FertilizerPrediction.objects
                             .filter(user=request.user)
                             .values('recommended_fertilizer')
                             .annotate(total=Count('id'))
                             .order_by('-total'))
        crop_counts = list(CropPrediction.objects
                       .filter(user=request.user)
                       .values('recommended_crop')
                       .annotate(total=Count('id'))
                       .order_by('-total'))
        pest_counts = list(PestClassificationRecord.objects
                       .filter(user=request.user)
                       .values('prediction')
                       .annotate(total=Count('id'))
                       .order_by('-total'))
    
    # Create a PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Reports", styles['Title']))
    elements.append(Spacer(1, 12))
    
    def create_section(title_text, table_header, data, key):
        elements.append(Paragraph(title_text, styles['Heading2']))
        elements.append(Spacer(1, 12))
        # Build table data: header row followed by each data row
        table_data = [[table_header, "Count"]]
        for item in data:
            table_data.append([item.get(key, "N/A"), item.get("total", 0)])
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('TOPPADDING', (0,0), (-1,0), 8),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))
    
    # Add sections
    create_section("Fertilizer Recommendations", "Fertilizer", fertilizer_counts, 'recommended_fertilizer')
    create_section("Crop Recommendations", "Crop", crop_counts, 'recommended_crop')
    create_section("Pest Classifications", "Pest", pest_counts, 'prediction')
    
    # Build PDF document
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="reports.pdf"'
    response.write(pdf)
    return response