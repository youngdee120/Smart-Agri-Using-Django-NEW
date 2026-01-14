from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

User = get_user_model()

class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            "placeholder": "Enter email-address", 
            "class": "form-control"
        })
    )
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            "placeholder": "Enter username", 
            "class": "form-control"
        })
    )
    location = forms.CharField(
        widget=forms.TextInput(attrs={
            "placeholder": "Enter location", 
            "class": "form-control"
        })
    )
    password1 = forms.CharField(
        label="Password", 
        widget=forms.PasswordInput(attrs={
            "placeholder": "Enter password", 
            "class": "form-control"
        })
    )
    password2 = forms.CharField(
        label="Confirm Password", 
        widget=forms.PasswordInput(attrs={
            "placeholder": "Confirm password", 
            "class": "form-control"
        })
    )
    
    class Meta:
        model = User
        fields = ["email", "username","location", "password1", "password2"]

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already exists!")
        return email

class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your account email"
        })
    )

class ResetPasswordForm(forms.Form):
    password1 = forms.CharField(
        label="New Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Enter new password"
        })
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Confirm new password"
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get('password1')
        p2 = cleaned_data.get('password2')
        if p1 and p2 and p1 != p2:
            self.add_error('password2', "Passwords do not match!")
        return cleaned_data
    
class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your account email"
        })
    )

class ResetPasswordForm(forms.Form):
    password1 = forms.CharField(
        label="New Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Enter new password"
        })
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Confirm new password"
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get('password1')
        p2 = cleaned_data.get('password2')
        if p1 and p2 and p1 != p2:
            self.add_error('password2', "Passwords do not match!")
        return cleaned_data    