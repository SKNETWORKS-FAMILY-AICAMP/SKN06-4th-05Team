from django.urls import path
from .views import adchat_message, chchat_message

urlpatterns = [
    path('adchat_message/', adchat_message, name='adchat_message'),
    path('chchat_message/', chchat_message, name='chchat_message'),
]