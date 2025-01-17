from django.urls import path
from .views import adchat_message, chchat_message

urlpatterns = [
    path('adchat_message/<str:message>/', adchat_message, name='adchat_message'),
    path('chchat_message/<str:message>/', chchat_message, name='chchat_message'),
]
# api/chat_message/안녕하세요.