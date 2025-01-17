"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='main.html'), name='main'),  # 기존 home.html
    path('home/', TemplateView.as_view(template_name='home.html'), name='home'),  # 기존 home.html
    path('intro1/', TemplateView.as_view(template_name='introkid.html'), name='intro1'),  # 추가된 introkid.html
    path('intro2/', TemplateView.as_view(template_name='introadult.html'), name='intro2'),
    path('chat1/', TemplateView.as_view(template_name='kidchat.html'), name='chat1'), 
    path('chat2/', TemplateView.as_view(template_name='adultchat.html'), name='chat2'),
    path('api/', include('api.urls')),
]
# http://127.0.0.1:8000
