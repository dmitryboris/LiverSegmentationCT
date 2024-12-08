"""
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
"""
from . import views
from django.urls import path


app_name = 'image_processing'

urlpatterns = [
    path('', views.upload, name='upload'),
    path('result/', views.result, name='result'),
]
