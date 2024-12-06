"""
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
"""
from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.home, name='home'),
]
