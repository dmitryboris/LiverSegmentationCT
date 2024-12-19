"""
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
"""
from . import views
from django.urls import path


app_name = 'image_processing'

urlpatterns = [
    path('', views.upload, name='upload'),
    path('result/<int:pk>/', views.show_result, name='result'),
    path('image/<int:pk>/', views.show_and_edit_image, name='show_image'),
    path('image/save-edited/<int:pk>/', views.save_edited_image, name='save_edited_image'),
]
