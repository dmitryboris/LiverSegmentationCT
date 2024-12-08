import os

from django.conf import settings
from django.shortcuts import render, redirect
from .processing import save_image, process_image
from . import forms


def upload(request):
    if request.method == 'POST':
        form = forms.DCMFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['dcm_file']

            file_path = save_image(uploaded_file)

            request.session['original_image'] = file_path

            return redirect('image_processing:result')
    else:
        form = forms.DCMFileUploadForm()
    return render(request, 'image_processing/upload.html', {'form': form})


def result(request):
    image_path = request.session.get('original_image')
    overlay_url = process_image(image_path)

    return render(request, 'image_processing/result.html', {
        'overlay': overlay_url
    })

