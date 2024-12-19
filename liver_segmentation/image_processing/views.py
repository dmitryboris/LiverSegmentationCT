import base64
import os
import json
import zipfile
from io import BytesIO
from urllib.parse import urlparse

from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from .processing import save_image, process_images
from .forms import DCMFileUploadForm
from .models import LiverImage, SegmentationResult


def upload(request):
    if request.method == 'POST':
        form = DCMFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = form.cleaned_data['dcm_files']

            paths = []
            for file in files:
                file_path = save_image(file)
                paths.append(file_path)

            result = process_images(paths)

            return redirect(reverse('image_processing:result', kwargs={'pk': result.id}))
    else:
        form = DCMFileUploadForm()
    return render(request, 'image_processing/upload.html', {'form': form})


def show_result(request, pk):
    result = get_object_or_404(SegmentationResult, pk=pk)
    images = result.images.all()

    return render(request, 'image_processing/result.html', {'result': result, 'images': images})


def show_and_edit_image(request, pk):
    image = get_object_or_404(LiverImage, pk=pk)
    return render(request, 'image_processing/show_image.html', {'image': image})


@csrf_exempt
def save_edited_image(request, pk):
    if request.method == 'POST':
        liver_image = get_object_or_404(LiverImage, pk=pk)

        data = json.loads(request.body)
        img_data = data.get('image')
        format, imgstr = img_data.split(';base64,')
        ext = format.split('/')[-1]

        save_dir_url = os.path.join(settings.MEDIA_URL, 'result/processed/')
        save_dir_path = os.path.join(settings.MEDIA_ROOT, 'result/processed/')
        filename = f'processed_{pk}.{ext}'
        new_save_path = os.path.join(save_dir_path, filename)
        new_processed_url = os.path.join(save_dir_url, filename)

        img_data_decoded = base64.b64decode(imgstr)
        with open(new_save_path, 'wb') as file:
            file.write(img_data_decoded)

        liver_image.processed_image = f'result/processed/{filename}'

        liver_image.save()
        result = liver_image.segmentation_results.first()
        if result:
            redirect_url = reverse('image_processing:result', kwargs={'pk': result.id})
            return JsonResponse({'status': 'success', 'redirect_url': redirect_url})
        return JsonResponse({'status': 'error', 'message': 'Segmentation result not found'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


def save_processed_images(request, pk):
    result = get_object_or_404(SegmentationResult, pk=pk)
    images = result.images.all()

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for image in images:
            absolute_path = os.path.join(settings.MEDIA_ROOT, image.processed_image.path)
            zip_file.write(absolute_path, os.path.basename(absolute_path))


    zip_buffer.seek(0)
    response = HttpResponse(zip_buffer, content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename=segmentation_result_{pk}_processed_images.zip'

    return response
