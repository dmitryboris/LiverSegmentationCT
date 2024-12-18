import base64

from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from .processing import save_image, process_images
from .forms import DCMFileUploadForm
from .models import LiverImage


def upload(request):
    if request.method == 'POST':
        form = DCMFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = form.cleaned_data['dcm_files']

            paths = []
            for file in files:
                file_path = save_image(file)
                paths.append(file_path)

            request.session['original_images'] = paths

            return redirect('image_processing:result')
    else:
        form = DCMFileUploadForm()
    return render(request, 'image_processing/upload.html', {'form': form})


def result(request):
    image_paths = request.session.get('original_images')
    results = process_images(image_paths)

    return render(request, 'image_processing/result.html', {
        'results': results
    })


def show_and_edit_image(request, pk):
    image = get_object_or_404(LiverImage, pk=pk)
    return render(request, 'image_processing/show_image.html', {'image': image})


@csrf_exempt
def save_edited_image(request, pk):
    if request.method == 'POST':
        contour = get_object_or_404(LiverImage, pk=pk)
        data = request.POST.get('image')
        image_data = data.split(",")[1]  # Убираем "data:image/png;base64,"
        """contour.edited_mask.save(
            f"edited_{pk}.png",
            ContentFile(base64.b64decode(image_data))
        )"""
        return JsonResponse({'status': 'success'})