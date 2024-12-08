from django.shortcuts import render, redirect
from .processing import save_image, process_images
from . import forms


def upload(request):
    if request.method == 'POST':
        form = forms.DCMFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = form.cleaned_data['dcm_files']

            paths = []
            for file in files:
                file_path = save_image(file)
                paths.append(file_path)

            request.session['original_images'] = paths

            return redirect('image_processing:result')
    else:
        form = forms.DCMFileUploadForm()
    return render(request, 'image_processing/upload.html', {'form': form})


def result(request):
    image_paths = request.session.get('original_images')
    results = process_images(image_paths)

    return render(request, 'image_processing/result.html', {
        'results': results
    })


