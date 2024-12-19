import json
import os
from django.conf import settings
from .models import LiverImage, SegmentationResult
from .create_model import unet_1

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from medpy.io import load, header
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import uuid


def buffer_images(filenames):
    buffer_dir = os.path.join(settings.MEDIA_ROOT, 'buffer')
    os.makedirs(buffer_dir, exist_ok=True)

    paths = []
    for filename in filenames:
        safe_name = os.path.basename(filename)
        safe_name = safe_name.replace('/', '_').replace('\\', '_') + '.tiff'

        img, header = load(filename)
        pil = Image.fromarray(img.squeeze())

        save_path = os.path.join(buffer_dir, safe_name)
        pil.save(save_path, 'TIFF', compression='none')
        paths.append(save_path)

    return pd.DataFrame(paths)


def draw_contours(image_array, predicted_mask):
    pred_8uc1 = (predicted_mask.squeeze() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(pred_8uc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)

    return image_with_contours, contours


def process_images(image_paths):
    X = buffer_images(image_paths)

    w_size = np.array(Image.open(X[0][0])).shape[0]
    model = unet_1(1, w_size, w_size, neurons=8)
    model.load_weights('unet_r.h5')

    val_gen_params = {
        'x_col': 0,
        'target_size': (512, 512),
        'color_mode': 'grayscale',
        'batch_size': len(image_paths),
        'class_mode': None,
        'shuffle': False,
        'seed': 42,
    }

    idg_test_data = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    data_g = idg_test_data.flow_from_dataframe(X, **val_gen_params)

    batch_images = next(data_g)
    predicted_masks = model.predict(batch_images).astype(np.uint8).squeeze()

    save_dir_orig = os.path.join(settings.MEDIA_ROOT, 'result/original/')
    save_dir_proc = os.path.join(settings.MEDIA_ROOT, 'result/processed/')
    os.makedirs(save_dir_orig, exist_ok=True)
    os.makedirs(save_dir_proc, exist_ok=True)

    segmentation_result = SegmentationResult.objects.create()
    for i, (image, predicted_mask) in enumerate(zip(batch_images, predicted_masks)):
        image_path = X.iloc[i, 0]
        image_array = np.array(Image.open(image_path)).astype(np.uint8)

        name = os.path.splitext(os.path.basename(image_path))[0]
        random_prefix = uuid.uuid4()
        save_path_orig = os.path.join(save_dir_orig, f'{random_prefix}_{name}.png')
        Image.fromarray(image_array).save(save_path_orig)

        image_with_contours, contours = draw_contours(image_array, predicted_mask)

        save_path_proc = os.path.join(save_dir_proc, f'{random_prefix}_{name}.png')
        cv2.imwrite(save_path_proc, image_with_contours)

        processed_url = os.path.join(settings.MEDIA_URL, 'result/processed/', f'{random_prefix}_{name}.png')
        original_url = os.path.join(settings.MEDIA_URL, 'result/original/', f'{random_prefix}_{name}.png')

        # ускорить?
        contours_result = []
        for contour in contours:
            for point in contour.tolist():
                contours_result.append(point[0])

        contours_json = json.dumps(contours_result)

        liver_image = LiverImage.objects.create(
            title=name,
            original_url=original_url,
            processed_url=processed_url,
            contours=contours_json,
        )

        segmentation_result.images.add(liver_image)

    return segmentation_result


def save_image(file):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.name)

    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return file_path
