import os
from django.conf import settings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, PReLU, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, \
    BatchNormalization, LayerNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from medpy.io import load, header
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd

K.set_image_data_format('channels_first')


def dice_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    union = K.sum(y_true_f + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)


def unet_1(img_channels, image_rows, image_cols, neurons=16):
    inputs = Input((img_channels, image_rows, image_cols))

    conv1 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(neurons * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(neurons * 16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(neurons * 8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(neurons * 4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(neurons * 2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(neurons * 1, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    return model


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

    return image_with_contours


def process_images(image_paths):
    X = buffer_images(image_paths)

    w_size = np.array(Image.open(X[0][0])).shape[0]
    model = unet_1(1, w_size, w_size, neurons=8)
    model.load_weights('unet_r.h5')

    val_gen_params = {
        'x_col': 0,
        'target_size': (512, 512),
        'color_mode': 'grayscale',
        'batch_size': 1,
        'class_mode': None,
        'shuffle': False,
        'seed': 42,
    }

    idg_test_data = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    data_g = idg_test_data.flow_from_dataframe(X, **val_gen_params)

    save_dir = os.path.join(settings.MEDIA_ROOT, 'result')
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i, image in enumerate(data_g):
        if i >= len(image_paths):
            break

        image_path = X.iloc[i, 0]
        image_array = np.array(Image.open(image_path))
        image_array = image_array.astype(np.uint8)

        name = os.path.splitext(os.path.basename(image_paths[0]))[0]
        save_path_orig = os.path.join(save_dir, f'{name}.png')
        Image.fromarray(image_array).save(save_path_orig)

        predicted_mask = model.predict(image).astype(np.uint8).squeeze()
        image_with_contours = draw_contours(image_array, predicted_mask)

        save_path_seg = os.path.join(save_dir, f'processed_{name}.png')

        cv2.imwrite(save_path_seg, image_with_contours)

        processed_url = os.path.join(settings.MEDIA_URL, 'result', f"processed_{name}.png")
        original_url = os.path.join(settings.MEDIA_URL, 'result', f"{name}.png")


        results.append({
            'original': original_url,
            'processed': processed_url,
            'name': name
        })

    return results


def save_image(file):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.name)

    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return file_path
