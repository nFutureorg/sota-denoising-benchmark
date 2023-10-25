import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import sys


# Test the model on test data
test_img = load_img('SEM/pollen.spd3.0098.png', target_size=(256, 256))
test_img_array = img_to_array(test_img) / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)

# Assuming you have already loaded the model
model = tf.keras.models.load_model('models/unet_denoising_gamma_50_model.h5')
denoised_img_array = model.predict(test_img_array)

# Convert the denoised image array to an image and save it
denoised_img = array_to_img(denoised_img_array[0] * 255.0, scale=False)
denoised_img.save('denoised_pollen_image.png')

psnr = compare_psnr(test_img, denoised_img)
ssim = compare_ssim(test_img, denoised_img,channel_axis=2)

print(f'PSNR: {psnr}')
print(f'SSIM: {ssim}')
