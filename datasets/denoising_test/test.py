import tensorflow as tf
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Load the test image
test_img = load_img('SEM/pollen.spd3.0098.png', target_size=(256, 256))
test_img_array = img_to_array(test_img) / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)

# Assuming you have already loaded the model
model = tf.keras.models.load_model('models/unet_denoising_gamma_50_model.h5')
denoised_img_array = model.predict(test_img_array)

# Convert the denoised image array to an image and save it
denoised_img = array_to_img(denoised_img_array[0] * 255.0, scale=False)
denoised_img.save('denoised_pollen_image.png')

# Calculate PSNR and SSIM
test_img_array = np.squeeze(test_img_array, axis=0)
denoised_img_array = np.squeeze(denoised_img_array, axis=0)
psnr = compare_psnr(test_img_array, denoised_img_array)
ssim = compare_ssim(test_img_array, denoised_img_array, multichannel=True)

print(f"PSNR: {psnr}, SSIM: {ssim}")

