import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import sys

noise_type = str(sys.argv[1])
noise_level = sys.argv[2]

# Define the U-Net architecture with residual connections
def unet_res(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    resid1 = add([inputs, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(resid1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    resid2 = add([pool1, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(resid2)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    resid5 = add([pool4, conv5])

    up6 = UpSampling2D(size=(2, 2))(resid5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([resid4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    resid6 = add([resid4, conv6])

    up7 = UpSampling2D(size=(2, 2))(resid6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([resid3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    resid7 = add([resid3, conv7])

    up8 = UpSampling2D(size=(2, 2))(resid7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([resid2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    resid8 = add([resid2, conv8])

    up9 = UpSampling2D(size=(2, 2))(resid8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([resid1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    resid9 = add([resid1, conv9])

    conv10 = Conv2D(3, 1, activation='linear')(resid9)

    model = Model(inputs=inputs, outputs=conv10)


    return model

# Load data and preprocess
def load_data(data_directory, img_size=(256, 256)):
    clean_images = []
    noisy_images = []
    clean_directory = os.path.join(data_directory, 'clean'+'/'+str(noise_level))
    noisy_directory = os.path.join(data_directory, 'noisy'+'/'+str(noise_level))
    for img_file in os.listdir(clean_directory):
        clean_img = load_img(os.path.join(clean_directory, img_file), color_mode="grayscale", target_size=img_size)
        clean_img_array = img_to_array(clean_img) / 255.0
        clean_images.append(clean_img_array)

        noisy_img = load_img(os.path.join(noisy_directory, img_file), color_mode="grayscale", target_size=img_size)
        noisy_img_array = img_to_array(noisy_img) / 255.0
        noisy_images.append(noisy_img_array)

    return np.array(clean_images), np.array(noisy_images)

# Load your dataset
data_directory = 'dataset_denoise/'+str(noise_type)
clean_images, noisy_images = load_data(data_directory)



# Split the dataset into training, testing, and validation sets
train_clean, temp_clean, train_noisy, temp_noisy = train_test_split(clean_images, noisy_images, test_size=0.1, random_state=42)
test_clean, val_clean, test_noisy, val_noisy = train_test_split(temp_clean, temp_noisy, test_size=0.5, random_state=42)

# Define the model
model = unet_res()
model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
checkpoint = ModelCheckpoint('unet_res_denoising_'+str(noise_type)+'_'+str(noise_level)+'_weights.h5', monitor='val_loss', save_best_only=True)
callbacks_list = [checkpoint]
model.fit(train_noisy, train_clean, batch_size=32, epochs=20, verbose=1, validation_data=(val_noisy, val_clean), callbacks=callbacks_list)

# Denoise the test data
denoised_images = model.predict(test_noisy)

# Evaluate denoised images using PSNR and SSIM
psnr_values = []
ssim_values = []

for i in range(len(test_clean)):
    psnr = compare_psnr(test_clean[i], denoised_images[i])
    ssim = compare_ssim(test_clean[i], denoised_images[i])
    # Convert the denoised image array to an image and save it
    denoised_img = array_to_img(denoised_images[i] * 255.0, scale=False)
    denoised_img.save('den/'+str(noise_type)+'/'+str(noise_level)+'/denoised/denoised_res_image_'+str(i)+'.png')
    cn_img = array_to_img(test_clean[i] * 255.0, scale=False)
    cn_img.save('den/'+str(noise_type)+'/'+str(noise_level)+'/clean/clean_res_image_'+str(i)+'.png')
    psnr_values.append(psnr)
    ssim_values.append(ssim)

# Compute average PSNR and SSIM
avg_psnr = sum(psnr_values) / len(psnr_values)
avg_ssim = sum(ssim_values) / len(ssim_values)

print(f'Average PSNR: {avg_psnr}')
print(f'Average SSIM: {avg_ssim}')

# Save the model
model.save('models/unet_res_denoising_'+str(noise_type)+'_'+str(noise_level)+'_model.h5')

