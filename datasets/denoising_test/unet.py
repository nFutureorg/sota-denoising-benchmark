%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
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
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Flatten, Dense, Input, Add
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam

import sys
import dataset_generate
import pathlib
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import cv2
import skimage
from skimage.util import random_noise
import tensorflow as tf

from tqdm.notebook import tqdm
import random


noise_type = str(sys.argv[1])
noise_level = sys.argv[2]


def _up_down_flip(image, label):
    image = tf.image.flip_up_down(image)
    label = tf.image.flip_up_down(label)
    return image, label

def _left_right_flip(image, label):
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
    return image, label

def _rotate(image, label):
    random_angle = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, random_angle)
    label = tf.image.rot90(label, random_angle)
    return image, label

def _hue(image, label):
    rand_value = random.uniform(-1,1)
    image = tf.image.adjust_hue(image, rand_value)
    label = tf.image.adjust_hue(label, rand_value)
    return image, label

def _brightness(image, label):
    rand_value = random.uniform(-0.08,0.25)
    image = tf.image.adjust_brightness(image, rand_value)
    label = tf.image.adjust_brightness(label, rand_value)
    return image, label

def _saturation(image, label):
    rand_value = random.uniform(1, 5)
    image = tf.image.adjust_saturation(image, rand_value)
    label = tf.image.adjust_saturation(label, rand_value)
    return image, label

def _contrast(image, label):
    rand_value = random.uniform(1, 3)
    image = tf.image.adjust_contrast(image, rand_value)
    label = tf.image.adjust_contrast(label, rand_value)
    return image, label

# What does batch, repeat, and shuffle do with TensorFlow Dataset?
# https://stackoverflow.com/q/53514495/7697658
def tf_data_generator(X, y, batch_size=32, augmentations=None):
    dataset = tf.data.Dataset.from_tensor_slices((X, y)) # This is the main step for data generation
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)

    if augmentations:
        for f in augmentations:
            if np.random.uniform(0,1)<0.5:
                dataset = dataset.map(f, num_parallel_calls=2)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



# Inference
def inference_single_image(model, noisy_image):
    input_image = np.expand_dims(noisy_image, axis=0)
    predicted_image = model.predict(input_image)
    
    return predicted_image[0]

def inference_batch_images(model, noisy_images):
    predicted_image = model.predict(noisy_images, batch_size=4)
    return predicted_image


# Define the U-Net architecture
def unet(input_shape):
    tf.keras.backend.clear_session()

    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(3, 1, activation='linear')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

# Data loading and preprocessing
# Data loading and preprocessing
def load_data(data_directory, img_size=(256, 256)):
    clean_images = []
    noisy_images = []
    clean_directory = os.path.join(data_directory, 'clean'+'/'+str(noise_level))
    noisy_directory = os.path.join(data_directory, 'noisy'+'/'+str(noise_level))
    for img_file in os.listdir(clean_directory):
        clean_img = load_img(os.path.join(clean_directory, img_file), target_size=img_size)
        clean_img_array = img_to_array(clean_img) / 255.0  # Normalize to [0, 1]
        clean_images.append(clean_img_array)
        input_shape = clean_img.size[::-1] + (3,)
        noisy_img = load_img(os.path.join(noisy_directory, img_file), target_size=img_size)
        noisy_img_array = img_to_array(noisy_img) / 255.0  # Normalize to [0, 1]
        noisy_images.append(noisy_img_array)

    return np.array(clean_images), np.array(noisy_images),input_shape

# Load your dataset
data_directory = 'dataset_denoise/'+str(noise_type)
clean_images, noisy_images, input_shape = load_data(data_directory)


train_clean, temp_clean, train_noisy, temp_noisy = train_test_split(clean_images, noisy_images, test_size=0.1, random_state=42)
test_clean, val_clean, test_noisy, val_noisy = train_test_split(temp_clean, temp_noisy, test_size=0.5, random_state=42)

#SSIM loss function

#def SSIMLoss(y_true, y_pred):
#    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))



BATCH_SIZE=64
augmentation_lst = [_up_down_flip, _left_right_flip, _rotate]
image_generator_train = tf_data_generator(X=train_noisy, y=train_clean, batch_size=BATCH_SIZE, augmentations=augmentation_lst)
image_generator_val = tf_data_generator(X=val_noisy, y=val_clean, batch_size=BATCH_SIZE)

image_generator_test = tf_data_generator(X=test_noisy, y=test_clean, batch_size=BATCH_SIZE)

steps_per_epoch_train = len(noisy_train_images)
steps_per_epoch_validation = len(noisy_test_images)

callbacks_lst = [
    tf.keras.callbacks.ModelCheckpoint('models/unet_denoising_'+str(noise_type)+'_'+str(noise_level)+'_weights.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.000009, min_delta=0.0001, factor=0.75, patience=3, verbose=1, mode='min'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.0001, patience=10)
]

# Compile the model
model = unet(input_shape)
#model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mean_squared_error'])
#model.compile(optimizer=Adam(learning_rate=1e-4), loss=SSIMLoss)
# Train the model
#checkpoint = ModelCheckpoint('models/unet_denoising_'+str(noise_type)+'_'+str(noise_level)+'_weights.h5', monitor='val_loss', save_best_only=True)
#callbacks_list = [checkpoint]

# Fit the model
#model.fit(train_noisy, train_clean, batch_size=32, epochs=100, verbose=1, validation_data=(val_noisy, val_clean), callbacks=callbacks_list)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.00003))
model.fit(image_generator_train, 
          validation_data=image_generator_test,
                        steps_per_epoch=steps_per_epoch_train,
                        validation_steps=steps_per_epoch_validation,
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks_lst)

# Evaluate the model on the test data
#loss, mse = model.evaluate(image_generator_test)
#print(f'Test loss: {loss}, Test Mean Squared Error: {mse}')


predicted_images = inference_batch_images(model, test_noisy)
psnr_original_mean = 0
psnr_prediction_mean = 0

for gt_img, noisy_img, predicted_img in zip(test_clean, test_noisy, predicted_images):
    psnr_original_mean += peak_signal_noise_ratio(gt_img, noisy_img)
    psnr_prediction_mean += peak_signal_noise_ratio(gt_img, predicted_img)

psnr_original_mean/=gt_test_images.shape[0]
psnr_prediction_mean/=gt_test_images.shape[0]
print("Original average gt-noisy PSNR ->", psnr_original_mean)
print("Predicted average gt-predicted PSNR ->", psnr_prediction_mean)




predicted_images = inference_batch_images(model, noisy_test_images)
ssim_original_mean = 0
ssim_prediction_mean = 0

for gt_img, noisy_img, predicted_img in zip(test_clean, test_noisy, predicted_images):
    ssim_original_mean += ssim(gt_img, noisy_img, multichannel=True, data_range=noisy_img.max() - noisy_img.min())
    ssim_prediction_mean += ssim(gt_img, predicted_img, multichannel=True, data_range=predicted_img.max() - predicted_img.min())

ssim_original_mean/=gt_test_images.shape[0]
ssim_prediction_mean/=gt_test_images.shape[0]
print("Original average gt-noisy SSIM ->", ssim_original_mean)
print("Predicted average gt-predicted SSIM ->", ssim_prediction_mean)


os.makedirs('den/'+str(noise_type)+'/denoised/'+str(noise_level), exist_ok=True)
os.makedirs('den/'+str(noise_type)+'/clean/'+str(noise_level), exist_ok=True)

# Assuming you have defined the generator and want to predict on a batch
for i, batch in enumerate(image_generator_test):
    # Assuming you want to predict on a single batch
    predictions = model.predict(batch)
    for j in range(len(predictions)):
        #plt.figure(figsize=(8, 8))
        #plt.subplot(1, 2, 1)
        #plt.title('Input Image')
        #plt.imshow(batch[j].astype('uint8'))
        #plt.axis('off')
        batch[j].save('den/'+str(noise_type)+'/clean/'+str(noise_level)+'/clean_image_'+str(i)+'.png')
        #plt.subplot(1, 2, 2)
        #plt.title('Predicted Image')
        predictions[j].save('den/'+str(noise_type)+'/denoised/'+str(noise_level)+'/denoised_image_'+str(i)+'.png')
        #plt.imshow(predictions[j].astype('uint8'))
        #plt.axis('off')

        #plt.show()

    if i == 4:  # Modify the number of iterations based on your requirements
        break  # Stop after a certain number of iterations


