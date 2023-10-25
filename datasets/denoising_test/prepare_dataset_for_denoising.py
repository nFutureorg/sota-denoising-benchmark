import os
import cv2
import numpy as np
import sys
from pathlib import Path
import random

def add_gamma_noise_to_images(input_folder, output_folder_clean, output_folder_noisy, scale_parameters):
    """
    Read images from input_folder, add gamma-distributed noise with different scale parameters,
    and save noisy and clean versions in output_folder_noisy and output_folder_clean, respectively.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder_clean: Path to the folder where clean images will be saved.
    - output_folder_noisy: Path to the folder where noisy images will be saved.
    - scale_parameters: List of scale parameters (θ) for the gamma distribution.
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder_clean, exist_ok=True)
    os.makedirs(output_folder_noisy, exist_ok=True)

    # List all files in the input folder
    image_files = list_image_files(input_folder)
    y=0
    x=0
    h=600
    w=1024
    #print("Gamma Noise")
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            #image_path = os.path.join(input_folder, image_file)
            #print(image_file)
            image_or = cv2.imread(image_file)
            crop_img_clean = image_or[y:y+h, x:x+w]
            #image = crop_img_clean.copy()
            #print("Gamma noise for: " + str(image_file))
            for scale_parameter in scale_parameters:
                # Add gamma-distributed noise
                # Create output folders if they don't exist
                os.makedirs(output_folder_clean+'/'+str(scale_parameter), exist_ok=True)
                os.makedirs(output_folder_noisy+'/'+str(scale_parameter), exist_ok=True)
                gamma_noisy_image = crop_img_clean.copy()
                gamma_noise = np.random.gamma(shape=2, scale=scale_parameter, size=crop_img_clean.shape[:2])
                gamma_noise = np.repeat(gamma_noise[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
                gamma_noisy_image = cv2.add(crop_img_clean, gamma_noise)
                #print(Path(image_file).stem)
                # Save clean and noisy images in separate folders
                clean_image_output_path = os.path.join(output_folder_clean+'/'+str(scale_parameter), f"{Path(image_file).stem}_gamma_{scale_parameter}.png")
                noisy_image_output_path = os.path.join(output_folder_noisy+'/'+str(scale_parameter), f"{Path(image_file).stem}_gamma_{scale_parameter}.png")
                #print(clean_image_output_path)
                cv2.imwrite(clean_image_output_path, crop_img_clean)
                cv2.imwrite(noisy_image_output_path, gamma_noisy_image)


def add_gaussian_noise_to_images(input_folder, output_folder_clean, output_folder_noisy, sigma_parameters):
    """
    Read images from input_folder, add Gaussian noise with different sigma parameters,
    and save noisy and clean versions in output_folder_noisy and output_folder_clean, respectively.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder_clean: Path to the folder where clean images will be saved.
    - output_folder_noisy: Path to the folder where noisy images will be saved.
    - sigma_parameters: List of standard deviations (sigma) for Gaussian noise.
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder_clean, exist_ok=True)
    os.makedirs(output_folder_noisy, exist_ok=True)

    # List all files in the input folder
    image_files = list_image_files(input_folder)
    y=0
    x=0
    h=600
    w=1024
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            #image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_file)
            crop_img_clean = image[y:y+h, x:x+w]
            #image = crop_img_clean.copy()
            #print("Gaussian noise for : " + str(image_file))
            for sigma_parameter in sigma_parameters:
                # Add Gaussian noise
                
                gaussian_noisy_image = image.copy()
                gaussian_noise = np.random.normal(0, sigma_parameter, crop_img_clean.shape).astype(np.uint8)
                gaussian_noisy_image = cv2.add(crop_img_clean, gaussian_noise)
                # Create output folders if they don't exist
                os.makedirs(output_folder_clean+'/'+str(sigma_parameter), exist_ok=True)
                os.makedirs(output_folder_noisy+'/'+str(sigma_parameter), exist_ok=True)
                # Save clean and noisy images in separate folders
                clean_image_output_path = os.path.join(output_folder_clean+'/'+str(sigma_parameter), f"{Path(image_file).stem}_gaussian_{sigma_parameter}.png")
                noisy_image_output_path = os.path.join(output_folder_noisy+'/'+str(sigma_parameter), f"{Path(image_file).stem}_gaussian_{sigma_parameter}.png")

                cv2.imwrite(clean_image_output_path, crop_img_clean)
                cv2.imwrite(noisy_image_output_path, gaussian_noisy_image)

# Function to recursively list image files in a directory
def list_image_files(root_dir):
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    image_files_n = random.choices(image_files, k=1000)
    return image_files_n

if __name__ == "__main__":

    input_folder = "../clean"
    output_folder_clean_gamma = "dataset_denoise/gamma/clean"
    output_folder_noisy_gamma = "dataset_denoise/gamma/noisy"
    output_folder_clean_gaussian = "dataset_denoise/gaussian/clean"
    output_folder_noisy_gaussian = "dataset_denoise/gaussian/noisy"
    sigma_parameters = [5,10,15,20,25,30,35,40,45,50]
    scale_parameters = [5,10,15,20,25,30,35,40,45,50]
    #scale_parameters = [1.0, 2.0, 5.0]  # List of gamma scale parameters
    #sigma_parameters = [10, 20, 30]     # List of Gaussian standard deviations
    #print("Gamma Noise")
    #add_gamma_noise_to_images(input_folder, output_folder_clean_gamma, output_folder_noisy_gamma, scale_parameters)
    print("Gaussian Noise")
    add_gaussian_noise_to_images(input_folder, output_folder_clean_gaussian, output_folder_noisy_gaussian, sigma_parameters)
