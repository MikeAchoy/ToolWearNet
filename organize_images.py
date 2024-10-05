
import os
import shutil

# Define the main directory and target folders
main_directory = 'stage3_train'
target_folder_images = 'Tool Images'
target_folder_masks = 'Tool Image Masks'

# Ensure target directories exist
os.makedirs(target_folder_images, exist_ok=True)
os.makedirs(target_folder_masks, exist_ok=True)

# Iterate through each folder in the main directory
for folder_name in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder_name)
    if os.path.isdir(folder_path):
        # Define paths for the first and second folders inside curr folder
        first_subfolder = os.path.join(folder_path, 'images')  
        second_subfolder = os.path.join(folder_path, 'masks') 

        # Check and copy the image from the first subfolder
        if os.path.isdir(first_subfolder):
            for file_name in os.listdir(first_subfolder):
                file_path = os.path.join(first_subfolder, file_name)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, target_folder_images)

        # Check and copy the image from the second subfolder
        if os.path.isdir(second_subfolder):
            for file_name in os.listdir(second_subfolder):
                file_path = os.path.join(second_subfolder, file_name)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, target_folder_masks)
