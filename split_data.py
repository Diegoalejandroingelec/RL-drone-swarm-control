import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Folder containing the images and labels
left_hand = False
if (left_hand == True):

    data_folder = 'rl_dataset_left_hand'

    # Folders for the stratified split
    train_folder = 'rl_train_dataset_left_hand'
    test_folder = 'rl_test_dataset_left_hand'
else:
    data_folder = 'rl_dataset_right_hand'

    # Folders for the stratified split
    train_folder = 'rl_train_dataset_right_hand'
    test_folder = 'rl_test_dataset_right_hand'


# Ensure the train and test folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Helper function to copy files to the respective folders
def copy_files(image_files, dest_folder):
    for image_file in image_files:
        # Copy the image file
        image_path = os.path.join(data_folder, image_file)
        shutil.copy(image_path, os.path.join(dest_folder, image_file))
        
        # Copy the corresponding txt file
        txt_file = image_file.replace('.jpg', '.txt')
        txt_path = os.path.join(data_folder, txt_file)
        shutil.copy(txt_path, os.path.join(dest_folder, txt_file))

# Get all image files in the dataset folder
image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

# Extract class labels from image filenames
labels = []
if (left_hand == True):
    for image_file in image_files:
        if 'swarm_1' in image_file:
            labels.append('swarm_1')
        elif 'swarm_2' in image_file:
            labels.append('swarm_2')
        elif 'no_action' in image_file:
            labels.append('no_action')
else:
    for image_file in image_files:
        if 'up' in image_file:
            labels.append('up')
        elif 'backwards' in image_file:
            labels.append('backwards')
        elif 'forward' in image_file:
            labels.append('forward')
        elif 'down' in image_file:
            labels.append('down')
        elif 'left' in image_file:
            labels.append('left')
        elif 'right' in image_file:
            labels.append('right')
        elif 'take_off' in image_file:
            labels.append('take_off')
        elif 'land' in image_file:
            labels.append('land')
        elif 'no_action' in image_file:
            labels.append('no_action')


# Stratified split (80% for training and 20% for testing)
train_images, test_images = train_test_split(image_files, test_size=0.2, stratify=labels, random_state=42)

# Copy training data to the training folder
copy_files(train_images, train_folder)

# Copy testing data to the testing folder
copy_files(test_images, test_folder)

print(f'Training data saved to {train_folder}')
print(f'Testing data saved to {test_folder}')