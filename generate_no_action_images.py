import os
import re
from PIL import Image

# Define the folder path
folder_path = 'rl_dataset_right_hand'

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"The folder '{folder_path}' does not exist.")
    exit(1)

# Get a list of existing 'no_action_i.jpg' files
existing_files = os.listdir(folder_path)
pattern = re.compile(r'no_action_(\d+)\.jpg')

existing_numbers = []
for filename in existing_files:
    match = pattern.match(filename)
    if match:
        existing_numbers.append(int(match.group(1)))

# Determine the starting number
start_number = max(existing_numbers) + 1 if existing_numbers else 1

# Generate 200 black images and corresponding txt files
for i in range(start_number, start_number + 200):
    image_name = f'no_action_{i}.jpg'
    txt_name = f'no_action_{i}.txt'

    # Create a 64x64 black image
    image = Image.new('RGB', (64, 64), color='black')

    # Save the image
    image_path = os.path.join(folder_path, image_name)
    image.save(image_path)

    # Create the txt file with content 'no_action'
    txt_path = os.path.join(folder_path, txt_name)
    with open(txt_path, 'w') as txt_file:
        txt_file.write('no_action')

    print(f"Generated {image_name} and {txt_name}")

print("Done generating images and txt files.")
