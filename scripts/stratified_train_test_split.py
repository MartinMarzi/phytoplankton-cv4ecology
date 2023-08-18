from sklearn.model_selection import train_test_split
import os
import shutil

# Define paths
data_dir = '/Users/mmarzi/MLprojects/cv4e/dataset/data_bbox/train_test'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# List all image files
all_images = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Extract classes from filenames
classes = [img.split('_')[1] for img in all_images]

# Perform an 80-20 train-test split with stratification on classes
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42, stratify=classes)

# Move files to respective directories
for image in train_images:
    shutil.move(os.path.join(data_dir, image), os.path.join(train_dir, image))

for image in test_images:
    shutil.move(os.path.join(data_dir, image), os.path.join(test_dir, image))
