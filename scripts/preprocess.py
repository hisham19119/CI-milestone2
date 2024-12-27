# import os
# import shutil
# import random
# from pathlib import Path
# import cv2

# def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#     """Splits dataset into train, validation, and test sets."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
#     for category in categories:
#         category_path = os.path.join(input_dir, category)
#         images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
#         random.shuffle(images)

#         train_end = int(len(images) * train_ratio)
#         val_end = train_end + int(len(images) * val_ratio)

#         splits = {
#             'train': images[:train_end],
#             'val': images[train_end:val_end],
#             'test': images[val_end:]
#         }

#         for split, split_images in splits.items():
#             split_dir = os.path.join(output_dir, split, category)
#             os.makedirs(split_dir, exist_ok=True)
#             for image in split_images:
#                 shutil.copy(os.path.join(category_path, image), os.path.join(split_dir, image))


# def normalize_image(image):
#     """Normalizes an image by subtracting mean and dividing by standard deviation."""
#     mean, std = image.mean(), image.std()
#     return (image - mean) / std





import os
import shutil
import random
from pathlib import Path
import cv2

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits dataset into train, validation, and test sets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for category in categories:
        category_path = os.path.join(input_dir, category)
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)

        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for image in split_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(split_dir, image))

def normalize_image(image):
    """Normalizes an image by subtracting mean and dividing by standard deviation."""
    mean, std = image.mean(), image.std()
    return (image - mean) / std
