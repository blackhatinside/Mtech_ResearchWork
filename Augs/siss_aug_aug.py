# SISS Traditional Data Augmentation

import cv2
import numpy as np
import os
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'siss_augmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Path configuration
BASE_PATH = "/home/user/adithyaes/dataset/siss15_png/SISS2015_Training"
AUG_BASE_PATH = "/home/user/adithyaes/dataset/siss15_png_aug"
AUG_INPUT_PATH = os.path.join(AUG_BASE_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_BASE_PATH, "mask")

# Global Variables
IMG_SIZE = 112
DWI_MODALITY = 1
MASK_MODALITY = 5

class LesionClassifier:
    def __init__(self):
        self.class_ranges = {
            'C1': (1, 50),
            'C2': (51, 100),
            'C3': (101, 150),
            'C4': (151, 200),
            'C5': (201, float('inf'))
        }

    def get_class(self, mask):
        """Classify mask based on number of white pixels"""
        if mask is None:
            return None
        white_pixels = np.sum(mask > 0)
        if white_pixels == 0:
            return None
        for class_name, (min_val, max_val) in self.class_ranges.items():
            if min_val <= white_pixels <= max_val:
                return class_name
        return 'C5' if white_pixels > self.class_ranges['C4'][1] else None

    def get_white_pixel_count(self, mask):
        """Return the number of white pixels in mask"""
        return np.sum(mask > 0) if mask is not None else 0

class AugmentationFactory:
    @staticmethod
    def get_transform_for_class(class_name):
        """Get appropriate augmentation transform for each class"""
        base_transform = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)
        ])

        transforms = {
            'C1': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
                A.Rotate(
                    limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C2': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(p=0.7),
                A.Rotate(
                    limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C3': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(p=0.6),
                A.Rotate(
                    limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C4': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(
                    limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C5': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
            ], p=1.0),
        }

        return transforms.get(class_name, base_transform)

def main():
    try:
        logging.info("Starting SISS dataset augmentation...")

        # Create output directories
        os.makedirs(AUG_INPUT_PATH, exist_ok=True)
        os.makedirs(AUG_MASK_PATH, exist_ok=True)

        # Initialize processors
        classifier = LesionClassifier()
        aug_factory = AugmentationFactory()

        # Store classified images
        class_images = defaultdict(list)
        class_distribution = defaultdict(int)

        # First pass: classify all images
        logging.info("Processing and classifying images...")
        sample_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
        
        for sample_id in sample_dirs:
            sample_dir = os.path.join(BASE_PATH, sample_id)
            dwi_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png') and f'_{DWI_MODALITY}_' in f])
            mask_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png') and f'_{MASK_MODALITY}_' in f])
            
            for dwi_file, mask_file in zip(dwi_files, mask_files):
                # Load images
                dwi_img = cv2.imread(os.path.join(sample_dir, dwi_file), cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.imread(os.path.join(sample_dir, mask_file), cv2.IMREAD_GRAYSCALE)
                
                if dwi_img is None or mask_img is None:
                    logging.warning(f"Could not load images: {dwi_file}, {mask_file}")
                    continue
                
                # Resize images
                dwi_img = cv2.resize(dwi_img, (IMG_SIZE, IMG_SIZE))
                mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE))
                mask_img = (mask_img > 127).astype(np.uint8) * 255
                
                # Classify
                class_name = classifier.get_class(mask_img)
                if class_name is not None:
                    class_images[class_name].append((dwi_img, mask_img, dwi_file, mask_file, sample_id))
                    class_distribution[class_name] += 1

        # Find target count (maximum class size)
        target_count = max(class_distribution.values())

        logging.info("\nOriginal class distribution:")
        for class_name, count in class_distribution.items():
            logging.info(f"{class_name}: {count} samples")

        # Perform augmentation for each class
        logging.info("\nPerforming class-specific augmentation...")
        augmented_pairs = []
        augmented_distribution = defaultdict(int)

        for class_name, images in class_images.items():
            current_count = len(images)
            augmented_distribution[class_name] = current_count

            if current_count < target_count:
                multiplier = math.ceil(target_count / current_count)
                transform = aug_factory.get_transform_for_class(class_name)

                logging.info(f"\nAugmenting class {class_name} ({current_count} → {target_count})")

                for img, mask, dwi_file, mask_file, sample_id in tqdm(images):
                    # Generate augmented versions
                    for aug_idx in range(multiplier - 1):
                        try:
                            augmented = transform(
                                image=img.copy(),
                                mask=mask.copy()
                            )
                            aug_img = augmented['image']
                            aug_mask = augmented['mask']
                            
                            # Ensure mask is binary
                            aug_mask = (aug_mask > 127).astype(np.uint8) * 255
                            
                            # Check if mask still has lesions
                            if np.sum(aug_mask > 0) > 0:
                                slice_num = dwi_file.split('_slice_')[-1].replace('.png', '')
                                augmented_pairs.append((aug_img, aug_mask, sample_id, slice_num))
                                augmented_distribution[class_name] += 1
                        except Exception as e:
                            logging.error(f"Error in augmentation: {str(e)}")
                            continue

        # Save augmented pairs
        logging.info("\nSaving augmented images...")
        for img, mask, sample_id, slice_num in tqdm(augmented_pairs):
            dwi_filename = f'{sample_id}_{DWI_MODALITY}_slice_{slice_num}.png'
            mask_filename = f'{sample_id}_{MASK_MODALITY}_slice_{slice_num}.png'
            
            cv2.imwrite(os.path.join(AUG_INPUT_PATH, dwi_filename), img)
            cv2.imwrite(os.path.join(AUG_MASK_PATH, mask_filename), mask)

        logging.info("\nAugmentation complete!")
        logging.info(f"Total augmented pairs generated: {len(augmented_pairs)}")

        # Final class distribution
        logging.info("\nFinal class distribution:")
        for class_name, count in augmented_distribution.items():
            logging.info(f"{class_name}: {count} samples")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()



