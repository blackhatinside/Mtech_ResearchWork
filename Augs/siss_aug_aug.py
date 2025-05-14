import cv2
import numpy as np
import os
import albumentations as A
from tqdm import tqdm
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

def generate_new_filename(sample_id, modality_id, slice_num):
    """Generate filename following the pattern: {sample_id}_{modality_id}_slice_{slice_num}.png"""
    return f'{sample_id}_{modality_id}_slice_{slice_num}.png'

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

class AugmentationFactory:
    @staticmethod
    def get_transform_for_class(class_name):
        """Get minimal augmentation transforms to prevent data loss"""
        # Base transform just resizes the image
        base_transform = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)
        ])

        transforms = {
            'C1': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                # Very minimal brightness/contrast adjustments
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
                # Minimal rotation to avoid losing small lesions
                A.Rotate(
                    limit=5,  # Very small rotation angle
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C2': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
                A.Rotate(
                    limit=10,  # Slightly larger rotation for medium lesions
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C3': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
                A.Rotate(
                    limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),

            'C4': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.7
                ),
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
                A.Rotate(
                    limit=25,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
            ], p=1.0),
        }

        return transforms.get(class_name, base_transform)

class ImagePairProcessor:
    def __init__(self):
        self.classifier = LesionClassifier()
        self.aug_factory = AugmentationFactory()

    def validate_augmented_pair(self, image, mask):
        """Validate augmented image-mask pair with relaxed criteria"""
        try:
            # Check if mask still contains any lesion pixels
            white_pixels = np.sum(mask > 0)
            if white_pixels == 0:
                return False

            # More lenient check for binary mask - allow some interpolated values
            # Just ensure there are mostly 0s and high values
            return True

        except Exception as e:
            logging.error(f"Error in validate_augmented_pair: {str(e)}")
            return False

    def safe_augment_pair(self, image, mask, transform, max_attempts=5):
        """Attempt augmentation with validation and simple fallback"""
        if transform is None:
            return image, mask

        for attempt in range(max_attempts):
            try:
                augmented = transform(
                    image=image.copy(),
                    mask=mask.copy()
                )
                aug_image = augmented['image']
                aug_mask = augmented['mask']

                # Ensure mask is binary
                aug_mask = (aug_mask > 127).astype(np.uint8) * 255

                if self.validate_augmented_pair(aug_image, aug_mask):
                    return aug_image, aug_mask

            except Exception as e:
                logging.error(f"Error in augmentation attempt {attempt}: {str(e)}")
                continue

        # Fallback to just resizing if all augmentations fail
        logging.warning("All augmentation attempts failed, using resize-only fallback")
        try:
            resize_only = A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)
            ])
            augmented = resize_only(image=image.copy(), mask=mask.copy())
            return augmented['image'], (augmented['mask'] > 127).astype(np.uint8) * 255
        except Exception as e:
            logging.error(f"Fallback augmentation failed: {str(e)}")
            return image, mask

    def process_image_pair(self, raw_img, mask_img):
        """Process and classify a single image pair"""
        try:
            class_name = self.classifier.get_class(mask_img)
            if class_name is None:
                return None, None, None

            transform = self.aug_factory.get_transform_for_class(class_name)
            return raw_img, mask_img, class_name

        except Exception as e:
            logging.error(f"Error in process_image_pair: {str(e)}")
            return None, None, None

def main():
    try:
        logging.info("Starting SISS dataset augmentation...")

        os.makedirs(AUG_INPUT_PATH, exist_ok=True)
        os.makedirs(AUG_MASK_PATH, exist_ok=True)

        processor = ImagePairProcessor()

        sample_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]

        for sample_id in sample_dirs:
            sample_dir = os.path.join(BASE_PATH, sample_id)
            dwi_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png') and '_1_' in f])
            mask_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png') and '_5_' in f])

            for dwi_file, mask_file in tqdm(zip(dwi_files, mask_files), total=len(dwi_files)):
                dwi_img = cv2.imread(os.path.join(sample_dir, dwi_file), cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.imread(os.path.join(sample_dir, mask_file), cv2.IMREAD_GRAYSCALE)

                if dwi_img is None or mask_img is None:
                    logging.warning(f"Could not load images: {dwi_file}, {mask_file}")
                    continue

                dwi_img, mask_img, class_name = processor.process_image_pair(dwi_img, mask_img)

                if class_name is not None:
                    transform = processor.aug_factory.get_transform_for_class(class_name)
                    aug_img, aug_mask = processor.safe_augment_pair(dwi_img, mask_img, transform)

                    # Ensure both images are properly resized to IMG_SIZE
                    if aug_img.shape != (IMG_SIZE, IMG_SIZE):
                        aug_img = cv2.resize(aug_img, (IMG_SIZE, IMG_SIZE))
                    if aug_mask.shape != (IMG_SIZE, IMG_SIZE):
                        aug_mask = cv2.resize(aug_mask, (IMG_SIZE, IMG_SIZE))
                        aug_mask = (aug_mask > 127).astype(np.uint8) * 255

                    slice_num = dwi_file.split('_')[-1].replace('.png', '')
                    aug_dwi_filename = generate_new_filename(sample_id, 1, slice_num)
                    aug_mask_filename = generate_new_filename(sample_id, 5, slice_num)

                    cv2.imwrite(os.path.join(AUG_INPUT_PATH, aug_dwi_filename), aug_img)
                    cv2.imwrite(os.path.join(AUG_MASK_PATH, aug_mask_filename), aug_mask)

        logging.info("Augmentation complete!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()