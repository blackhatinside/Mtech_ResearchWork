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
        logging.FileHandler(f'augmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Path configuration
if os.name == 'nt':
    base_path = "C:\\Cyberkid\\MyMTech\\Labwork\\SecondYear\\MyWork\\Datasets\\ISLES-2022\\ISLES-2022\\"
else:
    base_path = "/home/user/adithyaes/dataset/isles2022_png/"

# Global Variables
IMG_SIZE = 112
PATH_DATASET = base_path
PATH_RAWDATA = os.path.join(base_path, "input")
PATH_DERIVATIVES = os.path.join(base_path, "mask")
PATH_OUTPUTRAWDATA = "./output/input"
PATH_OUTPUTDERIVATIVES = "./output/mask"

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
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.8,
                        always_apply=False
                    ),
                    A.GaussNoise(
                        var_limit=(5.0, 20.0),
                        mean=0,
                        p=0.8,
                        always_apply=False
                    ),
                    A.Blur(
                        blur_limit=3,
                        p=0.8,
                        always_apply=False
                    ),
                ], p=0.5),
                A.OneOf([
                    A.Rotate(
                        limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.8,
                        always_apply=False
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.0,
                        scale_limit=0.1,
                        rotate_limit=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.8,
                        always_apply=False
                    ),
                ], p=0.5),
            ], p=1.0),

            'C2': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.7),
                    A.GaussNoise(
                        var_limit=(5.0, 20.0),
                        mean=0,
                        p=0.7
                    ),
                ], p=0.5),
                A.OneOf([
                    A.Rotate(
                        limit=20,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.7
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.0,
                        scale_limit=0.15,
                        rotate_limit=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.7
                    ),
                ], p=0.5),
            ], p=1.0),

            'C3': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.6),
                    A.GaussNoise(var_limit=(5.0, 15.0), p=0.6),
                ], p=0.5),
                A.Rotate(
                    limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.6
                ),
            ], p=1.0),

            'C4': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.Rotate(
                        limit=45,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5
                    ),
                ], p=0.5)
            ], p=1.0),

            'C5': base_transform
        }

        return transforms.get(class_name, base_transform)

class ImagePairProcessor:
    def __init__(self):
        self.classifier = LesionClassifier()
        self.aug_factory = AugmentationFactory()

    def validate_augmented_pair(self, image, mask, original_image, original_mask):
        """Validate augmented image-mask pair"""
        try:
            # Only check if mask still contains lesions
            white_pixels = np.sum(mask > 0)
            if white_pixels == 0:
                return False

            # Simple check to ensure mask is still binary
            unique_values = np.unique(mask)
            if not np.all(np.isin(unique_values, [0, 255])):
                return False

            return True

        except Exception as e:
            logging.error(f"Error in validate_augmented_pair: {str(e)}")
            return False

    def safe_augment_pair(self, image, mask, transform, max_attempts=5):
        """Attempt augmentation with validation"""
        if transform is None:
            return image, mask

        for attempt in range(max_attempts):
            try:
                augmented = transform(
                    image=image.copy(),  # Create copies to prevent modifications
                    mask=mask.copy()
                )
                aug_image = augmented['image']
                aug_mask = augmented['mask']

                # Ensure mask remains binary
                aug_mask = (aug_mask > 127).astype(np.uint8) * 255

                if self.validate_augmented_pair(aug_image, aug_mask, image, mask):
                    return aug_image, aug_mask

            except Exception as e:
                logging.error(f"Error in augmentation attempt {attempt}: {str(e)}")
                continue

        logging.warning("All augmentation attempts failed, returning original pair")
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

def visualize_results(class_distribution, augmented_distribution):
    """Visualize class distribution before and after augmentation"""
    plt.figure(figsize=(12, 6))

    classes = list(class_distribution.keys())
    original_counts = [class_distribution[c] for c in classes]
    augmented_counts = [augmented_distribution[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width/2, original_counts, width, label='Original')
    plt.bar(x + width/2, augmented_counts, width, label='After Augmentation')

    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Before and After Augmentation')
    plt.xticks(x, classes)
    plt.legend()

    plt.savefig('class_distribution.png')
    plt.close()

def display_augmented_samples(num_samples=5):
    """
    Display random augmented image pairs in a 2x5 format
    First row: Original/Augmented images
    Second row: Their corresponding masks
    """
    plt.figure(figsize=(20, 8))

    # Get random samples from output directory
    output_files = [f for f in os.listdir(PATH_OUTPUTRAWDATA) if f.endswith('.png')]
    if len(output_files) < num_samples:
        print(f"Only {len(output_files)} samples available")
        num_samples = len(output_files)

    sample_files = random.sample(output_files, num_samples)

    for idx, filename in enumerate(sample_files):
        # Load image pair
        img = cv2.imread(os.path.join(PATH_OUTPUTRAWDATA, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(PATH_OUTPUTDERIVATIVES, filename), cv2.IMREAD_GRAYSCALE)

        # Display image
        plt.subplot(2, num_samples, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Image {idx+1}')
        plt.axis('off')

        # Display corresponding mask
        plt.subplot(2, num_samples, num_samples + idx + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {idx+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

def main():
    try:
        logging.info("Starting class-balanced medical image augmentation...")

        # Create output directories
        os.makedirs(PATH_OUTPUTRAWDATA, exist_ok=True)
        os.makedirs(PATH_OUTPUTDERIVATIVES, exist_ok=True)

        # Initialize processors
        processor = ImagePairProcessor()

        # Get and sort image pairs
        raw_files = sorted([f for f in os.listdir(PATH_RAWDATA) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(PATH_DERIVATIVES) if f.endswith('.png')])

        if len(raw_files) != len(mask_files):
            raise ValueError(f"Mismatch in number of files: {len(raw_files)} raw images vs {len(mask_files)} masks")

        # Store classified images
        class_images = defaultdict(list)
        class_distribution = defaultdict(int)

        logging.info("Processing and classifying images...")
        for raw_f, mask_f in tqdm(zip(raw_files, mask_files)):
            # Load images
            raw_img = cv2.imread(os.path.join(PATH_RAWDATA, raw_f), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(os.path.join(PATH_DERIVATIVES, mask_f), cv2.IMREAD_GRAYSCALE)

            if raw_img is None or mask_img is None:
                logging.warning(f"Could not load images: {raw_f}, {mask_f}")
                continue

            # Process image pair
            raw_img, mask_img, class_name = processor.process_image_pair(raw_img, mask_img)

            if class_name is not None:
                class_images[class_name].append((raw_img, mask_img, raw_f))
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

            if current_count < target_count and class_name != 'C5':
                multiplier = math.ceil(target_count / current_count)
                transform = processor.aug_factory.get_transform_for_class(class_name)

                logging.info(f"\nAugmenting class {class_name} ({current_count} → {target_count})")
                for img, mask, filename in tqdm(images):
                    base_name = os.path.splitext(filename)[0]

                    # Save original
                    augmented_pairs.append((img, mask, f"{base_name}_orig.png"))

                    # Generate augmented versions
                    for aug_idx in range(multiplier - 1):
                        aug_img, aug_mask = processor.safe_augment_pair(img, mask, transform)
                        if aug_img is not None and aug_mask is not None:
                            augmented_pairs.append((aug_img, aug_mask, f"{base_name}_aug_{aug_idx}.png"))
                            augmented_distribution[class_name] += 1

        # Save augmented pairs
        logging.info("\nSaving augmented images...")
        for img, mask, filename in tqdm(augmented_pairs):
            # cv2.imwrite(os.path.join(PATH_OUTPUTRAWDATA, filename), img)
            cv2.imwrite(os.path.join(PATH_OUTPUTRAWDATA, filename + "_raw"), img)
            # cv2.imwrite(os.path.join(PATH_OUTPUTDERIVATIVES, filename), mask)
            cv2.imwrite(os.path.join(PATH_OUTPUTDERIVATIVES, filename + "_msk"), mask)

        # Visualize results
        visualize_results(class_distribution, augmented_distribution)

        logging.info("\nAugmentation complete!")
        logging.info(f"Total augmented pairs generated: {len(augmented_pairs)}")

        display_augmented_samples(5)

        # Final class distribution
        logging.info("\nFinal class distribution:")
        for class_name, count in augmented_distribution.items():
            logging.info(f"{class_name}: {count} samples")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()