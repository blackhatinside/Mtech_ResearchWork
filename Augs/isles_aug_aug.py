# ISLES Traditional Data Augmentation

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
import re

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

# Output directories for augmented data
# AUG_BASE_PATH = "./aug_dataset"
AUG_BASE_PATH = "./aug_dataset2"
AUG_INPUT_PATH = os.path.join(AUG_BASE_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_BASE_PATH, "mask")

def get_next_case_number():
    """Get the next available case number after existing ones"""
    input_files = os.listdir(PATH_RAWDATA)
    case_numbers = []
    for f in input_files:
        if f.startswith('slice_sub-strokecase'):
            case_num = int(f.split('_')[1][14:18])  # extract number from 'sub-strokecase0001'
            case_numbers.append(case_num)
    return max(case_numbers) + 1 if case_numbers else 251  # Start from 251 if no files exist

def get_slice_range(case_id):
    """Get the range of slice numbers for a case"""
    input_files = [f for f in os.listdir(PATH_RAWDATA) if f'strokecase{case_id:04d}' in f]
    slice_numbers = [int(f.split('_')[-1].split('.')[0]) for f in input_files]
    return min(slice_numbers), max(slice_numbers)

def generate_new_filename(case_num, slice_num):
    """Generate filename following the exact pattern:
    slice_sub-strokecase{XXXX}_{YYYY}.png
    where XXXX is the case number and YYYY is the slice number
    """
    return f'slice_sub-strokecase{case_num:04d}_{slice_num:04d}.png'

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

            # 'C5': base_transform
            'C5': A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                # Only intensity-based augmentations - NO geometric transformations
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.05,  # Very small intensity changes (±5%)
                        contrast_limit=0.05,
                        p=0.7,
                        always_apply=False
                    ),
                    A.GaussNoise(
                        var_limit=(1.0, 3.0),  # Minimal noise
                        mean=0,
                        p=0.7,
                        always_apply=False
                    ),
                ], p=0.5)
            ], p=1.0),
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
    plt.figure(figsize=(12, 8))

    # Define class sizes for labels
    class_sizes = {
        'C1': '1-50 pixels',
        'C2': '51-100 pixels',
        'C3': '101-150 pixels',
        'C4': '151-200 pixels',
        'C5': '>200 pixels'
    }

    # Ensure sorted order
    classes = sorted(class_distribution.keys())
    original_counts = [class_distribution[c] for c in classes]
    augmented_counts = [augmented_distribution[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width/2, original_counts, width, label='Original')
    plt.bar(x + width/2, augmented_counts, width, label='After Augmentation')

    plt.xlabel('Classes (Lesion Size in Pixels)')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Before and After Augmentation')

    # Add class size labels
    plt.xticks(x, [f'{c}\n({class_sizes[c]})' for c in classes], rotation=0)
    plt.legend()

    plt.tight_layout()
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

def visualize_augmentation_process(processor):
    """
    Visualize how original images are transformed by augmentation techniques.
    Saves visualizations as {original image, original mask, augmented image, augmented mask}
    in a horizontal layout.
    """
    # Create visualization directory
    vis_dir = "augmentation_visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    logging.info("Generating augmentation visualizations...")
    
    # Store samples for each class (5 per class)
    class_samples = {f'C{i}': [] for i in range(1, 6)}
    samples_per_class = 5
    
    # Collect samples from each class
    raw_files = sorted([f for f in os.listdir(PATH_RAWDATA) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(PATH_DERIVATIVES) if f.endswith('.png')])
    
    for raw_f, mask_f in zip(raw_files, mask_files):
        # Load images
        raw_img = cv2.imread(os.path.join(PATH_RAWDATA, raw_f), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(os.path.join(PATH_DERIVATIVES, mask_f), cv2.IMREAD_GRAYSCALE)
        
        if raw_img is None or mask_img is None:
            continue
        
        # Get class
        class_name = processor.classifier.get_class(mask_img)
        if class_name is None:
            continue
        
        # Add to class samples if we need more
        if len(class_samples[class_name]) < samples_per_class:
            class_samples[class_name].append((raw_img, mask_img, raw_f, mask_f))
        
        # Check if we have enough samples in all classes
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    # Process each sample from each class
    for class_name, samples in class_samples.items():
        logging.info(f"Creating visualizations for class {class_name}: {len(samples)} samples")
        
        # Get transform for this class
        transform = processor.aug_factory.get_transform_for_class(class_name)
        
        for idx, (raw_img, mask_img, raw_f, mask_f) in enumerate(samples):
            # Apply augmentation
            aug_img, aug_mask = processor.safe_augment_pair(raw_img, mask_img, transform)
            
            # Extract case and slice numbers for filename
            parts = raw_f.split('_')
            case_part = parts[1]  # "sub-strokecase0252"
            slice_str = parts[2].split('.')[0]  # "0042"
            case_num = re.search(r'\d+', case_part).group()  # "0252"
            
            # Create visualization
            plt.figure(figsize=(20, 5))
            
            # Original image
            plt.subplot(1, 4, 1)
            plt.imshow(raw_img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # Original mask
            plt.subplot(1, 4, 2)
            plt.imshow(mask_img, cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')
            
            # Augmented image
            plt.subplot(1, 4, 3)
            plt.imshow(aug_img, cmap='gray')
            plt.title(f'Augmented Image ({class_name})')
            plt.axis('off')
            
            # Augmented mask
            plt.subplot(1, 4, 4)
            plt.imshow(aug_mask, cmap='gray')
            plt.title('Augmented Mask')
            plt.axis('off')
            
            # Add text with augmentation description
            if class_name == 'C1':
                aug_desc = "Small lesion (1-50 pixels): High deformation + intensity"
            elif class_name == 'C2':
                aug_desc = "Medium-small lesion (51-100 pixels): Medium deformation"
            elif class_name == 'C3':
                aug_desc = "Medium lesion (101-150 pixels): Moderate rotation + contrast"
            elif class_name == 'C4':
                aug_desc = "Medium-large lesion (151-200 pixels): Limited rotation"
            else:  # C5
                aug_desc = "Large lesion (>200 pixels): Minimal intensity changes only"
                
            plt.suptitle(f"Case {case_num}, Slice {slice_str}: {aug_desc}", fontsize=16)
            
            # Save visualization
            plt.tight_layout()
            filename = f"{class_name}_{case_num}_slice{slice_str}.png"
            plt.savefig(os.path.join(vis_dir, filename), bbox_inches='tight', dpi=300)
            plt.close()
    
    logging.info(f"Saved augmentation visualizations to {vis_dir}/")

def main():
    try:
        logging.info("Starting class-balanced medical image augmentation...")

        # Create output directories
        os.makedirs(AUG_INPUT_PATH, exist_ok=True)
        os.makedirs(AUG_MASK_PATH, exist_ok=True)

        # Initialize processors
        processor = ImagePairProcessor()
        
        # Generate augmentation visualizations
        visualize_augmentation_process(processor)

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

        # Get starting case number for synthetic images
        next_case_num = get_next_case_number()

        # Perform augmentation for each class
        logging.info("\nPerforming class-specific augmentation...")
        augmented_pairs = []
        augmented_distribution = defaultdict(int)
        current_case = next_case_num
        current_slice = 0

        for class_name, images in class_images.items():
            current_count = len(images)
            augmented_distribution[class_name] = current_count

            if current_count < target_count and class_name != 'C6': # class_name != 'C6'
                multiplier = math.ceil(target_count / current_count)
                transform = processor.aug_factory.get_transform_for_class(class_name)

                logging.info(f"\nAugmenting class {class_name} ({current_count} → {target_count})")

                for img, mask, filename in tqdm(images):
                    # Generate augmented versions
                    for aug_idx in range(multiplier - 1):
                        aug_img, aug_mask = processor.safe_augment_pair(img, mask, transform)
                        if aug_img is not None and aug_mask is not None:
                            new_filename = generate_new_filename(current_case, current_slice)

                            augmented_pairs.append((aug_img, aug_mask, new_filename))
                            augmented_distribution[class_name] += 1

                            current_slice += 1
                            if current_slice >= 100:  # Start new case after 100 slices
                                current_case += 1
                                current_slice = 0

        # Save augmented pairs
        logging.info("\nSaving augmented images...")
        for img, mask, filename in tqdm(augmented_pairs):
            cv2.imwrite(os.path.join(AUG_INPUT_PATH, filename), img)
            cv2.imwrite(os.path.join(AUG_MASK_PATH, filename), mask)

        # Visualize results
        visualize_results(class_distribution, augmented_distribution)

        logging.info("\nAugmentation complete!")
        logging.info(f"Total augmented pairs generated: {len(augmented_pairs)}")

        # Final class distribution
        logging.info("\nFinal class distribution:")
        for class_name, count in augmented_distribution.items():
            logging.info(f"{class_name}: {count} samples")

        # Log the paths for verification
        logging.info("\nSample augmented file paths:")
        if augmented_pairs:
            sample_file = augmented_pairs[0][2]
            logging.info(f"DWI image: {os.path.join(AUG_INPUT_PATH, sample_file)}")
            logging.info(f"Mask image: {os.path.join(AUG_MASK_PATH, sample_file)}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()