#!/usr/bin/env python
# coding: utf-8


# In[1]:

# ISLES Traditional Data Augmentation

'''
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

def main():
    try:
        logging.info("Starting class-balanced medical image augmentation...")

        # Create output directories
        os.makedirs(AUG_INPUT_PATH, exist_ok=True)
        os.makedirs(AUG_MASK_PATH, exist_ok=True)

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
'''

# ISLES Advanced Physiological Augmentation: HIMRA

'''
HIMRA: Advanced Physiological Augmentation: Building upon the class-aware
augmentation framework, this research implements HIMRA (Hemodynamically
Informed Mixed Reality Augmentation), a novel approach that incorporates
physiological and anatomical characteristics. The biomechanical deformation
component simulates tissue-aware deformation by generating stiffness maps using
exponential decay from lesion centroids. This deformation applies scale-dependent
transformations ranging from 15° for small lesions (C1) to 5° for larger lesions (C5),
while preserving anatomical constraints through brain-mask bounded transformations.
The hemodynamic simulation aspect replicates blood flow characteristics in stroke
regions by creating synthetic vessel patterns using controlled noise distribution.
It applies class-specific contrast ratios, varying from 1.7 for small lesions to 1.3
for larger ones, and generates penumbra effects within a 7-pixel radius of lesion
boundaries. This simulation enhances the physiological realism of augmented
images by mimicking actual stroke-related blood flow patterns. The attention-guided
occlusion component, currently in progress, focuses on selective feature emphasis
using Gaussian-based attention field generation and intensity modulation while
preserving brain mask boundaries. The implementation uses a 0.3-0.7 modulation
range for controlled feature emphasis. Current experimental results with partial
HIMRA implementation demonstrate promising anatomical realism, though Dice
scores (0.7057) remain comparable to previous class-aware augmentation results.
'''

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from datetime import datetime


# In[2]:


# Constants and Paths (unchanged)
BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
AUG_PATH = "/home/user/adithyaes/dataset/isles2022_png_aug"
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
AUG_INPUT_PATH = os.path.join(AUG_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_PATH, "mask")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = os.path.join("./output/ISLES22folder", timestamp)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# In[3]:


IMG_SIZE = 112
BATCH_SIZE = 4
LEARNINGRATE = 0.001
EPOCHS = 100
EARLYSTOPPING = 60
scaler = MinMaxScaler(feature_range=(-1, 1))


# In[4]:


def dice_coeff(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def iou(y_true, y_pred):
	intersection = K.sum(y_true * y_pred)
	union = K.sum(y_true + y_pred)
	return (intersection + 0.1) / (union - intersection + 0.1)

# def dice_score(y_true, y_pred, smooth=1e-5):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f)
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return dice

# Loss Functions
# ```
def single_dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

def binary_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return focal_loss
# ```

def hybrid_loss(y_true, y_pred):
	# Get lesion size for class weighting
	lesion_size = K.sum(y_true)
	
	# Class-specific weighting with More aggressive weighting for small lesions
	# weight = tf.where(lesion_size < 50, 2.0, tf.where(lesion_size < 100, 1.5, 1.0))
	weight = tf.where(lesion_size < 50, 2.0, 1.0)
	
	# Combine loss components with better weights
	dice_loss = single_dice_loss(y_true, y_pred)
	focal_loss = binary_focal_loss(gamma=2.5, alpha=0.3)(y_true, y_pred)
	bce_loss = binary_crossentropy_loss(y_true, y_pred)

	# Weighted combination of losses
	combined_loss = weight * (0.5 * dice_loss + 0.4 * focal_loss + 0.1 * bce_loss)
	
	return combined_loss


# In[5]:


def load_and_preprocess(file_path, is_mask=False):
	img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	if not is_mask:
		img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
	else:
		img = img / 255.0
	return img


# In[6]:


# HIMRA Augmentation Functions
def grow_small_lesion(image, mask, target_size=45):
	"""Grow small lesions to approach 50 pixels while staying within brain boundaries"""
	brain_mask = (image != -1)
	current_size = np.sum(mask > 0)
	if current_size == 0 or current_size >= 50:
		return image, mask
		
	iterations = 0
	grown_mask = mask.copy()
	grown_image = image.copy()
	
	while np.sum(grown_mask > 0) < target_size and iterations < 10:
		kernel = np.ones((3,3), np.uint8)
		dilated = ndimage.binary_dilation(grown_mask, kernel)
		# Only grow within brain boundaries
		dilated = dilated & brain_mask
		new_size = np.sum(dilated > 0)
		
		if new_size > 12:
			break
			
		# Update both mask and image
		grown_mask = dilated
		# Adjust image intensity in grown regions
		new_lesion_area = dilated & ~(mask > 0)
		grown_image[new_lesion_area] = np.mean(image[mask > 0])
		
		iterations += 1
	
	return grown_image, grown_mask.astype(mask.dtype)

def biomechanical_deformation(image, mask, lesion_class):
	"""
	Applies biomechanically realistic deformation using stiffness-weighted elastic deformation.
	"""
	if lesion_class == 1:  # Less than 50 pixels
		image, mask = grow_small_lesion(image, mask)
	
	lesion_pixels = np.where(mask > 0)
	if len(lesion_pixels[0]) == 0:
		return image, mask
	
	scales = {1: 15, 2: 12, 3: 9, 4: 7, 5: 5}
	deform_scale = scales[lesion_class]
	
	centroid = np.array([np.mean(lesion_pixels[0]), np.mean(lesion_pixels[1])])
	y_dist = np.abs(np.indices(image.shape)[0] - centroid[0])
	x_dist = np.abs(np.indices(image.shape)[1] - centroid[1])
	
	tissue_stiffness = np.exp(-0.02 * (x_dist**2 + 0.5*y_dist**2))
	dx = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=3)
	dy = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=3)
	
	y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
	deformed_coords = np.stack([y + dy * 5, x + dx * 5])
	
	return map_coordinates(image, deformed_coords, order=1, mode='reflect'), map_coordinates(mask, deformed_coords, order=0, mode='constant')

def simulate_hemodynamics(image, mask, lesion_class):
	"""
	Simulates hemodynamic effects using synthetic ADC map variations.
	"""
	contrasts = {
		1: (0.5, 1.7),  # Increased contrast for small lesions
		2: (0.4, 1.6), 
		3: (0.5, 1.5),
		4: (0.6, 1.4),
		5: (0.7, 1.3)
	}
	min_contrast, max_contrast = contrasts[lesion_class]
	
	# Create brain mask
	brain_mask = (image != -1)
	
	vessel_mask = gaussian_filter(np.random.binomial(1, 0.03, size=image.shape), sigma=1)
	perfusion_map = gaussian_filter(np.random.normal(1.0, 0.3, size=image.shape), sigma=2)
	
	perfusion_map[vessel_mask > 0.5] *= np.random.uniform(max_contrast - 0.2, max_contrast)
	
	if np.sum(mask > 0) > 0:
		# Enhance lesion visibility
		lesion_area = mask > 0
		perfusion_map[lesion_area] = np.random.uniform(1.3, 1.5)  # Make lesions brighter
		
		distance = ndimage.distance_transform_edt(1 - mask)
		penumbra = (distance < 7) & (mask == 0) & brain_mask
		perfusion_map[penumbra] *= np.random.uniform(min_contrast + 0.3, min_contrast + 0.5)
	
	# Apply perfusion only within brain
	enhanced_image = image.copy()
	enhanced_image[brain_mask] = image[brain_mask] * perfusion_map[brain_mask]
	enhanced_image[~brain_mask] = -1.0
	
	return np.clip(enhanced_image, -1, 1), mask

def attention_occlusion(image, mask):
	"""
	Applies attention-guided occlusion to simulate adversarial conditions.
	"""
	# Create brain mask (all non-zero pixels)
	brain_mask = (image != -1)
	
	noise_scale = np.random.uniform(3, 6)
	attention_field = gaussian_filter(np.random.randn(*image.shape), sigma=noise_scale)
	attention_field = np.abs(attention_field)
	attention_field = (attention_field - attention_field.min()) / (attention_field.max() - attention_field.min())
	attention_field = attention_field * 0.3 + 0.7  # Reduce effect strength
	
	# Only apply attention field within brain region
	attention_field[~brain_mask] = 1.0
	
	# Apply modulation
	modulated_image = image * attention_field
	
	# Preserve background
	modulated_image[~brain_mask] = image[~brain_mask]
	
	return modulated_image, mask



# In[7]:


class HIMRADataGenerator(tf.keras.utils.Sequence):
	def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.aug_ids = aug_ids if aug_ids is not None else []
		self.all_ids = list_IDs + self.aug_ids
		self.shuffle = shuffle
		self.__on_epoch_end()
		# Add class-aware sampling
		self.class_weights = {1: 2.0, 2: 1.5, 3: 1.2, 4: 1.0, 5: 1.0}  # Higher weight for C1

	def __len__(self):
		return int(np.floor(len(self.all_ids) / self.batch_size))

	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		batch_ids = [self.all_ids[k] for k in indexes]
		return self.__data_generation(batch_ids)

	def __on_epoch_end(self):
		self.indexes = np.arange(len(self.all_ids))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, batch_ids):
		X, y = [], []
		for case_id in batch_ids:
			is_aug = case_id in self.aug_ids
			input_dir = AUG_INPUT_PATH if is_aug else INPUT_PATH
			mask_dir = AUG_MASK_PATH if is_aug else MASK_PATH

			input_files = sorted([f for f in os.listdir(input_dir)
								if f.startswith(f'slice_{case_id}') and f.endswith('.png')])

			for f in input_files:
				img_path = os.path.join(input_dir, f)
				mask_path = os.path.join(mask_dir, f)

				img = load_and_preprocess(img_path)
				mask = load_and_preprocess(mask_path, is_mask=True)

				# Determine lesion class for class-aware augmentation
				lesion_size = np.sum(mask)
				lesion_class = 1 if lesion_size < 50 else 2 if lesion_size < 100 else 3 if lesion_size < 150 else 4 if lesion_size < 200 else 5

				# HIMRA Augmentation
				if is_aug:
					img, mask = biomechanical_deformation(img, mask, lesion_class)
					img, mask = simulate_hemodynamics(img, mask, lesion_class)
					# img, mask = attention_occlusion(img, mask)

					# Save augmented images and masks
					self.__save_augmented_data(img, mask, f)

				X.append(img)
				y.append(mask)

		return np.expand_dims(np.array(X), -1), np.expand_dims(np.array(y), -1)   # 4D mask
		# return np.array(X), np.array(y)     # 3D mask

	def __save_augmented_data(self, image, mask, filename):
		# Define directories for saving
		augmented_image_dir = "augmented_images"
		augmented_mask_dir = "augmented_masks"

		# Create directories if they don't exist
		os.makedirs(augmented_image_dir, exist_ok=True)
		os.makedirs(augmented_mask_dir, exist_ok=True)

		# Save image and mask with consistent naming
		image_filename = os.path.join(augmented_image_dir, filename)
		mask_filename = os.path.join(augmented_mask_dir, filename.replace('slice', 'mask'))

		# Save the image and mask
		cv2.imwrite(image_filename, image * 255)  # Scale image back to 0-255 for saving
		cv2.imwrite(mask_filename, mask * 255)    # Scale mask back to 0-255 for saving


# In[8]:


def conv_block(inp, filters):
	x = Conv2D(filters, 3, padding='same', use_bias=False)(inp)
	x = BatchNormalization()(x)
	return Activation('relu')(x)

def attention_gate(x, g, filters):
	g1 = Conv2D(filters, 1)(g)
	x1 = Conv2D(filters, 1)(x)
	out = Activation('relu')(add([g1, x1]))
	out = Conv2D(1, 1, activation='sigmoid')(out)
	return multiply([x, out])


# In[9]:


def create_model():   # Attention UNet (better than UNet)
	inputs = Input((IMG_SIZE, IMG_SIZE, 1))

	# Encoder with reduced filters
	x = conv_block(inputs, 32)
	skip1 = x
	x = MaxPooling2D()(x)

	x = conv_block(x, 64)
	skip2 = x
	x = MaxPooling2D()(x)

	x = conv_block(x, 128)
	skip3 = x
	x = MaxPooling2D()(x)

	x = conv_block(x, 256)
	skip4 = x
	x = MaxPooling2D()(x)

	# Bridge
	x = conv_block(x, 512)

	# Decoder with attention
	x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
	x = attention_gate(skip4, x, 256)
	x = concatenate([x, skip4])
	x = conv_block(x, 256)

	x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
	x = attention_gate(skip3, x, 128)
	x = concatenate([x, skip3])
	x = conv_block(x, 128)

	x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
	x = attention_gate(skip2, x, 64)
	x = concatenate([x, skip2])
	x = conv_block(x, 64)

	x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
	x = attention_gate(skip1, x, 32)
	x = concatenate([x, skip1])
	x = conv_block(x, 32)

	outputs = Conv2D(1, 1, activation='sigmoid')(x)

	model = Model(inputs, outputs)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
		loss=hybrid_loss,
		metrics=['accuracy', dice_coeff, iou]
	)

	return model


# In[10]:


def get_case_ids(path):
	files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
	return sorted(list({f.split('_')[1] for f in files}))


# In[11]:


def plot_training_metrics(history):
    """
    Visualize training and validation metrics over epochs
    """
    metrics = ['loss', 'dice_coeff', 'iou', 'accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16)

    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        ax.plot(history.history[metric], label=f'Training {metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        ax.set_title(f'{metric.replace("_", " ").title()} vs Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(loc='best')
        ax.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'training_metrics_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def visualize_augmented_samples(train_gen, num_samples=3):
    plt.figure(figsize=(15, 10))

    for batch_x, batch_y in train_gen:
        aug_images = batch_x
        aug_masks = batch_y
        break

    for idx in range(min(num_samples, len(aug_images))):
        # Augmented Image
        plt.subplot(2, num_samples, idx + 1)
        plt.imshow(aug_images[idx, :, :, 0], cmap='gray')
        plt.title(f'Augmented Image {idx+1}')
        plt.axis('off')

        # Corresponding Mask
        plt.subplot(2, num_samples, num_samples + idx + 1)
        plt.imshow(aug_masks[idx], cmap='gray')
        plt.title(f'Augmented Mask {idx+1}')
        plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'augmentation_samples_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def visualize_segmentation_results(model, test_gen, num_samples=3):
    for batch_x, batch_y in test_gen:
        test_images = batch_x
        true_masks = batch_y
        break

    predictions = model.predict(test_images)
    predictions = (predictions > 0.5).astype(np.float32)  # Change to float32 instead of uint8

    plt.figure(figsize=(15, 20))

    for idx in range(min(num_samples, len(test_images))):
        plt.subplot(num_samples, 2, 2*idx + 1)
        plt.imshow(test_images[idx, :, :, 0], cmap='gray')
        plt.title(f'Input DWI Image {idx+1}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*idx + 2)
        plt.imshow(predictions[idx, :, :, 0], cmap='gray')
        dice_score = dice_coeff(true_masks[idx], predictions[idx, :, :, 0])
        iou_score = iou(true_masks[idx], predictions[idx, :, :, 0])
        plt.title(f'Predicted Segmentation {idx+1}\n' +
                 f'Dice: {dice_score:.4f}, IoU: {iou_score:.4f}')
        plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'segmentation_results_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# In[12]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

case_ids = get_case_ids(INPUT_PATH)
aug_case_ids = get_case_ids(AUG_INPUT_PATH) if os.path.exists(AUG_INPUT_PATH) else []

train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Aug: {len(aug_case_ids)}")

train_gen = HIMRADataGenerator(train_ids, aug_ids=aug_case_ids)
val_gen = HIMRADataGenerator(val_ids, shuffle=False)
test_gen = HIMRADataGenerator(test_ids, shuffle=False)

model = create_model()

print("Model Summary: ", model.summary())

callbacks = [
    ModelCheckpoint('best_model.h5', monitor='val_dice_coeff', mode='max',
                   save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

results = model.evaluate(test_gen)

print("\nGenerating Training Metrics Visualization...")
plot_training_metrics(history)


# In[13]:


print("\nTest Results:")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Dice Score: {results[2]:.4f}")
print(f"IoU: {results[3]:.4f}")


# In[14]:


print("Generating Augmentation Samples Visualization...")
visualize_augmented_samples(train_gen)


# In[15]:


def numpy_dice_coeff(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

def numpy_iou(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    return (intersection + 0.1) / (union - intersection + 0.1)

def visualize_segmentation_results(model, test_gen, num_samples=3):
    for batch_x, batch_y in test_gen:
        test_images = batch_x
        true_masks = batch_y
        break

    predictions = model.predict(test_images)
    predictions = (predictions > 0.5).astype(np.float32)  # Change to float32 instead of uint8

    plt.figure(figsize=(15, 20))

    for idx in range(min(num_samples, len(test_images))):
        plt.subplot(num_samples, 2, 2*idx + 1)
        plt.imshow(test_images[idx, :, :, 0], cmap='gray')
        plt.title(f'Input DWI Image {idx+1}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*idx + 2)
        plt.imshow(predictions[idx, :, :, 0], cmap='gray')
        dice_score = numpy_dice_coeff(true_masks[idx], predictions[idx, :, :, 0])
        iou_score = numpy_iou(true_masks[idx], predictions[idx, :, :, 0])
        plt.title(f'Predicted Segmentation {idx+1}\n' +
                 f'Dice: {dice_score:.4f}, IoU: {iou_score:.4f}')
        plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'segmentation_results_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# In[16]:


print("\nGenerating Segmentation Results Visualization...")
visualize_segmentation_results(model, test_gen)


# In[17]:


print(f"\nAll visualizations have been saved to: {OUTPUT_DIRECTORY}")


# In[18]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0009_0038.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0009_0038.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0009_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[19]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0013_0045.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0013_0045.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0013_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[20]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0015_0041.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0015_0041.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0015_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[21]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0024_0039.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0024_0039.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0024_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[22]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0031_0036.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0031_0036.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0031_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[23]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0036_0056.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0036_0056.png")

# Load and preprocess
test_image = load_and_preprocess(test_image_path)
test_image = np.expand_dims(test_image, axis=[0,-1])
test_mask = load_and_preprocess(test_mask_path, is_mask=True)

# Predict
prediction = model.predict(test_image)
prediction = (prediction > 0.5).astype(np.float32)[0,:,:,0]

# Calculate metrics
dice = numpy_dice_coeff(test_mask, prediction)
iou = numpy_iou(test_mask, prediction)

print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# Visualize and save
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(test_image[0,:,:,0], cmap='gray')
plt.title('Input Image')
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.title(f'Prediction\nDice: {dice:.4f}, IoU: {iou:.4f}')
plt.savefig(os.path.join(OUTPUT_DIRECTORY, 'case0036_prediction.png'), bbox_inches='tight', dpi=300)
plt.close()


# In[24]:


def grow_small_lesion(image, mask, target_size=45):
    """Grow small lesions to approach 50 pixels while staying within brain boundaries"""
    brain_mask = (image != -1)
    current_size = np.sum(mask > 0)
    if current_size == 0 or current_size >= 50:
        return image, mask
        
    iterations = 0
    grown_mask = mask.copy()
    grown_image = image.copy()
    
    while np.sum(grown_mask > 0) < target_size and iterations < 10:
        kernel = np.ones((3,3), np.uint8)
        dilated = ndimage.binary_dilation(grown_mask, kernel)
        # Only grow within brain boundaries
        dilated = dilated & brain_mask
        new_size = np.sum(dilated > 0)
        
        if new_size > 12: # lesions with minimal pixels
            break
            
        # Update both mask and image
        grown_mask = dilated
        # Adjust image intensity in grown regions
        new_lesion_area = dilated & ~(mask > 0)
        grown_image[new_lesion_area] = np.mean(image[mask > 0])
        
        iterations += 1
    
    return grown_image, grown_mask.astype(mask.dtype)

def biomechanical_deformation(image, mask, lesion_class):
    """Process small lesions differently"""
    if lesion_class == 1:  # Less than 50 pixels
        image, mask = grow_small_lesion(image, mask)
    
    lesion_pixels = np.where(mask > 0)
    if len(lesion_pixels[0]) == 0:
        return image, mask
    
    scales = {1: 15, 2: 12, 3: 9, 4: 7, 5: 5}
    deform_scale = scales[lesion_class]
    
    centroid = np.array([np.mean(lesion_pixels[0]), np.mean(lesion_pixels[1])])
    y_dist = np.abs(np.indices(image.shape)[0] - centroid[0])
    x_dist = np.abs(np.indices(image.shape)[1] - centroid[1])
    
    tissue_stiffness = np.exp(-0.02 * (x_dist**2 + 0.5*y_dist**2))
    dx = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=3)
    dy = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=3)
    
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    deformed_coords = np.stack([y + dy * 5, x + dx * 5])
    
    return map_coordinates(image, deformed_coords, order=1, mode='reflect'), map_coordinates(mask, deformed_coords, order=0, mode='constant')

def simulate_hemodynamics(image, mask, lesion_class):
    contrasts = {
        1: (0.5, 1.7),  # Increased contrast for small lesions
        2: (0.4, 1.6), 
        3: (0.5, 1.5),
        4: (0.6, 1.4),
        5: (0.7, 1.3)
    }
    min_contrast, max_contrast = contrasts[lesion_class]
    
    # Create brain mask
    brain_mask = (image != -1)
    
    vessel_mask = gaussian_filter(np.random.binomial(1, 0.03, size=image.shape), sigma=1)
    perfusion_map = gaussian_filter(np.random.normal(1.0, 0.3, size=image.shape), sigma=2)
    
    perfusion_map[vessel_mask > 0.5] *= np.random.uniform(max_contrast - 0.2, max_contrast)
    
    if np.sum(mask > 0) > 0:
        # Enhance lesion visibility
        lesion_area = mask > 0
        perfusion_map[lesion_area] = np.random.uniform(1.3, 1.5)  # Make lesions brighter
        
        distance = ndimage.distance_transform_edt(1 - mask)
        penumbra = (distance < 7) & (mask == 0) & brain_mask
        perfusion_map[penumbra] *= np.random.uniform(min_contrast + 0.3, min_contrast + 0.5)
    
    # Apply perfusion only within brain
    enhanced_image = image.copy()
    enhanced_image[brain_mask] = image[brain_mask] * perfusion_map[brain_mask]
    enhanced_image[~brain_mask] = -1.0
    
    return np.clip(enhanced_image, -1, 1), mask

def attention_occlusion(image, mask):
    # Create precise brain mask
    brain_mask = (image != -1)
    # brain_mask = ndimage.binary_erosion(brain_mask, structure=np.ones((3,3)))
    
    noise_scale = np.random.uniform(3, 6)
    attention_field = gaussian_filter(np.random.randn(*image.shape), sigma=noise_scale)
    attention_field = np.abs(attention_field)
    attention_field = (attention_field - attention_field.min()) / (attention_field.max() - attention_field.min())
    attention_field = attention_field * 0.3 + 0.7  # Reduce effect strength
    
    # Strictly preserve background
    attention_field[~brain_mask] = 1.0
    modulated_image = image * attention_field
    modulated_image[~brain_mask] = -1.0  # Set to background value
    
    return modulated_image, mask


# In[25]:


def visualize_himra_augmentation(image_path, mask_path, output_dir):
    """
    Visualize the original image and the three HIMRA augmentation steps side by side.
    """
    # Load and preprocess original image and mask
    original_image = load_and_preprocess(image_path)
    original_mask = load_and_preprocess(mask_path, is_mask=True)

    # Determine lesion class for class-aware augmentation
    lesion_size = np.sum(original_mask)
    lesion_class = 1 if lesion_size < 50 else 2 if lesion_size < 100 else 3 if lesion_size < 150 else 4 if lesion_size < 200 else 5

    # Apply HIMRA augmentation steps
    biomech_image, biomech_mask = biomechanical_deformation(original_image.copy(), original_mask.copy(), lesion_class)
    hemo_image, hemo_mask = simulate_hemodynamics(biomech_image.copy(), biomech_mask.copy(), lesion_class)
    occluded_image, occluded_mask = attention_occlusion(hemo_image.copy(), hemo_mask.copy())

    # Plot the original image and the three HIMRA steps
    plt.figure(figsize=(20, 10))
    plt.suptitle("HIMRA Augmentation Steps", fontsize=16, y=1.05)

    # Original Image and Mask
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Biomechanical Deformation
    plt.subplot(2, 4, 2)
    plt.imshow(biomech_image, cmap='gray')
    plt.title("Biomechanical Deformation")
    plt.axis('off')

    # Hemodynamic Simulation
    plt.subplot(2, 4, 3)
    plt.imshow(hemo_image, cmap='gray')
    plt.title("Hemodynamic Simulation")
    plt.axis('off')

    # Attention-Guided Occlusion
    plt.subplot(2, 4, 4)
    plt.imshow(occluded_image, cmap='gray')
    plt.title("Attention Occlusion")
    plt.axis('off')
    
    # Masks row
    plt.subplot(2, 4, 5)
    plt.imshow(original_mask, cmap='gray')
    plt.title("Original Mask")
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(biomech_mask, cmap='gray')
    plt.title("Deformed Mask")
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(hemo_mask, cmap='gray')
    plt.title("Hemodynamic Mask")
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(occluded_mask, cmap='gray')
    plt.title("Occluded Mask")
    plt.axis('off')
    
    filename = os.path.basename(image_path).replace('.png', '_himra_steps.png')
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path


# Directory to save visualizations
output_dir = os.path.join(OUTPUT_DIRECTORY, 'himra_visualizations')
os.makedirs(output_dir, exist_ok=True)

# Test samples to visualize
test_samples = [
    "slice_sub-strokecase0009_0038.png",
    "slice_sub-strokecase0013_0045.png",
    "slice_sub-strokecase0015_0041.png",
    "slice_sub-strokecase0024_0039.png",
    "slice_sub-strokecase0031_0036.png",
    "slice_sub-strokecase0036_0056.png"
]

# Generate and save visualizations
for sample in test_samples:
    image_path = os.path.join(INPUT_PATH, sample)
    mask_path = os.path.join(MASK_PATH, sample)
    save_path = visualize_himra_augmentation(image_path, mask_path, output_dir)
    print(f"Saved HIMRA visualization for {sample} to {save_path}")


def calculate_class_wise_dice(model, test_gen):
    # Initialize dictionaries to store results
    class_dice_scores = {f'C{i}': [] for i in range(1, 6)}
    
    # Process all test samples
    for batch_x, batch_y in test_gen:
        predictions = model.predict(batch_x)
        predictions = (predictions > 0.5).astype(np.float32)
        
        for i in range(len(batch_y)):
            true_mask = batch_y[i]
            pred_mask = predictions[i,:,:,0]
            
            # Calculate lesion size
            lesion_size = np.sum(true_mask)
            
            # Determine class
            if lesion_size == 0:
                continue
            elif lesion_size <= 50:
                class_name = 'C1'
            elif lesion_size <= 100:
                class_name = 'C2'
            elif lesion_size <= 150:
                class_name = 'C3'
            elif lesion_size <= 200:
                class_name = 'C4'
            else:
                class_name = 'C5'
            
            # Calculate Dice score
            dice = numpy_dice_coeff(true_mask, pred_mask)
            class_dice_scores[class_name].append(dice)
    
    # Calculate mean Dice scores for each class
    mean_dice_scores = {cls: np.mean(scores) if scores else 0 for cls, scores in class_dice_scores.items()}
    
    return mean_dice_scores

def visualize_class_wise_dice(scores, output_dir):
    # Prepare data for plotting
    classes = sorted(scores.keys())
    values = [scores[cls] for cls in classes]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(classes, values, color='skyblue')
    plt.xlabel('Lesion Size Class')
    plt.ylabel('Mean Dice Score')
    plt.title('Class-wise Dice Scores')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Save and show plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'class_wise_dice_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path

# After model training and evaluation
class_wise_dice = calculate_class_wise_dice(model, test_gen)
print("\nClass-wise Dice Scores:")
for cls, score in class_wise_dice.items():
    print(f"{cls}: {score:.4f}")

# Visualize results
save_path = visualize_class_wise_dice(class_wise_dice, OUTPUT_DIRECTORY)
print(f"\nSaved class-wise Dice scores visualization to: {save_path}")

