#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
OUTPUT_DIRECTORY = "./output/ISLESfolder"
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

def dice_loss(y_true, y_pred):
	# Get lesion size for class weighting
	lesion_size = K.sum(y_true)
	
	# More aggressive weighting for small lesions
	weight = tf.where(lesion_size < 50, 3.0,  # Increased from 2.0 to 3.0 for tiny lesions
					  tf.where(lesion_size < 100, 1.5, 1.0))
	
	# Calculate class imbalance ratio for focal loss adaptation
	# Smaller lesions need higher gamma to focus more on hard examples
	adaptive_gamma = tf.where(lesion_size < 50, 3.0, 2.5)
	adaptive_alpha = tf.where(lesion_size < 50, 0.35, 0.3)
	
	# Combine loss components with better weights for small lesions
	dice_term = 1 - dice_coeff(y_true, y_pred)
	focal_loss = binary_focal_loss(gamma=adaptive_gamma, alpha=adaptive_alpha)(y_true, y_pred)
	bce_loss = binary_crossentropy_loss(y_true, y_pred)
	
	# Give more weight to focal loss for small lesions
	focal_weight = tf.where(lesion_size < 50, 0.5, 0.4)
	dice_weight = tf.where(lesion_size < 50, 0.4, 0.5)
	bce_weight = 0.1
	
	return weight * (dice_weight * dice_term + focal_weight * focal_loss + bce_weight * bce_loss)


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
    
    # Enhance visibility of very small lesions (< 20 pixels)
    if current_size < 20:
        # Increase contrast for tiny lesions to make them more visible
        lesion_pixels = mask > 0
        if np.sum(lesion_pixels) > 0:
            # Increase intensity of existing lesion by 15-25% for better visibility
            intensity_boost = np.random.uniform(0.15, 0.25)
            intensity_val = np.mean(image[lesion_pixels])
            grown_image[lesion_pixels] = np.clip(intensity_val * (1 + intensity_boost), -1, 1)
    
    # Different growth strategies based on lesion size
    dilate_iterations = 1
    if current_size < 10:
        kernel_size = 3  # Larger kernel for very tiny lesions
        target_size = min(40, target_size)  # More conservative target for tiny lesions
        dilate_iterations = 2  # More aggressive dilation for tiny lesions
    elif current_size < 20:
        kernel_size = 3
        target_size = min(42, target_size)
    else:
        kernel_size = 3
        target_size = min(45, target_size)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    while np.sum(grown_mask > 0) < target_size and iterations < 10:
        # Apply dilation multiple times for very small lesions
        dilated = grown_mask.copy()
        for _ in range(dilate_iterations):
            dilated = ndimage.binary_dilation(dilated, kernel)
        
        # Only grow within brain boundaries
        dilated = dilated & brain_mask
        new_size = np.sum(dilated > 0)
        
        if new_size > target_size * 1.2:  # Don't grow too much
            break
            
        # Update both mask and image
        grown_mask = dilated
        
        # Adjust image intensity in grown regions with a gradual falloff from center
        new_lesion_area = dilated & ~(mask > 0)
        if np.sum(new_lesion_area) > 0:
            if np.sum(mask > 0) > 0:
                # Calculate distance from original lesion
                dist = ndimage.distance_transform_edt(~(mask > 0))
                dist = dist * new_lesion_area
                
                # Apply intensity gradient (stronger near original lesion)
                orig_intensity = np.mean(image[mask > 0])
                max_dist = np.max(dist[new_lesion_area])
                if max_dist > 0:
                    for d in range(1, int(max_dist) + 1):
                        d_mask = (dist >= d-1) & (dist < d) & new_lesion_area
                        if np.sum(d_mask) > 0:
                            factor = 1.0 - (0.15 * d / max_dist)  # Gradual decrease in intensity
                            grown_image[d_mask] = orig_intensity * factor
        
        iterations += 1
    
    return grown_image, grown_mask.astype(mask.dtype)

def biomechanical_deformation(image, mask, lesion_class):
    """
    Applies biomechanically realistic deformation using stiffness-weighted elastic deformation.
    """
    # Always grow small lesions first, before applying deformations
    if lesion_class == 1:  # Less than 50 pixels
        image, mask = grow_small_lesion(image, mask)
    
    lesion_pixels = np.where(mask > 0)
    if len(lesion_pixels[0]) == 0:
        return image, mask
    
    # Customize scale based on lesion size, more careful with small lesions
    scales = {1: 12, 2: 10, 3: 8, 4: 6, 5: 5}  # Reduced scales for small lesions
    deform_scale = scales[lesion_class]
    
    # Calculate centroid of the lesion
    centroid = np.array([np.mean(lesion_pixels[0]), np.mean(lesion_pixels[1])])
    y_dist = np.abs(np.indices(image.shape)[0] - centroid[0])
    x_dist = np.abs(np.indices(image.shape)[1] - centroid[1])
    
    # Create a tissue stiffness map (affects deformation strength)
    # For small lesions, make the area near the lesion more rigid (less deformation)
    if lesion_class <= 2:  # Class 1 and 2 (small lesions)
        # Create a protected zone around small lesions to preserve them
        # Calculate the distance transform from the lesion
        dist_from_lesion = ndimage.distance_transform_edt(~(mask > 0))
        # Create a protection factor that decreases with distance from lesion
        protection_factor = np.exp(-0.1 * dist_from_lesion)
        # Apply more rigid tissue properties near small lesions
        tissue_stiffness = np.exp(-0.04 * (x_dist**2 + 0.5*y_dist**2)) * (1 - 0.5 * protection_factor)
    else:
        tissue_stiffness = np.exp(-0.02 * (x_dist**2 + 0.5*y_dist**2))
    
    # For small lesions, use smaller sigma for more localized deformations
    sigma = 4 if lesion_class > 2 else 2
    
    # Generate deformation fields
    dx = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=sigma)
    dy = deform_scale * tissue_stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=sigma)
    
    # For small lesions, reduce deformation strength at lesion center
    if lesion_class <= 2:
        lesion_factor = 0.7  # Reduce deformation by 30% at lesion
        dx = dx * (1 - lesion_factor * (mask > 0))
        dy = dy * (1 - lesion_factor * (mask > 0))
    
    # Create coordinate grids and apply deformation
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    deformation_strength = 3 if lesion_class > 2 else 2  # Less aggressive for small lesions
    deformed_coords = np.stack([y + dy * deformation_strength, x + dx * deformation_strength])
    
    return map_coordinates(image, deformed_coords, order=1, mode='reflect'), map_coordinates(mask, deformed_coords, order=0, mode='constant')

def generate_synthetic_small_lesions(image, existing_mask=None, num_lesions=2, size_range=(10, 40)):
    """
    Generate synthetic small lesions and add them to the image.
    This function helps the model learn to detect small lesions by artificially creating them.
    
    Args:
        image: The input image
        existing_mask: Existing mask (if any)
        num_lesions: Number of synthetic lesions to generate
        size_range: Size range of lesions in pixels
        
    Returns:
        Augmented image and mask
    """
    # Create brain mask (non-background region)
    brain_mask = (image != -1)
    
    # If there's an existing mask, extract it
    if existing_mask is not None:
        mask = existing_mask.copy()
    else:
        mask = np.zeros_like(image)
    
    # Create empty mask for synthetic lesions
    synthetic_mask = np.zeros_like(image)
    
    # Generate random small lesions
    for _ in range(num_lesions):
        # Randomly choose lesion size
        lesion_size = np.random.randint(size_range[0], size_range[1])
        lesion_radius = int(np.sqrt(lesion_size / np.pi))
        
        # Find valid location within the brain (not on existing lesions)
        valid_locations = brain_mask & (mask == 0)
        if np.sum(valid_locations) == 0:
            continue  # Skip if no valid locations
            
        # Find valid coordinates
        valid_coords = np.where(valid_locations)
        if len(valid_coords[0]) == 0:
            continue
            
        # Randomly choose a location
        idx = np.random.randint(0, len(valid_coords[0]))
        center_y, center_x = valid_coords[0][idx], valid_coords[1][idx]
        
        # Create the lesion (circle)
        y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        
        # Create a gradual intensity profile for realistic appearance
        raw_lesion = (dist_from_center <= lesion_radius).astype(np.float32)
        
        # Ensure lesion is within brain boundaries
        lesion_mask = raw_lesion & brain_mask
        
        # Check if lesion is too small after brain masking
        if np.sum(lesion_mask) < size_range[0] * 0.7:
            continue
            
        # Add the lesion to the synthetic mask
        synthetic_mask = np.maximum(synthetic_mask, lesion_mask)
        
        # Modify the image - make lesion brighter with realistic internal texture
        lesion_area = lesion_mask > 0
        if np.sum(lesion_area) > 0:
            # Calculate distance transform within lesion for intensity gradient
            distance = ndimage.distance_transform_edt(lesion_area)
            normalized_distance = distance / (np.max(distance) + 1e-10)
            
            # Create intensity variation - brighter in center, fading at edges
            intensity_gradient = 1.0 - 0.3 * normalized_distance
            
            # Add random texture
            texture = gaussian_filter(np.random.randn(*image.shape), sigma=0.7)[lesion_area]
            
            # Determine lesion intensity based on surrounding area
            # Get a ring around the lesion
            dilated = ndimage.binary_dilation(lesion_area, iterations=2)
            surrounding = dilated & ~lesion_area & brain_mask
            
            if np.sum(surrounding) > 0:
                surrounding_intensity = np.mean(image[surrounding])
                # Make lesion brighter than surroundings (typical for DWI)
                lesion_intensity = surrounding_intensity + np.random.uniform(0.2, 0.4)
            else:
                # Fallback if no surrounding area found
                lesion_intensity = np.random.uniform(0.5, 0.8)
                
            # Apply intensity with texture
            texture_scale = 0.1
            image[lesion_area] = lesion_intensity * intensity_gradient[lesion_area] * (1 + texture_scale * texture)
    
    # Combine synthetic mask with existing mask
    combined_mask = np.maximum(mask, synthetic_mask)
    
    return image, combined_mask

def simulate_hemodynamics(image, mask, lesion_class):
    """
    Simulates hemodynamic effects using synthetic ADC map variations.
    Enhances small lesion visibility with stronger contrast.
    """
    contrasts = {
        1: (0.4, 1.9),  # Increased contrast range for small lesions
        2: (0.4, 1.7), 
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
        
        # Stronger contrast enhancement for small lesions (class 1)
        if lesion_class == 1:
            # Make tiny lesions significantly brighter for better visibility
            perfusion_boost = np.random.uniform(1.5, 1.8)  # Higher boost for small lesions
            perfusion_map[lesion_area] = perfusion_boost
            
            # Create a sharper contrast at lesion boundary
            distance = ndimage.distance_transform_edt(1 - mask)
            
            # Narrow, darker penumbra around small lesions for better contrast
            close_penumbra = (distance < 3) & (mask == 0) & brain_mask
            far_penumbra = (distance >= 3) & (distance < 6) & (mask == 0) & brain_mask
            
            # Create a stronger dark ring immediately around the lesion
            perfusion_map[close_penumbra] *= np.random.uniform(0.3, 0.4)  # Darker boundary
            perfusion_map[far_penumbra] *= np.random.uniform(min_contrast + 0.2, min_contrast + 0.3)
            
            # Add texture to lesion for better feature representation
            if np.sum(lesion_area) > 0:
                texture = gaussian_filter(np.random.randn(*image.shape), sigma=0.5)[lesion_area]
                texture_scale = 0.1  # Small variation to prevent overfitting
                perfusion_map[lesion_area] *= (1 + texture_scale * texture)
        else:
            # Standard processing for larger lesions
            perfusion_map[lesion_area] = np.random.uniform(1.3, 1.5)
            
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

# Additional traditional augmentation techniques
def random_rotation(image, mask, max_angle=15):
    """Apply random rotation to image and mask"""
    angle = np.random.uniform(-max_angle, max_angle)
    # Get center of the image (where the brain is likely centered)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to image and mask
    rotated_img = cv2.warpAffine(image, M, image.shape, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    rotated_mask = cv2.warpAffine(mask, M, mask.shape, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_img, rotated_mask

def random_brightness_contrast(image, mask, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2)):
    """Apply random brightness and contrast adjustments"""
    # Only modify brain region
    brain_mask = (image != -1)
    
    # Apply brightness adjustment
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    adjusted_img = image.copy()
    adjusted_img[brain_mask] = image[brain_mask] + brightness
    
    # Apply contrast adjustment
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    adjusted_img[brain_mask] = ((adjusted_img[brain_mask] - np.mean(adjusted_img[brain_mask])) * contrast) + np.mean(adjusted_img[brain_mask])
    
    # Clip values to valid range
    adjusted_img = np.clip(adjusted_img, -1, 1)
    
    # Keep background unchanged
    adjusted_img[~brain_mask] = -1
    
    return adjusted_img, mask

def random_noise(image, mask, noise_level=0.05):
    """Add random Gaussian noise to the image"""
    brain_mask = (image != -1)
    
    # Add Gaussian noise only to brain region
    noisy_img = image.copy()
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_img[brain_mask] = image[brain_mask] + noise[brain_mask]
    
    # Clip values to valid range
    noisy_img = np.clip(noisy_img, -1, 1)
    
    # Keep background unchanged
    noisy_img[~brain_mask] = -1
    
    return noisy_img, mask

def random_flip(image, mask):
    """Randomly flip image horizontally"""
    if np.random.random() > 0.5:
        return np.fliplr(image), np.fliplr(mask)
    return image, mask

def elastic_transform(image, mask, alpha=50, sigma=5):
    """Apply elastic transform to both image and mask"""
    brain_mask = (image != -1)
    
    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha
    
    # Create meshgrid
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    
    # Apply deformation
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Interpolate image
    transformed_img = map_coordinates(image, indices, order=1).reshape(image.shape)
    transformed_mask = map_coordinates(mask, indices, order=0).reshape(mask.shape)
    
    # Preserve background
    transformed_img[~brain_mask] = -1
    
    return transformed_img, transformed_mask


# In[7]:


class HIMRADataGenerator(tf.keras.utils.Sequence):
	def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.aug_ids = aug_ids if aug_ids is not None else []
		self.shuffle = shuffle
		self.indexes = np.arange(len(self.list_IDs))
		
		# Analyze and categorize samples by lesion size
		self.small_lesion_indexes = []
		self.other_indexes = []
		
		for i, img_id in enumerate(self.list_IDs):
			mask_path = os.path.join(MASK_PATH, img_id)
			mask = load_and_preprocess(mask_path, is_mask=True)
			lesion_size = np.sum(mask > 0)
			
			if lesion_size > 0 and lesion_size <= 50:
				self.small_lesion_indexes.append(i)
			else:
				self.other_indexes.append(i)
				
		print(f"Found {len(self.small_lesion_indexes)} images with small lesions (≤50 pixels)")
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		batch_ids = [self.list_IDs[i] for i in batch_indexes]
		X, y = self.__data_generation(batch_ids)
		return X, y

	def on_epoch_end(self):
		if self.shuffle:
			# Oversample small lesions by duplicating them
			if len(self.small_lesion_indexes) > 0:
				# Calculate oversampling ratio - more small lesions
				small_lesion_ratio = 2.0  # Small lesions appear twice as often
				
				# Create balanced index list with oversampling
				balanced_indexes = self.other_indexes.copy()
				
				# Add small lesion indexes with oversampling
				small_lesion_samples = int(len(self.small_lesion_indexes) * small_lesion_ratio)
				oversample_indexes = np.random.choice(
					self.small_lesion_indexes, 
					size=small_lesion_samples, 
					replace=True  # Allow replacement for more diversity
				)
				balanced_indexes.extend(oversample_indexes)
				
				# Shuffle the combined indexes
				np.random.shuffle(balanced_indexes)
				self.indexes = np.array(balanced_indexes)
			else:
				np.random.shuffle(self.indexes)

	def __data_generation(self, batch_ids):
		X = np.empty((len(batch_ids), IMG_SIZE, IMG_SIZE, 1))
		y = np.empty((len(batch_ids), IMG_SIZE, IMG_SIZE, 1))
		
		for i, img_id in enumerate(batch_ids):
			img_path = os.path.join(INPUT_PATH, img_id)
			mask_path = os.path.join(MASK_PATH, img_id)
			
			# Load and preprocess
			img = load_and_preprocess(img_path)
			mask = load_and_preprocess(mask_path, is_mask=True)
			
			# Apply HIMRA augmentation if this is an augmentation sample
			if img_id in self.aug_ids:
				# Determine lesion size class
				lesion_size = np.sum(mask > 0)
				if lesion_size == 0:
					lesion_class = 0
					
					# For empty masks, randomly generate synthetic small lesions ~20% of the time
					if np.random.random() < 0.2:
						img, mask = generate_synthetic_small_lesions(
							img, mask, 
							num_lesions=np.random.randint(1, 3),
							size_range=(10, 35)
						)
						lesion_class = 1  # Now it's a small lesion class
						
				elif lesion_size <= 50:
					lesion_class = 1
					# Prepare a random number for augmentation decisions
					rand_prob = np.random.uniform(0, 1)
					
					# For small lesions, use specific sequences of augmentations
					if rand_prob < 0.25:
						# Strategy 1: Grow small lesion followed by hemodynamic simulation
						img, mask = grow_small_lesion(img, mask, target_size=np.random.randint(35, 45))
						img, mask = simulate_hemodynamics(img, mask, lesion_class)
					elif rand_prob < 0.5:
						# Strategy 2: More conservative growth + biomechanical deformation
						img, mask = grow_small_lesion(img, mask, target_size=np.random.randint(30, 40))
						img, mask = biomechanical_deformation(img, mask, lesion_class)
					elif rand_prob < 0.75:
						# Strategy 3: Pure contrast enhancement
						img, mask = simulate_hemodynamics(img, mask, lesion_class)
					else:
						# Strategy 4: Careful growth and less aggressive deformation
						img, mask = grow_small_lesion(img, mask, target_size=np.random.randint(25, 35))
						img, mask = attention_occlusion(img, mask)
					
					# For even better C1 training, occasionally add another small lesion
					if np.random.random() < 0.3:
						img, mask = generate_synthetic_small_lesions(
							img, mask, 
							num_lesions=1,
							size_range=(8, 30)
						)
						
					# Additional augmentations: brightness/contrast
					if np.random.random() < 0.7:  # Higher probability for small lesions
						img, mask = random_brightness_contrast(
							img, mask, 
							brightness_range=(-0.1, 0.2),  # Biased toward brightening
							contrast_range=(0.9, 1.3)      # Biased toward higher contrast
						)
						
				elif lesion_size <= 100:
					lesion_class = 2
					
					# For C2 lesions, occasionally add a small synthetic lesion
					if np.random.random() < 0.2:
						img, mask = generate_synthetic_small_lesions(
							img, mask, 
							num_lesions=1,
							size_range=(8, 25)
						)
						
				elif lesion_size <= 150:
					lesion_class = 3
				elif lesion_size <= 200:
					lesion_class = 4
				else:
					lesion_class = 5
				
				# For all other lesion classes, use standard augmentation approach
				if lesion_class > 1 or lesion_class == 0:
					# Biomechanical deformation
					if np.random.random() < 0.6:
						img, mask = biomechanical_deformation(img, mask, lesion_class)
					
					# Hemodynamic simulation
					if np.random.random() < 0.5:
						img, mask = simulate_hemodynamics(img, mask, lesion_class)
					
					# Attention occlusion
					if np.random.random() < 0.4:
						img, mask = attention_occlusion(img, mask)
					
					# Traditional augmentations
					if np.random.random() < 0.5:
						img, mask = random_brightness_contrast(img, mask)
				
				# Common augmentations for all classes
				if np.random.random() < 0.3:
					img, mask = random_rotation(img, mask, max_angle=10)
				
				if np.random.random() < 0.3:
					img, mask = random_flip(img, mask)
				
				if np.random.random() < 0.2:
					img, mask = random_noise(img, mask, noise_level=0.03)
				
			# Add to batch
			X[i,] = img.reshape((IMG_SIZE, IMG_SIZE, 1))
			y[i,] = mask.reshape((IMG_SIZE, IMG_SIZE, 1))
		
		return X, y


# In[8]:


def conv_block(inp, filters):
    """Enhanced convolutional block with residual connection to preserve small lesion features"""
    x = Conv2D(filters, 3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add residual connection if input and output have same shape
    if K.int_shape(inp)[-1] == filters:
        x = Add()([x, inp])
    
    return x

def attention_gate(x, g, filters):
    """Enhanced attention gate with sharper focus for small lesions"""
    # X: skip connection input (feature maps from encoder)
    # g: gating signal (feature maps from decoder)
    
    # Compress channels for both inputs
    theta_x = Conv2D(filters//2, 1, padding='same')(x)  # Skip connection processing
    phi_g = Conv2D(filters//2, 1, padding='same')(g)    # Gating signal processing

    # Combine signals
    f = Activation('relu')(add([theta_x, phi_g]))
    
    # Create attention coefficients - use 1x1 followed by 3x3 for better small feature capture
    f = Conv2D(filters//4, 1, padding='same')(f)
    f = Conv2D(filters//4, 3, padding='same')(f)  # Added 3x3 conv for better spatial context
    f = Conv2D(1, 1, padding='same')(f)
    
    # Apply sigmoid activation to get attention map
    alpha = Activation('sigmoid')(f)
    
    # Apply attention weights
    attention_weighted_x = multiply([x, alpha])
    
    # Residual connection to preserve small features that might be missed by attention
    output = Add()([attention_weighted_x, x * 0.1])  # Residual with small weight to maintain stability
    
    return output


# In[9]:


def create_model():
	inputs = Input((IMG_SIZE, IMG_SIZE, 1))

	# Encoder path with deeper network for better feature extraction
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
	
	# Additional deep feature extraction for small lesions
	x = SpatialDropout2D(0.2)(x)  # Helps prevent overfitting to larger lesions
	
	# Multi-scale feature extraction at bridge level
	# This helps capture features at multiple scales, beneficial for different lesion sizes
	bridge_1x1 = Conv2D(128, 1, activation='relu', padding='same')(x)
	bridge_3x3 = Conv2D(128, 3, activation='relu', padding='same')(x)
	bridge_5x5 = Conv2D(128, 5, activation='relu', padding='same')(x)
	bridge_pooling = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
	bridge_pooling = Conv2D(128, 1, activation='relu', padding='same')(bridge_pooling)
	
	# Concatenate multi-scale features
	x = concatenate([bridge_1x1, bridge_3x3, bridge_5x5, bridge_pooling])
	x = conv_block(x, 512)  # Reduced filter count to manage model complexity

	# Decoder with enhanced attention mechanism
	x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
	attention4 = attention_gate(skip4, x, 256)
	x = concatenate([attention4, skip4])
	x = conv_block(x, 256)

	x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
	attention3 = attention_gate(skip3, x, 128)
	x = concatenate([attention3, skip3])
	x = conv_block(x, 128)

	x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
	attention2 = attention_gate(skip2, x, 64)
	x = concatenate([attention2, skip2])
	x = conv_block(x, 64)

	x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
	attention1 = attention_gate(skip1, x, 32)
	x = concatenate([attention1, skip1])
	x = conv_block(x, 32)
	
	# Add specialized small lesion detection path - higher resolution features from earlier layers
	# This helps retain fine details that might be lost in deeper layers
	small_lesion_path = Conv2D(16, 1, activation='relu', padding='same')(attention1)
	small_lesion_path = Conv2D(16, 3, activation='relu', padding='same')(small_lesion_path)
	
	# Combine with main path
	x = concatenate([x, small_lesion_path])
	x = Conv2D(32, 3, activation='relu', padding='same')(x)
	
	# Final convolution
	outputs = Conv2D(1, 1, activation='sigmoid')(x)

	model = Model(inputs, outputs)

	# Use a lower learning rate to stabilize training with the enhanced loss function
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
		loss=dice_loss,
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

def post_process_predictions(predictions, images, threshold=0.5, small_lesion_boost=0.15):
    """
    Post-processes model predictions to enhance small lesion detection.
    
    Args:
        predictions: Model output predictions (probability maps)
        images: Original input images
        threshold: Base threshold for binary prediction
        small_lesion_boost: Confidence boost for small detections
        
    Returns:
        Processed binary predictions
    """
    processed_predictions = []
    
    for i in range(len(predictions)):
        pred = predictions[i, :, :, 0]
        img = images[i, :, :, 0]
        
        # Create binary mask using base threshold
        binary_pred = (pred > threshold).astype(np.float32)
        
        # Find connected components
        labeled_mask, num_components = ndimage.label(binary_pred)
        
        # Process each component
        for comp_idx in range(1, num_components + 1):
            component_mask = (labeled_mask == comp_idx)
            component_size = np.sum(component_mask)
            
            # Special handling for small components (potential small lesions)
            if 5 < component_size <= 80:  # Small to medium lesions
                # Check if prediction confidence is borderline
                mean_confidence = np.mean(pred[component_mask])
                
                if 0.4 < mean_confidence < 0.7:  # Borderline confidence
                    # Check image characteristics in this region
                    component_img_values = img[component_mask]
                    
                    # Calculate local statistics
                    local_mean = np.mean(component_img_values)
                    
                    # Create a dilated region around the component to compare with surroundings
                    dilated = ndimage.binary_dilation(component_mask, iterations=2)
                    surrounding = dilated & ~component_mask
                    if np.sum(surrounding) > 0:
                        surrounding_mean = np.mean(img[surrounding])
                        
                        # If component is different from surroundings, boost confidence
                        if abs(local_mean - surrounding_mean) > 0.1:
                            # Boost prediction values for component
                            pred[component_mask] += small_lesion_boost
            
            # Remove very small components that are likely noise
            elif component_size <= 5:
                # Only keep very small components if they have very high confidence
                if np.mean(pred[component_mask]) < 0.8:
                    pred[component_mask] = 0.0
        
        # Re-threshold after adjustments
        final_pred = (pred > threshold).astype(np.float32)
        processed_predictions.append(final_pred)
    
    return np.array(processed_predictions)

def evaluate_with_postprocessing(model, test_gen):
    """Evaluates model with post-processing for better small lesion detection"""
    dice_scores = []
    iou_scores = []
    
    for batch_x, batch_y in test_gen:
        # Get raw predictions
        predictions = model.predict(batch_x)
        
        # Apply post-processing
        processed_predictions = post_process_predictions(
            predictions, batch_x, threshold=0.5, small_lesion_boost=0.15
        )
        
        # Calculate metrics
        for i in range(len(batch_y)):
            true_mask = batch_y[i, :, :, 0]
            pred_mask = processed_predictions[i]
            
            dice = numpy_dice_coeff(true_mask, pred_mask)
            iou_val = numpy_iou(true_mask, pred_mask)
            
            dice_scores.append(dice)
            iou_scores.append(iou_val)
    
    # Calculate mean scores
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    return mean_dice, mean_iou


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
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Save and show plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'class_wise_dice_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path

def main():
    """Main function to train and evaluate the model with all improvements"""
    print("Starting HIMRA-enhanced segmentation training for improved small lesion detection")
    
    # Get case IDs
    case_ids = get_case_ids(INPUT_PATH)
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    
    # Create augmented training set with oversampling of C1 (small lesion) class
    augmented_train_cases = train_cases.copy()
    
    # Create data generators
    train_gen = HIMRADataGenerator(train_cases, aug_ids=augmented_train_cases, batch_size=BATCH_SIZE)
    val_gen = HIMRADataGenerator(test_cases, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = HIMRADataGenerator(test_cases, batch_size=1, shuffle=False)
    
    # Create and compile model with small lesion enhancements
    model = create_model()
    model.summary()
    
    # Setup callbacks
    checkpoint_path = os.path.join(OUTPUT_DIRECTORY, 'best_model_small_lesion.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_dice_coeff', 
            verbose=1, 
            save_best_only=True, 
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLYSTOPPING,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6
        )
    ]
    
    # Train the model with improved small lesion focus
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training metrics
    plot_training_metrics(history)
    
    # Evaluate with standard metrics
    test_loss, test_acc, test_dice, test_iou = model.evaluate(test_gen)
    print(f"\nStandard Evaluation:")
    print(f"Dice Score: {test_dice:.4f}")
    print(f"IoU Score: {test_iou:.4f}")
    
    # Evaluate with post-processing for small lesions
    print("\nEvaluating with small lesion post-processing:")
    pp_dice, pp_iou = evaluate_with_postprocessing(model, test_gen)
    print(f"Post-processed Dice Score: {pp_dice:.4f}")
    print(f"Post-processed IoU Score: {pp_iou:.4f}")
    
    # Calculate and visualize class-wise dice scores
    class_wise_dice = calculate_class_wise_dice(model, test_gen)
    print("\nClass-wise Dice Scores:")
    for cls, score in class_wise_dice.items():
        print(f"{cls}: {score:.4f}")
    
    # Visualize results
    save_path = visualize_class_wise_dice(class_wise_dice, OUTPUT_DIRECTORY)
    print(f"\nSaved class-wise Dice scores visualization to: {save_path}")
    
    # Return model for further use if needed
    return model, history

if __name__ == "__main__":
    main()

