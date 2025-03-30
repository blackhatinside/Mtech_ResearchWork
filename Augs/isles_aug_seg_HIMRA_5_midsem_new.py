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

def dice_loss(y_true, y_pred):
	return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1 - dice_coeff(y_true, y_pred))

def single_dice_loss(y_true, y_pred):
    return 1.0 - dice_score(y_true, y_pred)

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
    """Improved function to grow small lesions with better control"""
    brain_mask = (image != -1)
    current_size = np.sum(mask > 0)
    
    # Don't process if no lesion or already large enough
    if current_size == 0 or current_size >= 50:
        return image, mask
        
    # Adjust target size based on current size
    # For very small lesions (<20 pixels), grow more conservatively
    if current_size < 20:
        target_size = min(current_size * 2, 40)  # Double size but cap at 40
    else:
        target_size = min(current_size * 1.5, 50)  # Increase by 50% but cap at 50
    
    iterations = 0
    grown_mask = mask.copy()
    grown_image = image.copy()
    
    while np.sum(grown_mask > 0) < target_size and iterations < 15:  # Increased max iterations
        # Use different kernel sizes based on lesion size
        if current_size < 20:
            kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for very small lesions
        else:
            kernel = np.ones((3, 3), np.uint8)
            
        dilated = ndimage.binary_dilation(grown_mask, kernel)
        # Only grow within brain boundaries
        dilated = dilated & brain_mask
        new_size = np.sum(dilated > 0)
        
        if new_size == np.sum(grown_mask > 0):  # If no growth, stop
            break
            
        # Update mask
        grown_mask = dilated
        
        # Adjust image intensity in grown regions with realistic gradient
        new_lesion_area = dilated & ~(mask > 0)
        if np.sum(mask > 0) > 0:
            # Create distance-based intensity gradient from original lesion
            distance = ndimage.distance_transform_edt(~mask)
            # Normalize distances in the new area
            if np.sum(new_lesion_area) > 0:
                dist_in_new_area = distance[new_lesion_area]
                max_dist = np.max(dist_in_new_area) if np.max(dist_in_new_area) > 0 else 1
                intensity_factor = 1 - (dist_in_new_area / (max_dist * 2))
                base_intensity = np.mean(image[mask > 0])
                # Apply graduated intensity based on distance
                grown_image[new_lesion_area] = base_intensity * intensity_factor
        
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

def large_lesion_enhancement(image, mask):
    """Special processing for large lesions (C5) to improve texture and boundary definition"""
    if np.sum(mask > 0) <= 200:  # Only apply to C5 lesions
        return image, mask
        
    # Create a distance map from the lesion boundary
    distance = ndimage.distance_transform_edt(mask)
    
    # Enhance the core of the lesion
    core = distance > 5  # Inner region
    boundary = mask & ~core  # Boundary region
    
    # Modify the image
    enhanced_image = image.copy()
    
    # Add subtle texture variation to large lesion cores
    if np.sum(core) > 0:
        texture = gaussian_filter(np.random.randn(*image.shape), sigma=3) * 0.1
        enhanced_image[core] = image[core] + texture[core]
    
    # Enhance boundary definition
    if np.sum(boundary) > 0:
        # Increase contrast at boundaries
        boundary_outside = ndimage.binary_dilation(mask) & ~mask
        if np.sum(boundary_outside) > 0:
            # Create a gradient at the boundary
            enhanced_image[boundary] = image[boundary] * 1.2  # Make boundary brighter
    
    return enhanced_image, mask

# Update the HIMRA augmentation pipeline to include these improvements
def improved_himra_augmentation(image, mask):
    """Complete HIMRA augmentation pipeline with improvements for C1 and C5"""
    # Determine lesion class
    lesion_size = np.sum(mask > 0)
    if lesion_size == 0:
        lesion_class = 0
    elif lesion_size <= 50:
        lesion_class = 1
    elif lesion_size <= 100:
        lesion_class = 2
    elif lesion_size <= 150:
        lesion_class = 3
    elif lesion_size <= 200:
        lesion_class = 4
    else:
        lesion_class = 5
    
    # For small lesions (C1), apply enhanced growth
    if lesion_class == 1:
        image, mask = grow_small_lesion(image, mask)
    
    # For all lesions, apply biomechanical deformation
    image, mask = biomechanical_deformation(image, mask, lesion_class)
    
    # For large lesions (C5), apply special enhancement
    if lesion_class == 5:
        image, mask = large_lesion_enhancement(image, mask)
    
    # For all lesions, apply hemodynamic simulation
    image, mask = simulate_hemodynamics(image, mask, lesion_class)
    
    # Apply attention occlusion with class-specific parameters
    if lesion_class in [1, 5]:  # Apply milder occlusion to C1 and C5
        occlusion_strength = 0.15  # Less occlusion 
    else:
        occlusion_strength = 0.3  # Regular occlusion
        
    # Modified attention_occlusion with strength parameter
    brain_mask = (image != -1)
    noise_scale = np.random.uniform(3, 6)
    attention_field = gaussian_filter(np.random.randn(*image.shape), sigma=noise_scale)
    attention_field = np.abs(attention_field)
    attention_field = (attention_field - attention_field.min()) / (attention_field.max() - attention_field.min())
    attention_field = attention_field * occlusion_strength + (1 - occlusion_strength)
    attention_field[~brain_mask] = 1.0
    modulated_image = image * attention_field
    modulated_image[~brain_mask] = -1.0
    
    return modulated_image, mask


# In[7]:


class BalancedHIMRADataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.aug_ids = aug_ids if aug_ids is not None else []
        self.all_ids = list_IDs + self.aug_ids
        self.shuffle = shuffle
        
        # Categorize cases by lesion class
        self.class_indices = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.categorize_by_lesion_class()
        
        # Generate balanced indices
        self.indexes = self.generate_balanced_indices()

    def categorize_by_lesion_class(self):
        # Assign each slice to a class based on lesion size
        for case_id in self.all_ids:
            is_aug = case_id in self.aug_ids
            mask_dir = AUG_MASK_PATH if is_aug else MASK_PATH
            
            mask_files = sorted([f for f in os.listdir(mask_dir)
                               if f.startswith(f'slice_{case_id}') and f.endswith('.png')])
            
            for f in mask_files:
                mask_path = os.path.join(mask_dir, f)
                mask = load_and_preprocess(mask_path, is_mask=True)
                
                lesion_size = np.sum(mask)
                if lesion_size == 0:
                    continue  # Skip slices with no lesion
                    
                lesion_class = 1 if lesion_size <= 50 else \
                              2 if lesion_size <= 100 else \
                              3 if lesion_size <= 150 else \
                              4 if lesion_size <= 200 else 5
                              
                self.class_indices[lesion_class].append((case_id, f))
    
    def generate_balanced_indices(self):
        # Find minimum number of samples per class
        min_samples = min([len(indices) for indices in self.class_indices.values() if indices])
        if min_samples == 0:
            min_samples = 1
            
        # Calculate sampling weights (more samples from underrepresented classes)
        weights = {
            1: 2.5,  # More C1 samples (significantly underperforming)
            2: 1.5,  # More C2 samples
            3: 1.0,  # Base rate for well-performing classes
            4: 1.0,
            5: 1.3   # Slightly more C5 samples
        }
        
        # Calculate number of samples to take from each class
        samples_per_class = {
            cls: max(min_samples, int(min_samples * weights[cls]))
            for cls in self.class_indices.keys()
        }
        
        # Generate balanced batch indices
        balanced_indices = []
        for cls, count in samples_per_class.items():
            if self.class_indices[cls]:
                # Sample with replacement if needed
                indices = np.random.choice(
                    len(self.class_indices[cls]), 
                    size=min(count, len(self.class_indices[cls])),
                    replace=count > len(self.class_indices[cls])
                )
                balanced_indices.extend([self.class_indices[cls][i] for i in indices])
        
        # Shuffle the balanced indices
        if self.shuffle:
            np.random.shuffle(balanced_indices)
            
        return balanced_indices
    
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch_indices)
    
    def on_epoch_end(self):
        # Regenerate balanced indices each epoch
        self.indexes = self.generate_balanced_indices()
    
    def __data_generation(self, batch_indices):
        X, y = [], []
        for case_id, file_name in batch_indices:
            is_aug = case_id in self.aug_ids
            input_dir = AUG_INPUT_PATH if is_aug else INPUT_PATH
            mask_dir = AUG_MASK_PATH if is_aug else MASK_PATH
            
            img_path = os.path.join(input_dir, file_name)
            mask_path = os.path.join(mask_dir, file_name)
            
            img = load_and_preprocess(img_path)
            mask = load_and_preprocess(mask_path, is_mask=True)
            
            # Determine lesion class for class-aware augmentation
            lesion_size = np.sum(mask)
            lesion_class = 1 if lesion_size <= 50 else \
                          2 if lesion_size <= 100 else \
                          3 if lesion_size <= 150 else \
                          4 if lesion_size <= 200 else 5
            
            # Apply improved HIMRA augmentation
            if is_aug:
                img, mask = improved_himra_augmentation(img, mask)
            
            X.append(img)
            y.append(mask)
            
        return np.expand_dims(np.array(X), -1), np.array(y)


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


def class_weighted_dice_loss(y_true, y_pred):
    """
    A dimension-agnostic weighted dice loss function that handles tensor dimensionality safely.
    """
    # Flatten y_true and y_pred to 2D tensors
    # This works for both 3D and 4D inputs without needing to check rank
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # Becomes [batch_size, flattened_rest]
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    # Calculate lesion size per sample (sum across all dimensions except batch)
    lesion_size = tf.reduce_sum(y_true_flat, axis=1)
    
    # Assign weights based on lesion size
    weights = tf.ones_like(lesion_size)
    
    # Higher weights for smaller lesions (C1, C2)
    weights = tf.where(lesion_size <= 50, 2.5, weights)  # C1
    weights = tf.where((lesion_size > 50) & (lesion_size <= 100), 1.5, weights)  # C2
    
    # Higher weight for very large lesions (C5)
    weights = tf.where(lesion_size > 200, 1.3, weights)  # C5
    
    # Calculate intersection for Dice coefficient
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    
    # Calculate dice scores per sample
    dice_scores = (2. * intersection + 1) / (
        tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1) + 1)
    
    # Apply weights to dice scores (1 - dice_score to convert to loss)
    weighted_dice = weights * (1 - dice_scores)
    
    # Return mean loss
    return tf.reduce_mean(weighted_dice)

def create_enhanced_model():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))
    
    # Additional input preprocessing layer for enhancing small features
    x = Conv2D(32, 1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Encoder with deep supervision
    e1 = conv_block(x, 32)
    e1 = conv_block(e1, 32)
    skip1 = e1
    x = MaxPooling2D()(e1)
    
    e2 = conv_block(x, 64)
    e2 = conv_block(e2, 64)
    skip2 = e2
    x = MaxPooling2D()(e2)
    
    e3 = conv_block(x, 128)
    e3 = conv_block(e3, 128)
    skip3 = e3
    x = MaxPooling2D()(e3)
    
    e4 = conv_block(x, 256)
    e4 = conv_block(e4, 256)
    skip4 = e4
    x = MaxPooling2D()(e4)
    
    # Bridge with dilated convolutions to capture multi-scale context
    x = Conv2D(512, 3, padding='same', dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, 3, padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, 3, padding='same', dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder with attention gates and deep supervision
    x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    skip4 = attention_gate(skip4, x, 256)
    x = concatenate([x, skip4])
    d4 = conv_block(x, 256)
    d4 = conv_block(d4, 256)
    
    x = Conv2DTranspose(128, 3, strides=2, padding='same')(d4)
    skip3 = attention_gate(skip3, x, 128)
    x = concatenate([x, skip3])
    d3 = conv_block(x, 128)
    d3 = conv_block(d3, 128)
    
    x = Conv2DTranspose(64, 3, strides=2, padding='same')(d3)
    skip2 = attention_gate(skip2, x, 64)
    x = concatenate([x, skip2])
    d2 = conv_block(x, 64)
    d2 = conv_block(d2, 64)
    
    x = Conv2DTranspose(32, 3, strides=2, padding='same')(d2)
    skip1 = attention_gate(skip1, x, 32)
    x = concatenate([x, skip1])
    d1 = conv_block(x, 32)
    d1 = conv_block(d1, 32)
    
    # Deep supervision outputs
    output1 = Conv2D(1, 1, activation='sigmoid', name='output1')(d1)
    output2 = Conv2D(1, 1, activation='sigmoid', name='output2')(
        Conv2DTranspose(32, 3, strides=2, padding='same')(d2)
    )
    output3 = Conv2D(1, 1, activation='sigmoid', name='output3')(
        Conv2DTranspose(32, 3, strides=4, padding='same')(d3)
    )
    output4 = Conv2D(1, 1, activation='sigmoid', name='output4')(
        Conv2DTranspose(32, 3, strides=8, padding='same')(d4)
    )
    
    # Multi-scale feature integration (for small lesion detection)
    # First, upsample all deep features to full resolution
    d1_full = d1
    d2_full = Conv2DTranspose(64, 3, strides=2, padding='same')(d2)
    d3_full = Conv2DTranspose(128, 3, strides=4, padding='same')(d3)
    d4_full = Conv2DTranspose(256, 3, strides=8, padding='same')(d4)
    
    # Second, reduce channel count for feature concatenation
    d1_reduced = Conv2D(32, 1, padding='same')(d1_full)
    d2_reduced = Conv2D(32, 1, padding='same')(d2_full)
    d3_reduced = Conv2D(32, 1, padding='same')(d3_full)
    d4_reduced = Conv2D(32, 1, padding='same')(d4_full)
    
    # Concatenate all features for final prediction
    multi_scale = concatenate([d1_reduced, d2_reduced, d3_reduced, d4_reduced])
    multi_scale = conv_block(multi_scale, 64)
    
    # Final output
    output_final = Conv2D(1, 1, activation='sigmoid', name='final_output')(multi_scale)
    
    # Create model with multiple outputs
    model = Model(inputs, [output_final, output1, output2, output3, output4])
    
    # Custom loss weighting based on outputs
    def weighted_loss(y_true, y_pred):
        # Main output gets highest weight
        final_loss = dice_loss(y_true, y_pred)
        return final_loss
    
    # Custom multi-output loss
    def multi_output_loss(y_true, y_pred_list):
        # Weight for each output
        weights = [0.7, 0.075, 0.075, 0.075, 0.075]
        loss = 0
        for i, pred in enumerate(y_pred_list):
            loss += weights[i] * dice_loss(y_true, pred)
        return loss
    
    # Compile model with multiple outputs and custom loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
        loss={
            'final_output': class_weighted_dice_loss,
            'output1': dice_loss,
            'output2': dice_loss,
            'output3': dice_loss,
            'output4': dice_loss
        },
        loss_weights={
            'final_output': 0.7,
            'output1': 0.075,
            'output2': 0.075,
            'output3': 0.075,
            'output4': 0.075
        },
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

train_gen = BalancedHIMRADataGenerator(train_ids, aug_ids=aug_case_ids)
val_gen = BalancedHIMRADataGenerator(val_ids, shuffle=False)
test_gen = BalancedHIMRADataGenerator(test_ids, shuffle=False)

model = create_enhanced_model()

print("Model Summary: ", model.summary())

# Implement two-stage training: first focus on small lesions, then full dataset
# Stage 1: Small lesion focus
callbacks_stage1 = [
    ModelCheckpoint('stage1_model.h5', monitor='val_final_output_dice_coeff', 
                    mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_final_output_loss', patience=30, 
                 restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_final_output_loss', factor=0.5, patience=10, verbose=1)
]

print("Stage 1: Training with focus on small lesions...")
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,  # First stage with fewer epochs
    callbacks=callbacks_stage1,
    workers=4,
    use_multiprocessing=True
)

# Stage 2: Fine-tune on full dataset with reduced learning rate
model.load_weights('stage1_model.h5')
K.set_value(model.optimizer.learning_rate, LEARNINGRATE * 0.5)

callbacks_stage2 = [
    ModelCheckpoint('best_model.h5', monitor='val_final_output_dice_coeff', 
                   mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_final_output_loss', patience=EARLYSTOPPING, 
                 restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_final_output_loss', factor=0.5, patience=5, verbose=1)
]

print("Stage 2: Fine-tuning on full dataset...")
history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks_stage2,
    workers=4,
    use_multiprocessing=True,
    initial_epoch=len(history_stage1.history['loss'])  # Continue from stage 1
)

results = model.evaluate(test_gen)

print("\nGenerating Training Metrics Visualization...")
plot_training_metrics(history_stage2)


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


# In[ ]:





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

# Visualize
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
plt.show()


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

# Visualize
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
plt.show()


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

# Visualize
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
plt.show()


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

# Visualize
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
plt.show()


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

# Visualize
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
plt.show()


# In[ ]:





# In[23]:


# Load specific image
test_image_path = os.path.join(INPUT_PATH, "slice_sub-strokecase0240_0068.png")
test_mask_path = os.path.join(MASK_PATH, "slice_sub-strokecase0240_0068.png")

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

# Visualize
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
plt.show()


# In[24]:


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

# Visualize
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
plt.show()


# In[25]:


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


def apply_class_specific_postprocessing(image, prediction):
    """
    Apply class-specific post-processing to improve segmentation results
    """
    # Determine approximate lesion size in the prediction
    pred_size = np.sum(prediction > 0.5)
    
    # Create brain mask
    brain_mask = (image != -1)
    
    # Enhanced prediction
    enhanced_pred = prediction.copy()
    
    # Class 1 (small lesions): Remove isolated pixels, keep only the largest connected component
    if pred_size <= 50 and pred_size > 0:
        # Binary prediction
        binary_pred = prediction > 0.5
        
        # Label connected components
        labeled, num_components = ndimage.label(binary_pred)
        
        if num_components > 1:
            # Find sizes of all components
            component_sizes = np.bincount(labeled.flatten())[1:]
            
            # Keep only the largest component
            largest_component = np.argmax(component_sizes) + 1
            enhanced_pred = (labeled == largest_component).astype(np.float32)
            
            # Slightly dilate the small lesion to improve recall
            enhanced_pred = ndimage.binary_dilation(enhanced_pred, 
                                                   structure=np.ones((2,2))).astype(np.float32)
            
            # Apply brain mask
            enhanced_pred = enhanced_pred * brain_mask
    
    # Class 2-3 (medium lesions): Smooth boundaries
    elif pred_size > 50 and pred_size <= 150:
        # Binary prediction
        binary_pred = prediction > 0.5
        
        # Apply small opening to remove noise, then small closing to fill gaps
        enhanced_pred = ndimage.binary_opening(binary_pred, 
                                             structure=np.ones((2,2))).astype(np.float32)
        enhanced_pred = ndimage.binary_closing(enhanced_pred, 
                                             structure=np.ones((2,2))).astype(np.float32)
        
        # Apply brain mask
        enhanced_pred = enhanced_pred * brain_mask
    
    # Class 4-5 (large lesions): Fix boundaries and fill holes
    elif pred_size > 150:
        # Binary prediction
        binary_pred = prediction > 0.5
        
        # Fill holes
        filled_pred = ndimage.binary_fill_holes(binary_pred).astype(np.float32)
        
        # For very large regions, remove small isolated parts
        if pred_size > 200:
            # Label connected components
            labeled, num_components = ndimage.label(filled_pred)
            
            if num_components > 1:
                # Find sizes of all components
                component_sizes = np.bincount(labeled.flatten())[1:]
                
                # Identify small components (less than 5% of the largest component)
                largest_size = np.max(component_sizes)
                small_components = np.where(component_sizes < 0.05 * largest_size)[0] + 1
                
                # Remove small components
                for comp in small_components:
                    filled_pred[labeled == comp] = 0
        
        enhanced_pred = filled_pred
        
        # Apply brain mask
        enhanced_pred = enhanced_pred * brain_mask
    
    return enhanced_pred

# Test the post-processing
def evaluate_with_postprocessing(model, test_gen):
    dice_before = []
    dice_after = []
    class_dice_before = {f'C{i}': [] for i in range(1, 6)}
    class_dice_after = {f'C{i}': [] for i in range(1, 6)}
    
    for batch_x, batch_y in test_gen:
        predictions = model.predict(batch_x)[0] if isinstance(model.output, list) else model.predict(batch_x)
        predictions = (predictions > 0.5).astype(np.float32)
        
        for i in range(len(batch_y)):
            image = batch_x[i, :, :, 0]
            true_mask = batch_y[i]
            pred_mask = predictions[i, :, :, 0]
            
            # Calculate lesion size and class
            lesion_size = np.sum(true_mask)
            if lesion_size == 0:
                continue
                
            lesion_class = 1 if lesion_size <= 50 else \
                          2 if lesion_size <= 100 else \
                          3 if lesion_size <= 150 else \
                          4 if lesion_size <= 200 else 5
            
            # Calculate dice before post-processing
            dice_before_pp = numpy_dice_coeff(true_mask, pred_mask)
            dice_before.append(dice_before_pp)
            class_dice_before[f'C{lesion_class}'].append(dice_before_pp)
            
            # Apply post-processing
            enhanced_pred = apply_class_specific_postprocessing(image, pred_mask)
            
            # Calculate dice after post-processing
            dice_after_pp = numpy_dice_coeff(true_mask, enhanced_pred)
            dice_after.append(dice_after_pp)
            class_dice_after[f'C{lesion_class}'].append(dice_after_pp)
    
    # Calculate mean dice scores
    mean_dice_before = np.mean(dice_before)
    mean_dice_after = np.mean(dice_after)
    
    # Calculate class-wise mean dice scores
    class_mean_before = {cls: np.mean(scores) if scores else 0 
                         for cls, scores in class_dice_before.items()}
    class_mean_after = {cls: np.mean(scores) if scores else 0 
                        for cls, scores in class_dice_after.items()}
    
    print("\nOverall Results:")
    print(f"Mean Dice Score Before Post-processing: {mean_dice_before:.4f}")
    print(f"Mean Dice Score After Post-processing: {mean_dice_after:.4f}")
    
    print("\nClass-wise Results:")
    for cls in sorted(class_mean_before.keys()):
        print(f"{cls}: Before = {class_mean_before[cls]:.4f}, After = {class_mean_after[cls]:.4f}, " + 
              f"Improvement = {class_mean_after[cls] - class_mean_before[cls]:.4f}")
    
    return class_mean_before, class_mean_after

# Test the post-processing
print("Evaluating with class-specific post-processing...")
class_results_before, class_results_after = evaluate_with_postprocessing(model, test_gen)

def visualize_class_balance_improvement(before, after, output_dir):
    """
    Visualize the improvement in class balance before and after our changes
    """
    classes = sorted(before.keys())
    before_values = [before[cls] for cls in classes]
    after_values = [after[cls] for cls in classes]
    
    # Standard deviation as a measure of balance
    std_before = np.std(before_values)
    std_after = np.std(after_values)
    
    # Min-max ratio as another measure of balance
    min_max_ratio_before = min(before_values) / max(before_values)
    min_max_ratio_after = min(after_values) / max(after_values)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(classes))
    
    bar1 = plt.bar(index, before_values, bar_width, label='Before', color='skyblue')
    bar2 = plt.bar(index + bar_width, after_values, bar_width, label='After', color='lightgreen')
    
    plt.xlabel('Lesion Size Class')
    plt.ylabel('Mean Dice Score')
    plt.title('Class-wise Dice Scores: Before vs After Improvements')
    plt.xticks(index + bar_width/2, classes)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(before_values, after_values)):
        plt.text(i - 0.05, v1 + 0.02, f"{v1:.3f}", ha='center', fontsize=9)
        plt.text(i + bar_width - 0.05, v2 + 0.02, f"{v2:.3f}", ha='center', fontsize=9)
    
    # Add balance metrics to plot
    plt.figtext(0.15, 0.01, f"Standard Deviation - Before: {std_before:.3f}, After: {std_after:.3f}",
               fontsize=12, ha='left')
    plt.figtext(0.15, 0.05, f"Min/Max Ratio - Before: {min_max_ratio_before:.3f}, After: {min_max_ratio_after:.3f}",
               fontsize=12, ha='left')
    
    # Save and show plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'class_balance_improvement_{timestamp}.png')
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path

# Visualize the class balance improvement
improvement_viz_path = visualize_class_balance_improvement(
    class_results_before, class_results_after, OUTPUT_DIRECTORY
)
print(f"\nSaved class balance improvement visualization to: {improvement_viz_path}")
