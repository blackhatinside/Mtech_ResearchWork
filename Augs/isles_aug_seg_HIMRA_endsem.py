#!/usr/bin/env python
# coding: utf-8

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
import matplotlib.pyplot as plt
from datetime import datetime

# Constants and Paths
BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
AUG_PATH = "/home/user/adithyaes/dataset/isles2022_png_aug"
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
AUG_INPUT_PATH = os.path.join(AUG_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_PATH, "mask")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = os.path.join("./output/ISLES22folder", timestamp)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

IMG_SIZE = 112
BATCH_SIZE = 4
LEARNINGRATE = 0.001
EPOCHS = 100
EARLYSTOPPING = 60
scaler = MinMaxScaler(feature_range=(-1, 1))

# Metrics
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

def hybrid_loss(y_true, y_pred):
    lesion_size = K.sum(y_true)
    weight = tf.where(lesion_size < 50, 2.0, 1.0)
    
    dice_loss = single_dice_loss(y_true, y_pred)
    focal_loss = binary_focal_loss(gamma=2.5, alpha=0.3)(y_true, y_pred)
    bce_loss = binary_crossentropy_loss(y_true, y_pred)
    
    combined_loss = weight * (0.5 * dice_loss + 0.4 * focal_loss + 0.1 * bce_loss)
    return combined_loss

# Data Loading
def load_and_preprocess(file_path, is_mask=False):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if not is_mask:
        img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    else:
        img = img / 255.0
    return img

# HIMRA Augmentation Functions
def grow_small_lesion(image, mask, target_size=45):
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
        dilated = dilated & brain_mask
        new_size = np.sum(dilated > 0)
        
        if new_size > 12:
            break
            
        grown_mask = dilated
        new_lesion_area = dilated & ~(mask > 0)
        grown_image[new_lesion_area] = np.mean(image[mask > 0])
        
        iterations += 1
    
    return grown_image, grown_mask.astype(mask.dtype)

def biomechanical_deformation(image, mask, lesion_class):
    if lesion_class == 1:
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
        1: (0.5, 1.7),
        2: (0.4, 1.6), 
        3: (0.5, 1.5),
        4: (0.6, 1.4),
        5: (0.7, 1.3)
    }
    min_contrast, max_contrast = contrasts[lesion_class]
    
    brain_mask = (image != -1)
    vessel_mask = gaussian_filter(np.random.binomial(1, 0.03, size=image.shape), sigma=1)
    perfusion_map = gaussian_filter(np.random.normal(1.0, 0.3, size=image.shape), sigma=2)
    
    perfusion_map[vessel_mask > 0.5] *= np.random.uniform(max_contrast - 0.2, max_contrast)
    
    if np.sum(mask > 0) > 0:
        lesion_area = mask > 0
        perfusion_map[lesion_area] = np.random.uniform(1.3, 1.5)
        
        distance = ndimage.distance_transform_edt(1 - mask)
        penumbra = (distance < 7) & (mask == 0) & brain_mask
        perfusion_map[penumbra] *= np.random.uniform(min_contrast + 0.3, min_contrast + 0.5)
    
    enhanced_image = image.copy()
    enhanced_image[brain_mask] = image[brain_mask] * perfusion_map[brain_mask]
    enhanced_image[~brain_mask] = -1.0
    
    return np.clip(enhanced_image, -1, 1), mask

def attention_occlusion(image, mask):
    brain_mask = (image != -1)
    
    noise_scale = np.random.uniform(3, 6)
    attention_field = gaussian_filter(np.random.randn(*image.shape), sigma=noise_scale)
    attention_field = np.abs(attention_field)
    attention_field = (attention_field - attention_field.min()) / (attention_field.max() - attention_field.min())
    attention_field = attention_field * 0.3 + 0.7
    
    attention_field[~brain_mask] = 1.0
    modulated_image = image * attention_field
    modulated_image[~brain_mask] = image[~brain_mask]
    
    return modulated_image, mask

# Add this function before the HIMRADataGenerator class
def has_lesion(mask_path):
    """Check if mask contains any lesion pixels"""
    if not os.path.exists(mask_path):
        return False
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.sum(mask > 0) > 0

# Then modify the HIMRADataGenerator class
class HIMRADataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.aug_ids = aug_ids if aug_ids is not None else []
        self.all_ids = list_IDs + self.aug_ids
        self.shuffle = shuffle
        self.file_pairs = self._get_valid_file_pairs()  # New method to filter slices
        self.__on_epoch_end()
        self.class_weights = {1: 2.0, 2: 1.5, 3: 1.2, 4: 1.0, 5: 1.0}

    def _get_valid_file_pairs(self):
        """Get all valid image-mask pairs that contain lesions"""
        valid_pairs = []
        
        for case_id in self.all_ids:
            is_aug = case_id in self.aug_ids
            input_dir = AUG_INPUT_PATH if is_aug else INPUT_PATH
            mask_dir = AUG_MASK_PATH if is_aug else MASK_PATH
            
            # Find all input files for this case
            input_files = sorted([f for f in os.listdir(input_dir)
                                if f.startswith(f'slice_{case_id}') and f.endswith('.png')])

            for f in input_files:
                # Check corresponding mask for lesions
                mask_path = os.path.join(mask_dir, f)
                
                if has_lesion(mask_path):
                    valid_pairs.append({
                        'case_id': case_id,
                        'filename': f,
                        'is_aug': is_aug
                    })
        
        return valid_pairs

    def __len__(self):
        return int(np.floor(len(self.file_pairs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_pairs = [self.file_pairs[k] for k in indexes]
        return self.__data_generation(batch_pairs)

    def __on_epoch_end(self):
        self.indexes = np.arange(len(self.file_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_pairs):
        X, y = [], []
        for pair in batch_pairs:
            case_id = pair['case_id']
            f = pair['filename']
            is_aug = pair['is_aug']
            
            input_dir = AUG_INPUT_PATH if is_aug else INPUT_PATH
            mask_dir = AUG_MASK_PATH if is_aug else MASK_PATH

            img_path = os.path.join(input_dir, f)
            mask_path = os.path.join(mask_dir, f)

            img = load_and_preprocess(img_path)
            mask = load_and_preprocess(mask_path, is_mask=True)

            lesion_size = np.sum(mask)
            lesion_class = 1 if lesion_size < 50 else 2 if lesion_size < 100 else 3 if lesion_size < 150 else 4 if lesion_size < 200 else 5

            if is_aug:
                img, mask = biomechanical_deformation(img, mask, lesion_class)
                img, mask = simulate_hemodynamics(img, mask, lesion_class)
                
                self.__save_augmented_data(img, mask, f)

            X.append(img)
            y.append(mask)

        return np.expand_dims(np.array(X), -1), np.expand_dims(np.array(y), -1)

    def __save_augmented_data(self, image, mask, filename):
        augmented_image_dir = os.path.join(OUTPUT_DIRECTORY, "augmented_images")
        augmented_mask_dir = os.path.join(OUTPUT_DIRECTORY, "augmented_masks")

        os.makedirs(augmented_image_dir, exist_ok=True)
        os.makedirs(augmented_mask_dir, exist_ok=True)

        image_filename = os.path.join(augmented_image_dir, filename)
        mask_filename = os.path.join(augmented_mask_dir, filename.replace('slice', 'mask'))

        cv2.imwrite(image_filename, image * 255)
        cv2.imwrite(mask_filename, mask * 255)

# Model Architecture
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

def create_model():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))

    x = conv_block(inputs, 16)
    skip1 = x
    x = MaxPooling2D()(x)

    x = conv_block(x, 32)
    skip2 = x
    x = MaxPooling2D()(x)

    x = conv_block(x, 64)
    skip3 = x
    x = MaxPooling2D()(x)

    x = conv_block(x, 128)
    skip4 = x
    x = MaxPooling2D()(x)

    x = conv_block(x, 256)

    x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = attention_gate(skip4, x, 128)
    x = concatenate([x, skip4])
    x = conv_block(x, 128)

    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = attention_gate(skip3, x, 64)
    x = concatenate([x, skip3])
    x = conv_block(x, 64)

    x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = attention_gate(skip2, x, 32)
    x = concatenate([x, skip2])
    x = conv_block(x, 32)

    x = Conv2DTranspose(16, 3, strides=2, padding='same')(x)
    x = attention_gate(skip1, x, 16)
    x = concatenate([x, skip1])
    x = conv_block(x, 16)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
        loss=hybrid_loss,
        metrics=['accuracy', dice_coeff, iou]
    )

    return model

# Utility Functions
def get_case_ids(path):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    return sorted(list({f.split('_')[1] for f in files}))

def numpy_dice_coeff(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

def plot_training_metrics(history):
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

# Class-wise Dice Score Calculation
def calculate_class_wise_dice(model, test_gen):
    class_dice_scores = {f'C{i}': [] for i in range(1, 6)}
    
    for batch_x, batch_y in test_gen:
        predictions = model.predict(batch_x)
        predictions = (predictions > 0.5).astype(np.float32)
        
        for i in range(len(batch_y)):
            true_mask = batch_y[i]
            pred_mask = predictions[i,:,:,0]
            
            lesion_size = np.sum(true_mask)
            
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
            
            dice = numpy_dice_coeff(true_mask, pred_mask)
            class_dice_scores[class_name].append(dice)
    
    mean_dice_scores = {cls: np.mean(scores) if scores else 0 for cls, scores in class_dice_scores.items()}
    
    return mean_dice_scores

def visualize_class_wise_dice(scores, output_dir):
    classes = sorted(scores.keys())
    values = [scores[cls] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, values, color='skyblue')
    plt.xlabel('Lesion Size Class')
    plt.ylabel('Mean Dice Score')
    plt.title('Class-wise Dice Scores')
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'class_wise_dice_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path

# Main Training Loop
if __name__ == "__main__":
    # GPU Configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Data Preparation
    case_ids = get_case_ids(INPUT_PATH)
    aug_case_ids = get_case_ids(AUG_INPUT_PATH) if os.path.exists(AUG_INPUT_PATH) else []

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Aug: {len(aug_case_ids)}")

    # Data Generators
    train_gen = HIMRADataGenerator(train_ids, aug_ids=aug_case_ids)
    val_gen = HIMRADataGenerator(val_ids, shuffle=False)
    test_gen = HIMRADataGenerator(test_ids, shuffle=False)

    # Model Creation
    model = create_model()
    print("Model Summary: ", model.summary())

    # Callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(OUTPUT_DIRECTORY, 'best_model.h5'), 
                       monitor='val_dice_coeff', mode='max',
                       save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING, 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # Training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    # Evaluation
    results = model.evaluate(test_gen)
    print("\nTest Results:")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Dice Score: {results[2]:.4f}")
    print(f"IoU: {results[3]:.4f}")

    # Save Training Metrics
    plot_training_metrics(history)

    # Calculate and Visualize Class-wise Dice Scores
    class_wise_dice = calculate_class_wise_dice(model, test_gen)
    print("\nClass-wise Dice Scores:")
    for cls, score in class_wise_dice.items():
        print(f"{cls}: {score:.4f}")

    save_path = visualize_class_wise_dice(class_wise_dice, OUTPUT_DIRECTORY)
    print(f"\nAll outputs saved to: {OUTPUT_DIRECTORY}")