#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from scipy import ndimage
import matplotlib.pyplot as plt
from datetime import datetime

# Constants and Paths for ISLES2022
BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY = os.path.join("./output/ISLES22folder", timestamp)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Model parameters
IMG_SIZE = 112
BATCH_SIZE = 4
LEARNINGRATE = 0.0001
EPOCHS = 100
EARLYSTOPPING = 60
scaler = MinMaxScaler(feature_range=(-1, 1))

# [Keep all the metrics and loss functions as they are]
def dice_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    return (intersection + 0.1) / (union - intersection + 0.1)

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

# Modified function to extract case IDs from the new naming convention
def get_case_ids(path):
    """Extract unique case IDs from files in input directory"""
    input_dir = os.path.join(path, "input")
    if not os.path.exists(input_dir):
        return []
    
    case_ids = set()
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Extract case ID from pattern: slice_sub-strokecaseXXXX_YYYY.png
            match = re.match(r'slice_sub-(strokecase\d+)_\d+\.png', filename)
            if match:
                case_ids.add(match.group(1))
    
    return sorted(list(case_ids))

# Modified DataGenerator for new file structure
class SimpleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sample_ids, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.sample_ids = sample_ids
        self.shuffle = shuffle
        self.slice_indices = []
        self._build_slice_indices()
        self.__on_epoch_end()
        
    def _build_slice_indices(self):
        """Build a list of (sample_id, slice_num) tuples"""
        self.slice_indices = []
        
        input_dir = os.path.join(BASE_PATH, "input")
        
        for sample_id in self.sample_ids:
            # Find all files for this case
            pattern = f'slice_sub-{sample_id}_*.png'
            files = [f for f in os.listdir(input_dir) if re.match(f'slice_sub-{sample_id}_\\d+\\.png', f)]
            
            for f in files:
                # Extract slice number from filename
                match = re.match(f'slice_sub-{sample_id}_(\\d+)\\.png', f)
                if match:
                    slice_num = match.group(1)
                    self.slice_indices.append((sample_id, slice_num))
    
    def __len__(self):
        return int(np.floor(len(self.slice_indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_slice_indices = self.slice_indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch_slice_indices)
    
    def __on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.slice_indices)
    
    def __data_generation(self, batch_slice_indices):
        X, y = [], []
        for sample_id, slice_num in batch_slice_indices:
            # Construct filenames based on new naming convention
            input_filename = f'slice_sub-{sample_id}_{slice_num}.png'
            mask_filename = f'slice_sub-{sample_id}_{slice_num}.png'
            
            img_path = os.path.join(BASE_PATH, "input", input_filename)
            mask_path = os.path.join(BASE_PATH, "mask", mask_filename)
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            img = load_and_preprocess(img_path)
            mask = load_and_preprocess(mask_path, is_mask=True)
            
            X.append(img)
            y.append(mask)
        
        return np.expand_dims(np.array(X), -1), np.expand_dims(np.array(y), -1)

# [Keep the model architecture and utility functions as they are]
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

    x = conv_block(x, 512)

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

# [Keep all utility functions as they are]
def plot_training_metrics(history):
    metrics = ['loss', 'dice_coeff', 'iou', 'accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ISLES2022 Training Metrics Over Epochs (No Augmentation)', fontsize=16)

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
    predictions = (predictions > 0.5).astype(np.float32)

    plt.figure(figsize=(15, 20))

    for idx in range(min(num_samples, len(test_images))):
        plt.subplot(num_samples, 2, 2*idx + 1)
        plt.imshow(test_images[idx, :, :, 0], cmap='gray')
        plt.title(f'Input Image {idx+1}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*idx + 2)
        plt.imshow(predictions[idx, :, :, 0], cmap='gray')
        dice_score = numpy_dice_coeff(true_masks[idx, :, :, 0], predictions[idx, :, :, 0])
        iou_score = numpy_iou(true_masks[idx, :, :, 0], predictions[idx, :, :, 0])
        plt.title(f'Predicted Segmentation {idx+1}\n' +
                 f'Dice: {dice_score:.4f}, IoU: {iou_score:.4f}')
        plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'segmentation_results_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_class_wise_dice(model, test_gen):
    class_dice_scores = {f'C{i}': [] for i in range(1, 6)}
    
    for batch_x, batch_y in test_gen:
        predictions = model.predict(batch_x)
        predictions = (predictions > 0.5).astype(np.float32)
        
        for i in range(len(batch_y)):
            true_mask = batch_y[i, :, :, 0]
            pred_mask = predictions[i, :, :, 0]
            
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
    plt.title('ISLES2022 Class-wise Dice Scores (No Augmentation)')
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'class_wise_dice_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return save_path

# Main execution
if __name__ == "__main__":
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Get sample IDs from the modified dataset structure
    sample_ids = get_case_ids(BASE_PATH)
    
    print(f"Found {len(sample_ids)} unique cases: {sample_ids}")
    
    # Split into train, validation, and test sets
    train_ids, test_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Create data generators without augmentation
    train_gen = SimpleDataGenerator(train_ids)
    val_gen = SimpleDataGenerator(val_ids, shuffle=False)
    test_gen = SimpleDataGenerator(test_ids, shuffle=False)

    # Create and compile model
    model = create_model()
    
    print("Model Summary:")
    model.summary()

    # Define callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIRECTORY, 'best_model.h5'),
        monitor='val_dice_coeff',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_dice_coeff',
        patience=EARLYSTOPPING,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Plot training metrics
    plot_training_metrics(history)
    
    # Evaluate on test set
    test_loss, test_acc, test_dice, test_iou = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Dice Coefficient: {test_dice:.4f}")
    print(f"IoU: {test_iou:.4f}")
    
    # Visualize segmentation results
    visualize_segmentation_results(model, test_gen)
    
    # Calculate and visualize class-wise dice scores
    class_dice_scores = calculate_class_wise_dice(model, test_gen)
    print("\nClass-wise Dice Scores:")
    for cls, score in class_dice_scores.items():
        print(f"{cls}: {score:.4f}")
    
    visualize_class_wise_dice(class_dice_scores, OUTPUT_DIRECTORY)
    
    # Save the final model
    model.save(os.path.join(OUTPUT_DIRECTORY, 'final_model.h5'))
    print(f"\nModel and results saved to: {OUTPUT_DIRECTORY}")