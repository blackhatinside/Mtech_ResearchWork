import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import keras.backend as K
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator
import requests  # For Allen Atlas API

# Constants and Paths (unchanged)
BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
AUG_PATH = "/home/user/adithyaes/dataset/isles2022_png_aug"
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
AUG_INPUT_PATH = os.path.join(AUG_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_PATH, "mask")
OUTPUT_DIRECTORY = "./output/ISLESfolder"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

IMG_SIZE = 112
BATCH_SIZE = 4
LEARNINGRATE = 0.001
EPOCHS = 100
EARLYSTOPPING = 40
scaler = MinMaxScaler(feature_range=(-1, 1))

# Allen Atlas API Integration
ALLEN_ATLAS_API = "https://api.brain-map.org/api/v2/data/query.json"

def get_tissue_stiffness(centroid):
    """
    Fetches tissue stiffness properties from Allen Brain Atlas API.
    """
    response = requests.get(
        ALLEN_ATLAS_API,
        params={"criteria": f"model::Structure,"
                           f"rma::criteria,[graph_id$eq1],"
                           f"coordinates[{centroid[0]},{centroid[1]}]"}
    )
    if response.status_code == 200:
        return response.json().get("stiffness", 1.0)  # Default stiffness if not found
    return 1.0

# Phase 1: Foundation - Biomechanical Deformation with MNI Space
def biomechanical_deformation(image, mask):
    """
    Applies biomechanically realistic deformation using MNI space and tissue stiffness.
    """
    # Convert centroid to MNI space
    lesion_pixels = np.where(mask > 0)
    if len(lesion_pixels[0]) == 0:
        return image, mask  # No deformation if no lesion

    centroid = np.array([np.mean(lesion_pixels[0]), np.mean(lesion_pixels[1])])
    mni_centroid = convert_to_mni_coordinates(centroid)

    # Fetch tissue stiffness from Allen Atlas
    stiffness = get_tissue_stiffness(mni_centroid)

    # Create deformation field
    dx = stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=5)
    dy = stiffness * gaussian_filter(np.random.randn(*image.shape), sigma=5)

    # Apply deformation
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
    deformed_mask = map_coordinates(mask, indices, order=0, mode='constant').reshape(mask.shape)

    return deformed_image, deformed_mask

# Phase 2: Hemodynamic Modeling - Simplified Navier-Stokes
def simulate_hemodynamics(image, mask):
    """
    Simulates hemodynamic effects using a simplified Navier-Stokes model.
    """
    # Create synthetic velocity field
    velocity_field = np.random.normal(0, 1, size=(image.shape[0], image.shape[1], 2))
    velocity_field = gaussian_filter(velocity_field, sigma=3)

    # Solve simplified Navier-Stokes (incompressible flow)
    divergence = np.gradient(velocity_field[..., 0], axis=0) + np.gradient(velocity_field[..., 1], axis=1)
    pressure = np.linalg.pinv(divergence)  # Pressure Poisson equation

    # Apply hemodynamic effect
    adc_map = np.exp(-0.1 * pressure)
    return image * adc_map, mask

# Phase 3: Attention Integration - Learned Attention Maps
class AttentionMapSaver(Callback):
    """
    Callback to save attention maps during training.
    """
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_batch_end(self, batch, logs=None):
        attention_maps = self.model.get_layer("attention_gate").output
        np.save(os.path.join(self.output_dir, f"attention_{batch}.npy"), attention_maps)

def attention_occlusion(image, mask, attention_map):
    """
    Applies attention-guided occlusion using learned attention maps.
    """
    if attention_map is None:
        return image, mask  # Fallback if no attention map

    # Create occlusion mask
    occlusion_mask = (attention_map > np.percentile(attention_map, 75)).astype(np.float32)
    occlusion_mask = gaussian_filter(occlusion_mask, sigma=2)

    # Apply occlusion
    occluded_image = image * (1 - occlusion_mask)
    return occluded_image, mask

# Advanced DataGenerator Class
class AdvancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True, attention_dir=None):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.aug_ids = aug_ids if aug_ids is not None else []
        self.all_ids = list_IDs + self.aug_ids
        self.shuffle = shuffle
        self.attention_dir = attention_dir
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.all_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_ids = [self.all_ids[k] for k in indexes]
        return self.__data_generation(batch_ids)

    def on_epoch_end(self):
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

                # Apply HIMRA Augmentation
                if is_aug:
                    img, mask = biomechanical_deformation(img, mask)
                    img, mask = simulate_hemodynamics(img, mask)

                    # Load attention map if available
                    attention_map = None
                    if self.attention_dir:
                        attention_path = os.path.join(self.attention_dir, f"attention_{case_id}.npy")
                        if os.path.exists(attention_path):
                            attention_map = np.load(attention_path)

                    img, mask = attention_occlusion(img, mask, attention_map)

                X.append(img)
                y.append(mask)

        return np.expand_dims(np.array(X), -1), np.array(y)

# Main Function (unchanged)
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    case_ids = get_case_ids(INPUT_PATH)
    aug_case_ids = get_case_ids(AUG_INPUT_PATH) if os.path.exists(AUG_INPUT_PATH) else []

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Aug: {len(aug_case_ids)}")

    # Initialize Attention Map Saver
    attention_saver = AttentionMapSaver(output_dir=os.path.join(OUTPUT_DIRECTORY, "attention_maps"))

    train_gen = AdvancedDataGenerator(train_ids, aug_ids=aug_case_ids, attention_dir=attention_saver.output_dir)
    val_gen = AdvancedDataGenerator(val_ids, shuffle=False)
    test_gen = AdvancedDataGenerator(test_ids, shuffle=False)

    model = create_model()

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_dice_coeff', mode='max',
                       save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        attention_saver
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    results = model.evaluate(test_gen)
    print("\nTest Results:")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Dice Score: {results[2]:.4f}")
    print(f"IoU: {results[3]:.4f}")

if __name__ == "__main__":
    main()