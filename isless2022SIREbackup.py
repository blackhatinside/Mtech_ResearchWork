# Imports
import nibabel as nib
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import concatenate
import types

scaler = MinMaxScaler()

# Verify imports
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__
print(list(imports()))

# Paths
TRAIN_DATASET_PATH = '/home/user/Tf_script/dataset/ISLES_2022/rawdata/'
TRAINMask_DATASET_PATH = '/home/user/Tf_script/dataset/ISLES_2022/derivatives/'

# Get dataset details
trainfolders = os.listdir(TRAIN_DATASET_PATH)
train_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
train_directory_startindex = train_directories[0].find("sub")
train_ids = sorted([train_directories[i][train_directory_startindex:] for i in range(len(train_directories))])

maskfolders = os.listdir(TRAINMask_DATASET_PATH)
mask_directories = [f.path for f in os.scandir(TRAINMask_DATASET_PATH) if f.is_dir()]
mask_id_startindex = mask_directories[0].find("sub")
mask_ids = sorted([mask_directories[i][mask_id_startindex:] for i in range(len(mask_directories))])

from sklearn.model_selection import train_test_split
train_test_ids, val_ids, train_test_mask, val_mask = train_test_split(train_ids, mask_ids, test_size=0.15, random_state=42)
train_ids, test_ids, train_mask, test_mask = train_test_split(train_test_ids, train_test_mask, test_size=0.15, random_state=42)

# Performance metrics
def dice_coeff(y_true, y_pred):
    y_true_new = K.flatten(y_true)
    y_pred_new = K.flatten(y_pred)
    denominator = K.sum(y_true_new) + K.sum(y_pred_new)
    numerator = K.sum(y_true_new * y_pred_new)
    return (2 * numerator + 1) / (denominator + 1)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred):
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    return (intersec + 0.1) / (union - intersec + 0.1)

IMG_SIZE = 112

# Data generator class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = []
        y = []
        for i in Batch_ids:
            case_path = os.path.join(TRAIN_DATASET_PATH, i, 'ses-0001/dwi')
            nii_files = [f for f in os.listdir(case_path) if f.endswith('.nii.gz')]

            if not nii_files:
                print(f"No .nii.gz files found in {case_path}")
                continue

            file_path = os.path.join(case_path, nii_files[0])
            dwi = nib.load(file_path).get_fdata()
            dwi = scaler.fit_transform(dwi.reshape(-1, dwi.shape[-1])).reshape(dwi.shape)
            slices = dwi.shape[2]
            X_case = np.zeros((slices, IMG_SIZE, IMG_SIZE, 1))

            case_path2 = os.path.join(TRAINMask_DATASET_PATH, i)
            data_path_2 = os.path.join(case_path2 + '/ses-0001', f'{i}_ses-0001_msk.nii.gz')

            if not os.path.exists(data_path_2):
                print(f"Mask file not found: {data_path_2}")
                continue

            msk = nib.load(data_path_2).get_fdata()
            msk_slices = msk.shape[2]
            y_case = np.zeros((msk_slices, IMG_SIZE, IMG_SIZE))

            for j in range(slices):
                X_case[j, :, :, 0] = cv2.resize(dwi[:, :, j], (IMG_SIZE, IMG_SIZE))
                y_case[j, :, :] = cv2.resize(msk[:, :, j], (IMG_SIZE, IMG_SIZE))

            X.append(X_case)
            y.append(y_case)

        X = np.concatenate(X, axis=0).astype(np.float32)
        y = np.concatenate(y, axis=0).astype(np.float32)
        mask = tf.one_hot(y, depth=1)
        return X, mask

training_generator = DataGenerator(train_ids)
val_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

# Vision Transformer model
def ViT(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Patch creation
    patches = tf.keras.layers.Conv2D(filters=96, kernel_size=4, strides=4, padding='valid')(inputs)
    patches = tf.keras.layers.Reshape((-1, 96))(patches)

    # Positional encoding
    num_patches = patches.shape[1]
    projection_dim = patches.shape[2]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positional_encoding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)

    # Add positional encoding
    encoded_patches = patches + positional_encoding

    # Transformer encoder
    for _ in range(4):  # Moderate number of transformer blocks
        # Layer normalization 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=projection_dim)(x1, x1)
        # Skip connection 1
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        mlp_output = tf.keras.layers.Dense(units=projection_dim, activation=tf.keras.activations.gelu)(x3)
        mlp_output = tf.keras.layers.Dense(units=projection_dim)(mlp_output)
        # Skip connection 2
        encoded_patches = tf.keras.layers.Add()([mlp_output, x2])

    # Decoder path
    x = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 96))(encoded_patches)
    x = tf.keras.layers.Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(24, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu')(x)

    # Output layer to match input dimensions
    output = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 1  # Binary segmentation
model = ViT(input_shape, num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=dice_loss,
              metrics=['accuracy', dice_coeff, iou])

# Debugging
for batch in training_generator:
    X, y = batch
    print("Input shape:", X.shape)
    print("Output shape:", y.shape)
    break

# Training
wt_path = "weights.h5"
checkpoint = ModelCheckpoint(filepath=wt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

history = model.fit(training_generator,
                    steps_per_epoch=len(train_ids),
                    validation_data=val_generator,
                    callbacks=[checkpoint, early_stop],
                    epochs=150)

# Testing
test_loss, test_acc, test_dice, test_iou = model.evaluate(test_generator, steps=len(test_ids))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
print(f"Test Dice Coefficient: {test_dice}")
print(f"Test IoU: {test_iou}")
