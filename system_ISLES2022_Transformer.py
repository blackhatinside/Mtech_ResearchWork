#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nibabel as nib
import numpy as np
import os
import cv2
import types
import tensorflow as tf
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import concatenate
from sklearn.model_selection import train_test_split

# -----DEBUG-----
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__
print(list(imports()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


# Paths
TRAIN_DATASET_PATH = '/home/user/Tf_script/dataset/ISLES_2022/rawdata/'
TRAINMask_DATASET_PATH = '/home/user/Tf_script/dataset/ISLES_2022/derivatives/'

# Get dataset details
trainfolders = os.listdir(TRAIN_DATASET_PATH)
train_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# train_ids = [train_directories[i][48:66] for i in range(len(train_directories))]
train_directory_startindex = train_directories[0].find("sub")
train_ids = sorted([train_directories[i][train_directory_startindex:] for i in range(len(train_directories))])

maskfolders = os.listdir(TRAINMask_DATASET_PATH)
mask_directories = [f.path for f in os.scandir(TRAINMask_DATASET_PATH) if f.is_dir()]

# mask_ids = [mask_directories[i][48:66] for i in range(len(mask_directories))]
mask_id_startindex = mask_directories[0].find("sub")
mask_ids = sorted([mask_directories[i][mask_id_startindex:] for i in range(len(mask_directories))])

# -----DEBUG-----
print("Train IDs: ", len(train_ids))
# print(train_ids[0], "to", train_ids[-1])
print(sorted(train_ids)[:5])
print("Mask IDs: ", len(mask_ids))
# print(mask_ids[0], "to", mask_ids[-1])
print(sorted(mask_ids)[:5])


# In[4]:


# train_test_ids, val_ids, train_test_mask, val_mask = train_test_split(train_ids, mask_ids, test_size=0.15, random_state=42)
# train_ids, test_ids, train_mask, test_mask = train_test_split(train_test_ids, train_test_mask, test_size=0.15, random_state=42)
train_test_ids, val_ids,train_test_mask, val_mask = train_test_split(train_ids,mask_ids,test_size=0.2,random_state = 32) 
train_ids,  test_ids, train_mask , test_mask = train_test_split(train_test_ids,train_test_mask,test_size=0.2,random_state = 32)

scaler = MinMaxScaler()

IMG_SIZE = 112

# -----DEBUG-----
print("Dimensions: ", ("{} X {}".format(IMG_SIZE, IMG_SIZE)))


# In[5]:


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


# In[6]:


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
            # global slices_per_sample
            # if slices_per_sample is None:
            #     slices_per_sample = slices
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
#         mask = tf.one_hot(y, depth=1)
#         return X, mask
        return X, y

training_generator = DataGenerator(train_ids, batch_size=1)  # Reduce batch size
val_generator = DataGenerator(val_ids, batch_size=1)
test_generator = DataGenerator(test_ids, batch_size=1)

# -----DEBUG-----
sample1 = nib.load("/home/user/Tf_script/dataset/ISLES_2022/rawdata/sub-strokecase0001/\
ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz").get_fdata()
sample2 = nib.load("/home/user/Tf_script/dataset/ISLES_2022/rawdata/sub-strokecase0002/\
ses-0001/dwi/sub-strokecase0002_ses-0001_dwi.nii.gz").get_fdata()
sample3 = nib.load("/home/user/Tf_script/dataset/ISLES_2022/rawdata/sub-strokecase0003/\
ses-0001/dwi/sub-strokecase0003_ses-0001_dwi.nii.gz").get_fdata()
sample4 = nib.load("/home/user/Tf_script/dataset/ISLES_2022/rawdata/sub-strokecase0004/\
ses-0001/dwi/sub-strokecase0004_ses-0001_dwi.nii.gz").get_fdata()
sample5 = nib.load("/home/user/Tf_script/dataset/ISLES_2022/rawdata/sub-strokecase0005/\
ses-0001/dwi/sub-strokecase0005_ses-0001_dwi.nii.gz").get_fdata()
print(sample1.shape[2])
print(sample2.shape[2])
print(sample3.shape[2])
print(sample4.shape[2])
print(sample5.shape[2])

print("Training: ", len(training_generator))
print("Validation: ", len(val_generator))
print("Testing: ", len(test_generator))


# In[7]:


# # Vision Transformer model
# def ViT(input_shape, num_classes):
#     inputs = tf.keras.layers.Input(shape=input_shape)
    
#     # Patch creation
#     patches = tf.keras.layers.Conv2D(filters=96, kernel_size=4, strides=4, padding='valid')(inputs)
#     patches = tf.keras.layers.Reshape((-1, 96))(patches)
    
#     # Positional encoding
#     num_patches = patches.shape[1]
#     projection_dim = patches.shape[2]
#     positions = tf.range(start=0, limit=num_patches, delta=1)
#     positional_encoding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    
#     # Add positional encoding
#     encoded_patches = patches + positional_encoding
    
#     # Transformer encoder
#     for _ in range(6):  # Moderate -4/Increased-6 number of transformer blocks
#         # Layer normalization 1
#         x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         # Multi-head attention MOderate - 6/Increased - 8
#         attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=projection_dim)(x1, x1)
#         # Skip connection 1
#         x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2
#         x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
#         # MLP
#         mlp_output = tf.keras.layers.Dense(units=projection_dim, activation=tf.keras.activations.gelu)(x3)
#         mlp_output = tf.keras.layers.Dense(units=projection_dim)(mlp_output)
#         # Skip connection 2
#         encoded_patches = tf.keras.layers.Add()([mlp_output, x2])
    
#     # Decoder path
#     x = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 96))(encoded_patches)
#     x = tf.keras.layers.Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = tf.keras.layers.Conv2DTranspose(24, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu')(x)
    
#     # Output layer to match input dimensions
#     output = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x)
    
#     return tf.keras.Model(inputs=inputs, outputs=output)


# In[8]:


# # Vision Transformer model
# def ViT(input_shape, num_classes):
#     inputs = tf.keras.layers.Input(shape=input_shape)
    
#     # Patch creation
#     patches = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=4, padding='valid')(inputs)
#     patches = tf.keras.layers.Reshape((-1, 64))(patches)
    
#     # Positional encoding
#     num_patches = patches.shape[1]
#     projection_dim = patches.shape[2]
#     positions = tf.range(start=0, limit=num_patches, delta=1)
#     positional_encoding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    
#     # Add positional encoding
#     encoded_patches = patches + positional_encoding
    
#     # Transformer encoder
#     for _ in range(8):  # Number of transformer blocks
#         # Layer normalization 1
#         x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         # Multi-head attention
#         attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=projection_dim)(x1, x1)
#         # Skip connection 1
#         x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2
#         x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
#         # MLP
#         mlp_output = tf.keras.layers.Dense(units=projection_dim, activation=tf.keras.activations.gelu)(x3)
#         mlp_output = tf.keras.layers.Dense(units=projection_dim)(mlp_output)
#         # Skip connection 2
#         encoded_patches = tf.keras.layers.Add()([mlp_output, x2])
    
#     # Decoder path
#     x = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 64))(encoded_patches)
#     x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
#     x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    
#     # Output layer to match input dimensions
#     output = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x)
    
#     return tf.keras.Model(inputs=inputs, outputs=output)


# In[9]:


def ViT(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Patch creation
    patches = tf.keras.layers.Conv2D(filters=96, kernel_size=4, strides=4, padding='valid',
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    patches = tf.keras.layers.BatchNormalization()(patches)
    patches = tf.keras.layers.Reshape((-1, 96))(patches)
    
    # Positional encoding
    num_patches = patches.shape[1]
    projection_dim = patches.shape[2]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positional_encoding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    
    # Add positional encoding
    encoded_patches = patches + positional_encoding
    
    # Transformer encoder
    for _ in range(4):
        # Layer normalization 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=projection_dim,
                                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x1, x1)
        # Skip connection 1
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        mlp_output = tf.keras.layers.Dense(units=projection_dim, activation=tf.keras.activations.gelu,
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x3)
        mlp_output = tf.keras.layers.Dense(units=projection_dim, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(mlp_output)
        # Skip connection 2
        encoded_patches = tf.keras.layers.Add()([mlp_output, x2])
    
    # Decoder path
    x = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 96))(encoded_patches)
    x = tf.keras.layers.Conv2DTranspose(48, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(24, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layer to match input dimensions
    output = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=output)


# In[10]:


input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 1  # Binary classification
# num_classes = 2 # Legion vs no Legion
learn_rate = 1e-5
epochs_count = 150

model = ViT(input_shape, num_classes)
model.summary()


# In[11]:


import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

for x, y in training_generator:
    print(f"Input shape: {x.shape}, Label shape: {y.shape}")
    break

print(f"Layer {model.layers[0].name} output shape: {model.layers[0].output_shape}")
print(f"Layer {model.layers[-1].name} output shape: {model.layers[-1].output_shape}")


# In[ ]:


# Compile the model
# model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coeff, iou])
model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), 
            loss=dice_loss, 
            metrics=[dice_coeff, iou])

optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate)

# Checkpoints and learning rate adjustments
# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode="min", verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode="min")
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# Train the model
history = model.fit(training_generator,
                    validation_data=val_generator,
                    epochs=epochs_count,
                    steps_per_epoch=len(train_ids),
#                     callbacks=[checkpoint, reduce_lr, early_stopping])
                    callbacks=[checkpoint, early_stopping])

# Save the model
model.save("vit_brain_lesion_segmentation.h5")

