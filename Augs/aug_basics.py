# MAIN.PY

import cv2
# import nibabel as nib
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from focal_loss import BinaryFocalLoss
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, Reshape

import albumentations as A

print("DEBUG: Libs imported")

if os.name == 'nt':
    base_path = "C:\\Cyberkid\\MyMTech\\Labwork\\SecondYear\\MyWork\\Datasets\\ISLES-2022\\ISLES-2022"
else:
    # base_path = "/home/user/Tf_script/dataset/ISLES_2022/"
    base_path = "/home/user/adithyaes/dataset/isles2022_png/"

scaler = MinMaxScaler()

# transform = A.Compose([
#     A.RandomCrop(width=64, height=64),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

# transform = A.Compose([
#     A.RandomCrop(width=64, height=64),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Rotate(limit=40, p=0.5),
#     A.GaussNoise(var_limit=(10, 50), p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=40, p=0.5),
#     A.Normalize(),
# ])

transform = A.Compose([
    A.Resize(height=128, width=128),  # Resize to a standard size
    A.RandomCrop(height=64, width=64),  # Random crop for local context
    A.HorizontalFlip(p=0.5),  # Horizontal flip to augment the dataset
    A.VerticalFlip(p=0.5),  # Vertical flip for additional variation
    A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
    A.Rotate(limit=30, p=0.5),  # Rotate by up to 30 degrees to prevent orientation bias
    A.GaussNoise(var_limit=(10, 20), p=0.5),  # Add noise but keep it moderate
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  # Shift, scale, and rotate
    A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),  # Elastic transformations to mimic real-world variations
    A.Normalize(),  # Normalize the pixel values
    # A.pytorch.transforms.ToTensorV2()  # Convert to PyTorch tensor if needed
])

IMG_SIZE=112
PATH_DATASET = base_path
PATH_RAWDATA = os.path.join(base_path, "input")
PATH_DERIVATIVES = os.path.join(base_path, "mask")
OUTPUT_DIRECTORY = "./output/ISLESfolder"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

print("PATH_RAWDATA: ", PATH_RAWDATA)
print("PATH_DERIVATIVES: ", PATH_DERIVATIVES)
print("No of Folders Inside Training: ", len(os.listdir(PATH_RAWDATA)))
print("No of Folders Inside Ground Truth: ", len(os.listdir(PATH_DERIVATIVES)))

# # # # # Functions



# # # # # Loss Functions



# # # # # Classes



# # # # # Augmentation

image = cv2.imread(os.path.join(PATH_RAWDATA, "slice_sub-strokecase0038_0040.png"))
mask = cv2.imread(os.path.join(PATH_DERIVATIVES, "slice_sub-strokecase0038_0040.png"))

if image is None:
    print("Error: Unable to load image.")
if mask is None:
    print("Error: Unable to load mask.")

# Proceed only if both images are loaded successfully
if image is not None and mask is not None:
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    cv2.imshow('Original Image', image)
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.imshow('Transformed Mask', transformed_mask)

    key = cv2.waitKey(50000)  # Wait indefinitely for a key press

    # Optional: You can check which key was pressed
    if True:  # If 'q' is pressed, exit
        cv2.destroyAllWindows()
        exit(0)

    cv2.destroyAllWindows()  # Close the windows after the key is pressed
