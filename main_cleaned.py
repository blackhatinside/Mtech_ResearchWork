import nibabel as nib
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tqdm import tqdm
from skimage.io import imshow
from skimage.transform import resize

import tensorflow as tf
import keras.backend as K

from tensorflow.keras.layers.core import Lambda
from tensorflow.keras.layers.core import Reshape

from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers.convolutional import Conv2DTranspose

from tensorflow.keras.layers.pooling import MaxPooling2D

from tensorflow.keras.layers import concatenate

# from tensorflow.keras import layers
# from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split

print("Hi")

if os.name == 'nt':
    base_path = "C:\\Cyberkid\\MyMTech\\Labwork\\SecondYear\\MyWork\\Datasets\\ISLES-2022\\ISLES-2022"
else:
    base_path = "/home/user/Tf_script/dataset/ISLES_2022/"

PATH_DATASET = base_path
PATH_RAWDATA = os.path.join(base_path, "rawdata")
PATH_DERIVATIVES = os.path.join(base_path, "derivatives")

print("No of Folders Inside Training: ", len(os.listdir(PATH_RAWDATA)))
print("No of Folders Inside Ground Truth: ", len(os.listdir(PATH_DERIVATIVES)))

def get_ids(path):
    directories = [f.path for f in os.scandir(path) if f.is_dir()]
    ids = []
    id_startindex = directories[0].find("sub")
    for i in range(len(directories)):
        ids.append(directories[i][id_startindex:])
    return sorted(ids)

train_ids = get_ids(PATH_RAWDATA)
mask_ids = get_ids(PATH_DERIVATIVES)

print("No of train_ids: {}\nNo of mask_ids: {}\n".format(len(train_ids), len(mask_ids)))

train_test_ids, val_ids,train_test_mask, val_mask = train_test_split(train_ids,mask_ids,test_size=0.1)
train_ids,  test_ids, train_mask , test_mask = train_test_split(train_test_ids,train_test_mask,test_size=0.15)

tvt_ids = [train_ids, val_ids, test_ids]
print("train, validate, test: ", list(map(len, tvt_ids)))

def dice_coeff(y_true,y_pred):
    y_true_new = K.flatten(y_true)
    y_pred_new = K.flatten(y_pred)
    denominator = K.sum(y_true_new) + K.sum(y_pred_new)
    numerator = K.sum(y_true_new * y_pred_new)
    return (2*numerator + 1)/(denominator+1)

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def dsc(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    dsc = (2*tp) / ((2*tp) + fn + fp)
    return dsc

def iou(y_true,y_pred):
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    iou = (intersec + 0.1) / (union- intersec + 0.1)
    return iou

def dice_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice

def dice_loss(y_true, y_pred):
    return 1.0 -dice_score(y_true, y_pred)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy_loss = -y_true * K.log(y_pred)
        focal_loss_value = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy_loss
        return K.sum(focal_loss_value)

    return focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_pred) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return focal_loss

print("Hi")

IMG_SIZE=112

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 1, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(PATH_RAWDATA, i)
            data_path = os.path.join(case_path, 'ses-0001', 'dwi', f'{i}_ses-0001_dwi.nii.gz');
            dwi = nib.load(data_path).get_fdata()
            dwi=scaler.fit_transform(dwi.reshape(-1, dwi.shape[-1])).reshape(dwi.shape)
            slices = dwi.shape[2]
            X = np.zeros((slices, 112,112, 1))
            case_path2 = os.path.join(PATH_DERIVATIVES, i)
            data_path_2 = os.path.join(case_path2, 'ses-0001', f'{i}_ses-0001_msk.nii.gz');
            msk = nib.load(data_path_2).get_fdata()
            msk_slices = msk.shape[2]
            y = np.zeros((msk_slices, 112,112))
            for j in range(slices):
                X[j,:,:,0] = cv2.resize(dwi[:,:,j+0], (IMG_SIZE, IMG_SIZE));
                X=X.astype(np.float32)
                y[j] = cv2.resize(msk[:,:,j+0],(112,112));
        return X, y

print("Hi")

training_generator = DataGenerator(train_ids)
val_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

tvt_generator = [training_generator, val_generator, test_generator]
print("train, validate, test: ", list(map(len, tvt_generator)))

def conv_block(inp,filters):
    x=Conv2D(filters,(3,3),padding='same',activation='relu')(inp)
    x=Conv2D(filters,(3,3),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    return x

def encoder_block(inp,filters):
    x=conv_block(inp,filters)
    p=MaxPooling2D(pool_size=(2,2))(x)
    return x,p

def attention_block(l_layer,h_layer):
    phi=Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer)
    theta=Conv2D(h_layer.shape[-1],(1,1),strides=(2,2),padding='same')(h_layer)
    x=tf.keras.layers.add([phi,theta])
    x=Activation('relu')(x)
    x=Conv2D(1,(1,1),padding='same',activation='sigmoid')(x)
    x=UpSampling2D(size=(2,2))(x)
    x=tf.keras.layers.multiply([h_layer,x])
    x=BatchNormalization(axis=3)(x)
    return x

def decoder_block(inp,filters,concat_layer):
    x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp)
    x=concatenate([x,concat_layer])
    x=conv_block(x,filters)
    return x

VAL_EPOCH = 30
VAL_PATIENCE = 40

inputs=Input((112,112,1))

d1,p1=encoder_block(inputs,64)
d2,p2=encoder_block(p1,128)
d3,p3=encoder_block(p2,256)
d4,p4=encoder_block(p3,512)
b1=conv_block(p4,1024)
e2=decoder_block(b1,512,d4)
e3=decoder_block(e2,256,d3)
e4=decoder_block(e3,128,d2)
e5=decoder_block(e4,64,d1)

outputs = Conv2D(1, (1,1),activation="sigmoid")(e5)
model=Model(inputs=[inputs], outputs=[outputs],name='AttentionUnet')

model.compile(
    loss=focal_loss(gamma=2.0, alpha=0.25),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy', dice_coeff,dice_score,iou, precision]
)

model.summary()

print("Hi")

checkpoint = ModelCheckpoint(
    'DiceLoss_ISLES22_2DAttention_wts.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=VAL_PATIENCE,
    verbose=1,
    restore_best_weights=True
)

att_unet_history = model.fit(
    training_generator,
    steps_per_epoch=len(train_ids),
    validation_data=val_generator,
    callbacks= [checkpoint,early_stop],
    epochs=VAL_EPOCH
)

test_wt=model.predict(test_generator)
test_wt.shape

results = model.evaluate(test_generator, steps=len(test_ids))
print("Test loss: ",results[0])
print("Test Dice: ",results[2])

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(test_wt[10,:,:,:],cmap='gray')

y_pred_thresholded = test_wt > 0.4
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[10,:,:,:],cmap='gray')

def dice_coeff(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_new = K.flatten(y_true)
    y_pred_new = K.flatten(y_pred)
    denominator = K.sum(y_true_new) + K.sum(y_pred_new)
    if denominator == 0.0:
        return 1.0
    numerator = K.sum(y_true_new * y_pred_new)
    return (2.0*numerator)/(denominator)

def iou(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    if union == 0.0:
        return 1.0
    iou = (intersec) / (union- intersec)
    return iou

loss_values = []
dice_values = []
iou_values = []

for batch_x, batch_y in test_generator:
    mask_image = np.expand_dims(batch_y, axis=-1)
    y_predwts = model.predict(batch_x)
    y_pred = np.where(y_predwts < 0.2, 0.0, y_predwts).astype(np.float32)
    y_pred_thresholded = y_pred
    for i in range(len(batch_x)):
        dice = dice_coeff(batch_y[i], y_pred_thresholded[i])
        iou_value = iou(batch_y[i], y_pred_thresholded[i])
        dice_values.append(dice)
        iou_values.append(iou_value)
    if len(loss_values) >= len(test_generator):
        break

average_dice = np.mean(dice_values)
average_iou = np.mean(iou_values)

print("Average test dice: ", average_dice)
print("Average test IoU: ", average_iou)

example_case = 19

dwi_path = os.path.join(base_path, 'rawdata', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'dwi/'
                    'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" % example_case))
mask_path = os.path.join(base_path, 'derivatives', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'sub-strokecase{}_ses-0001_msk.nii.gz'.format("%04d" % example_case))

dwi_image = nib.load(dwi_path).get_fdata()
mask_image = nib.load(mask_path).get_fdata()

img_resize = lambda img, dims: cv2.resize(img[:,:], dims)

dwi_image=img_resize(dwi_image, (112, 112))
mask_image=img_resize(mask_image, (112, 112))
dwi_image.shape
mask_image.shape

fig, (ax1, ax2) = plt.subplots(1, 2)

slice2show=31
ax1.imshow(dwi_image[:,:,slice2show], cmap='gray')
ax1.set_title('Dwi')
ax1.set_axis_off()


ax2.imshow(mask_image[:,:,slice2show], cmap='gray')
ax2.set_title('GT')
ax2.set_axis_off()

dwi_image=scaler.fit_transform(dwi_image.reshape(-1, dwi_image.shape[-1])).reshape(dwi_image.shape)

X = np.zeros((72,112,112,1))
for j in range(72):
    X[j,:,:,0] =dwi_image[:,:,j]
X.shape

pred_wt=model.predict(X)

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(pred_wt[31,:,:,:],cmap='gray')

y_pred_thresholded = pred_wt > 0.1

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[31,:,:,:],cmap='gray')

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    dice = (2 * intersection +1 ) / (total + 1)
    dice = round(dice, 3)
    return dice

def iou(y_true,y_pred):
    intersec = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred)
    iou = (intersec + 1) / (union- intersec + 1)
    iou = round(iou, 3)
    return iou

output_directory = './output/ISLESfolder'
os.makedirs(output_directory, exist_ok=True)

for i in range(5,60):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    # plt.imshow(dwi_image[:,:,i], cmap='gray')
    plt.title('Input')
    plt.subplot(1, 4, 2)
    # plt.imshow(mask_image[:,:,i], cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(1, 4, 3)
    # plt.imshow(pred_wt[i,:,:,:], cmap='gray')
    plt.title('Predicted')
    plt.subplot(1, 4, 4)
    # plt.imshow(y_pred_thresholded[i,:,:,:], cmap='gray')
    plt.title('Threshold')
    dice = dice_score(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    Iou = iou(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    plt.suptitle(f"Sample_19_Slice_00{i}  ,Dice Score:{dice}  ,IOU:{Iou}")
    output_filename = f'Sample_19_Slice_00{i}.png'
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)
    # plt.show()
    plt.close()
