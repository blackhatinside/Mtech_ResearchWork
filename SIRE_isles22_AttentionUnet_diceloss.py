#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import nibabel as nib
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

import tensorflow as tf
import keras.backend as K
# from keras.models import Model, load_model
# from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate
#from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import CSVLogger
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models, layers, regularizers

from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split


# In[2]:


def get_ids(path):
    directories = [f.path for f in os.scandir(path) if f.is_dir()]
    ids = []
    id_startindex = directories[0].find("sub")
    for i in range(len(directories)):
        ids.append(directories[i][id_startindex:])
    return sorted(ids)


# In[3]:


PATH_DATASET = "/home/user/Tf_script/dataset/ISLES_2022/"
PATH_RAWDATA = "/home/user/Tf_script/dataset/ISLES_2022/rawdata/"
PATH_DERIVATIVES = "/home/user/Tf_script/dataset/ISLES_2022/derivatives/"

print("No of Folders Inside Dataset: ", len(os.listdir(PATH_DATASET)))
# print("Folders Inside Dataset: ", os.listdir("../input/isles2022small/ISLES2022/"))
print("No of Folders Inside Training: ", len(os.listdir(PATH_RAWDATA)))
# print("Folders Inside Training: ", os.listdir("../input/isles2022small/ISLES2022/rawdata/"))
print("No of Folders Inside Ground Truth: ", len(os.listdir(PATH_DERIVATIVES)))
# print("Folders Inside Ground Truth: ", os.listdir("../input/isles2022small/ISLES2022/derivatives/"))


# In[4]:


TRAIN_DATASET_PATH = PATH_RAWDATA
train_ids = get_ids(TRAIN_DATASET_PATH)
TRAINMask_DATASET_PATH = PATH_DERIVATIVES
mask_ids = get_ids(TRAINMask_DATASET_PATH)

print("no of train_ids: ", len(train_ids))
# print("train_ids: ", train_ids)
print("no of mask_ids: ", len(mask_ids))
# print("mask_ids: ", mask_ids)


# In[5]:


train_test_ids, val_ids,train_test_mask, val_mask = train_test_split(train_ids,mask_ids,test_size=0.1) 
train_ids,  test_ids, train_mask , test_mask = train_test_split(train_test_ids,train_test_mask,test_size=0.15)

tvt_ids = [train_ids, val_ids, test_ids]
print("train, validate, test: ", list(map(len, tvt_ids)))


# In[6]:


# defining the performance metrics
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


# In[7]:


#VOLUME_SLICES = 20
#VOLUME_START_AT = 5
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
        #X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        #y = np.zeros((self.batch_size*VOLUME_SLICES, 112, 112))
        #Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim))

        
        # Generate data
        for c, i in enumerate(Batch_ids):           
            
            
            case_path = os.path.join(TRAIN_DATASET_PATH, i)
            data_path = os.path.join(case_path + '/ses-0001/dwi', f'{i}_ses-0001_dwi.nii.gz');
            dwi = nib.load(data_path).get_fdata()
            #dwi=dwi.astype(np.uint8)
            dwi=scaler.fit_transform(dwi.reshape(-1, dwi.shape[-1])).reshape(dwi.shape)
            slices = dwi.shape[2]
            X = np.zeros((slices, 112,112, 1))
            #X=X.astype(np.float32)

            case_path2 = os.path.join(TRAINMask_DATASET_PATH, i)
            data_path_2 = os.path.join(case_path2 + '/ses-0001', f'{i}_ses-0001_msk.nii.gz');
            msk = nib.load(data_path_2).get_fdata()
            #msk=msk.astype(np.uint8)
            msk_slices = msk.shape[2]
            y = np.zeros((msk_slices, 112,112))
            #y=y.astype(np.float32)  

            
            for j in range(slices):
                X[j,:,:,0] = cv2.resize(dwi[:,:,j+0], (IMG_SIZE, IMG_SIZE));
                X=X.astype(np.float32)
                #X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                y[j] = cv2.resize(msk[:,:,j+0],(112,112));
                #y=y.astype(np.float32) 
#                 y[j] = msk[:,:,j+VOLUME_START_AT];
       
        #mask = tf.one_hot(y, 2)
        #print(X.shape)
        #print(X.max())
        
        #return X/np.max(X), mask
        return X, y


# In[8]:


training_generator = DataGenerator(train_ids)
val_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

tvt_generator = [training_generator, val_generator, test_generator]
print("train, validate, test: ", list(map(len, tvt_generator)))
# test_ids


# In[9]:


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

def attention_block(l_layer,h_layer): #Attention Block
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
    #concat_layer=attention_block(inp,concat_layer)
    x=concatenate([x,concat_layer])
    x=conv_block(x,filters)
    return x    


# In[10]:


VAL_EPOCH = 30
VAL_PATIENCE = 40


# In[11]:


inputs=Input((112,112,1))
#inputfloat=Lambda(lambda x: x / 255)(inputs)

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
    loss=dice_loss, 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    metrics = ['accuracy', dice_coeff,dice_score,iou, precision] 
)
#model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = ['accuracy', dice_coeff, iou] )

model.summary()


# In[12]:


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


# In[13]:


test_wt=model.predict(test_generator)
test_wt.shape


# In[14]:


results = model.evaluate(test_generator, steps=len(test_ids))
print("Test loss: ",results[0])
print("Test Dice: ",results[2])


# In[15]:


fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(test_wt[10,:,:,:],cmap='gray')


# In[16]:


y_pred_thresholded = test_wt > 0.4
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[10,:,:,:],cmap='gray')


# In[17]:


# Function to compute the Dice coefficient
def dice_coeff(y_true,y_pred):
    # Ensure both tensors are of type float32
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

# Initialize lists to store loss and metric values
loss_values = []
dice_values = []
iou_values = []

# Loop through the test generator
for batch_x, batch_y in test_generator:
    # Predict the output for the batch
    #print(batch_y.shape)
    mask_image = np.expand_dims(batch_y, axis=-1)
    y_predwts = model.predict(batch_x)
    #y_predwt = y_predwts
    #print('y_pred',y_predwts.shape)
    #y_pred_thresholded = np.where(y_predwts >= 0.5, 1.0, 0.0).astype(np.float32)#binary
    y_pred = np.where(y_predwts < 0.2, 0.0, y_predwts).astype(np.float32)#relu
    y_pred_thresholded = y_pred 
    
    # Loop through each sample in the batch
    for i in range(len(batch_x)):
        # Compute the loss and metrics for each sample
        #loss = compute_loss(batch_y[i], y_pred[i])
        dice = dice_coeff(batch_y[i], y_pred_thresholded[i])
        iou_value = iou(batch_y[i], y_pred_thresholded[i])
    
        # Store the computed values
        #loss_values.append(loss)
        dice_values.append(dice)
        iou_values.append(iou_value)

    
    # Stop if we've processed all steps
    if len(loss_values) >= len(test_generator):
        break

# Compute the average loss and metrics
#average_loss = np.mean(loss_values)
average_dice = np.mean(dice_values)
average_iou = np.mean(iou_values)

# Convert the list to a tensor for easy aggregation
hd_distances = tf.stack(hausdorff_distance_value)

#Exclude slices with inf values from the calculation
valid_hd_distances = tf.boolean_mask(hd_distances, tf.math.is_finite(hd_distances))
mean_hd = tf.reduce_mean(valid_hd_distances)
max_hd = tf.reduce_max(valid_hd_distances)

#print("Average test loss: ", average_loss)
print("Average test dice: ", average_dice)
print("Average test IoU: ", average_iou)


# In[ ]:


# paths.
isles_data_dir = '/home/user/Tf_script/dataset/ISLES_2022/'
example_case = 19

# Set images path.
dwi_path = os.path.join(isles_data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'dwi/'
                    'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" % example_case))
mask_path = os.path.join(isles_data_dir, 'derivatives', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'sub-strokecase{}_ses-0001_msk.nii.gz'.format("%04d" % example_case))


# In[ ]:


# Load image data.
dwi_image = nib.load(dwi_path).get_fdata()
mask_image = nib.load(mask_path).get_fdata()


# In[ ]:


# def img_resize(img, dims):
#     return cv2.resize(img[:,:], dims)

img_resize = lambda img, dims: cv2.resize(img[:,:], dims)


# In[ ]:


dwi_image=img_resize(dwi_image, (112, 112))
mask_image=img_resize(mask_image, (112, 112))
dwi_image.shape
mask_image.shape


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2)

slice2show=31
ax1.imshow(dwi_image[:,:,slice2show], cmap='gray')
ax1.set_title('Dwi')
ax1.set_axis_off()


# Show DWI image w/overlayed mask.
ax2.imshow(mask_image[:,:,slice2show], cmap='gray')
#ax2.imshow(mask_image[:,:,slice2show], alpha=0.5, cmap='copper')
ax2.set_title('GT')
ax2.set_axis_off()


# In[ ]:


dwi_image=scaler.fit_transform(dwi_image.reshape(-1, dwi_image.shape[-1])).reshape(dwi_image.shape)


# In[ ]:


X = np.zeros((72,112,112,1))
for j in range(72):
    X[j,:,:,0] =dwi_image[:,:,j]
X.shape


# In[ ]:


pred_wt=model.predict(X)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(pred_wt[31,:,:,:],cmap='gray')


# In[ ]:


y_pred_thresholded = pred_wt > 0.1


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[31,:,:,:],cmap='gray')


# In[ ]:


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    total = np.sum(y_true) + np.sum(y_pred)
    dice = (2 * intersection +1 ) / (total + 1)  # Adding a small epsilon to avoid division by zero
    #dice = np.mean(dice)
    dice = round(dice, 3)
    return dice

def iou(y_true,y_pred):
    intersec = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred)
    iou = (intersec + 1) / (union- intersec + 1)
    iou = round(iou, 3)
    return iou


# In[ ]:





# In[ ]:


# Specify the directory to save the plot images
output_directory = './output/ISLESfolder'
os.makedirs(output_directory, exist_ok=True)

# Plot each slice along with the original mask and predicted mask
for i in range(5,60):
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 4, 1)
    plt.imshow(dwi_image[:,:,i], cmap='gray')
    plt.title('Input Slice')

    # Plot the original mask
    plt.subplot(1, 4, 2)
    plt.imshow(mask_image[:,:,i], cmap='gray')
    plt.title('Original Mask')

    # Plot the predicted mask
    plt.subplot(1, 4, 3)
    plt.imshow(pred_wt[i,:,:,:], cmap='gray')
    plt.title('Predicted Mask')
    
    # Plot the predicted mask
    plt.subplot(1, 4, 4)
    plt.imshow(y_pred_thresholded[i,:,:,:], cmap='gray')
    plt.title('Thresholed Mask')

    #plt.suptitle(f"Slice: {i+1}")
    dice = dice_score(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    Iou = iou(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    plt.suptitle(f"Sample_19_Slice_00{i}  ,Dice Score:{dice}  ,IOU:{Iou}")
    #print(f'Dice Score: {dice}')
    #plt.savefig(f'plot_{i}.png')
    #plt.show()
        
    # Save the plot image in the output folder
    output_filename = f'Sample_19_Slice_00{i}.png'
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)
    plt.show()
    plt.close()  # Close the figure to release memory

