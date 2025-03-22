# import cv2, numpy as np, os, tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.regularizers import l2
# import keras.backend as K

# BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
# AUG_PATH = "/home/user/adithyaes/dataset/isles2022_png_aug"
# INPUT_PATH, MASK_PATH = os.path.join(BASE_PATH, "input"), os.path.join(BASE_PATH, "mask")
# AUG_INPUT_PATH, AUG_MASK_PATH = os.path.join(AUG_PATH, "input"), os.path.join(AUG_PATH, "mask")
# OUTPUT_DIRECTORY = "./output/ISLESfolder"
# os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# IMG_SIZE, BATCH_SIZE, INITIAL_LR, WEIGHT_DECAY = 112, 4, 3e-4, 1e-4
# scaler = StandardScaler()

# def dice_coeff(y_true, y_pred):
#     y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
#     return (2. * K.sum(K.flatten(y_true) * K.flatten(y_pred)) + 1) / (K.sum(K.flatten(y_true)) + K.sum(K.flatten(y_pred)) + 1)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coeff(y_true, y_pred)

# def binary_focal_loss(gamma=2., alpha=0.25):
#     def focal_loss(y_true, y_pred):
#         y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
#         p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
#         alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
#         return K.mean(-alpha_t * K.pow(1 - p_t, gamma) * K.log(K.clip(p_t, K.epsilon(), 1.0)))
#     return focal_loss

# def combined_loss(y_true, y_pred):
#     y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
#     return tf.cast(0.5 * dice_loss(y_true, y_pred) +
#                   0.3 * binary_focal_loss()(y_true, y_pred) +
#                   0.2 * K.mean(K.binary_crossentropy(y_true, y_pred) *
#                               tf.cast(tf.where(y_true > 0, 0.7, 0.3), tf.float32)), tf.float32)

# def precision(y_true, y_pred):
#     y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
#     return K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())

# def iou(y_true, y_pred):
#     y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
#     intersection = K.sum(y_true * y_pred)
#     return (intersection + 0.1) / (K.sum(y_true + y_pred) - intersection + 0.1)

# class ResidualBlock(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size=3):
#         super(ResidualBlock, self).__init__()
#         self.filters, self.conv1 = filters, Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(WEIGHT_DECAY))
#         self.bn1, self.conv2 = BatchNormalization(), Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(WEIGHT_DECAY))
#         self.bn2, self.relu = BatchNormalization(), Activation('relu')
#         self.dropout = Dropout(0.2)
#         self.projection = None

#     def build(self, input_shape):
#         if input_shape[-1] != self.filters:
#             self.projection = Conv2D(self.filters, 1, padding='same', kernel_regularizer=l2(WEIGHT_DECAY))
#         super().build(input_shape)

#     def call(self, inputs):
#         shortcut = self.projection(inputs) if self.projection else inputs
#         return self.relu(self.bn2(self.conv2(self.dropout(self.relu(self.bn1(self.conv1(inputs)))))) + shortcut)

# class AttentionGate(tf.keras.layers.Layer):
#     def __init__(self, filters):
#         super(AttentionGate, self).__init__()
#         self.W_g, self.W_x = Conv2D(filters, 1), Conv2D(filters, 1)
#         self.psi = Conv2D(1, 1)

#     def call(self, g, x):
#         return Multiply()([x, Activation('sigmoid')(self.psi(Activation('relu')(Add()([self.W_g(g), self.W_x(x)]))))])

# class AugmentedDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
#         self.batch_size, self.list_IDs = batch_size, list_IDs
#         self.aug_ids = aug_ids if aug_ids else []
#         self.all_ids = list_IDs + self.aug_ids
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.all_ids) / self.batch_size))

#     def __getitem__(self, index):
#         batch_ids = [self.all_ids[k] for k in range(index * self.batch_size,
#                     min((index + 1) * self.batch_size, len(self.all_ids)))]
#         return self.__data_generation(batch_ids)

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.all_ids))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, batch_ids):
#         X = np.empty((len(batch_ids), IMG_SIZE, IMG_SIZE, 1))
#         y = np.empty((len(batch_ids), IMG_SIZE, IMG_SIZE, 1))

#         for i, case_id in enumerate(batch_ids):
#             is_aug = case_id in self.aug_ids
#             img = cv2.resize(cv2.imread(os.path.join(AUG_INPUT_PATH if is_aug else INPUT_PATH,
#                            f'slice_{case_id}_0000.png'), cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#             X[i] = np.expand_dims(scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape), axis=-1)

#             mask = cv2.resize(cv2.imread(os.path.join(AUG_MASK_PATH if is_aug else MASK_PATH,
#                             f'slice_{case_id}_0000.png'), cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#             y[i] = np.expand_dims((mask > 0).astype(np.float32), axis=-1)

#         return X, y

# def create_model():
#     inputs = Input((IMG_SIZE, IMG_SIZE, 1))
#     x = Activation('relu')(BatchNormalization()(Conv2D(32, 3, padding='same', kernel_regularizer=l2(WEIGHT_DECAY))(inputs)))

#     skip1 = ResidualBlock(32)(x)
#     x = MaxPooling2D()(skip1)
#     skip2 = ResidualBlock(64)(x)
#     x = MaxPooling2D()(skip2)
#     skip3 = ResidualBlock(128)(x)
#     x = MaxPooling2D()(skip3)
#     skip4 = ResidualBlock(256)(x)
#     x = MaxPooling2D()(skip4)

#     x = Dropout(0.3)(ResidualBlock(512)(x))

#     x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
#     x = concatenate([AttentionGate(256)(g=x, x=skip4), skip4])
#     x = ResidualBlock(256)(x)

#     x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
#     x = concatenate([AttentionGate(128)(g=x, x=skip3), skip3])
#     x = ResidualBlock(128)(x)

#     x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
#     x = concatenate([AttentionGate(64)(g=x, x=skip2), skip2])
#     x = ResidualBlock(64)(x)

#     x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
#     x = concatenate([AttentionGate(32)(g=x, x=skip1), skip1])
#     x = ResidualBlock(32)(x)

#     outputs = Conv2D(1, 1, activation='sigmoid')(x)
#     model = Model(inputs, outputs)
#     model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=INITIAL_LR, weight_decay=WEIGHT_DECAY),
#                  loss=combined_loss,
#                  metrics=[tf.keras.metrics.BinaryAccuracy(dtype=tf.float32), dice_coeff, precision, iou])
#     return model

# def main():
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             tf.config.experimental.set_virtual_device_configuration(gpu,
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])

#     case_ids = sorted(list({f.split('_')[1] for f in os.listdir(INPUT_PATH) if f.endswith('.png')}))
#     aug_case_ids = sorted(list({f.split('_')[1] for f in os.listdir(AUG_INPUT_PATH) if f.endswith('.png')})) if os.path.exists(AUG_INPUT_PATH) else []

#     train_ids, test_ids = train_test_split(case_ids, test_size=0.15, random_state=42)
#     train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)

#     model = create_model()
#     model.fit(AugmentedDataGenerator(train_ids, aug_ids=aug_case_ids),
#              validation_data=AugmentedDataGenerator(val_ids, shuffle=False),
#              epochs=80,
#              callbacks=[
#                  ModelCheckpoint('best_model.h5', monitor='val_dice_coeff', mode='max', save_best_only=True),
#                  EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
#                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
#                  tf.keras.callbacks.CSVLogger('training_log.csv')
#              ],
#              workers=4,
#              use_multiprocessing=True)

#     print("Test Results:", dict(zip(model.metrics_names,
#           model.evaluate(AugmentedDataGenerator(test_ids, shuffle=False)))))

# def test_saved_model():
#     best_model = tf.keras.models.load_model('best_model.h5',
#                                           custom_objects={'dice_coeff': dice_coeff, 'iou': iou,
#                                                         'precision': precision, 'combined_loss': combined_loss,
#                                                         'ResidualBlock': ResidualBlock, 'AttentionGate': AttentionGate})

#     case_ids = sorted(list({f.split('_')[1] for f in os.listdir(INPUT_PATH) if f.endswith('.png')}))
#     _, test_ids = train_test_split(case_ids, test_size=0.15, random_state=42)
#     test_gen = AugmentedDataGenerator(test_ids, shuffle=False)

#     case_metrics = []
#     for i, (test_batch, test_masks) in enumerate(test_gen):
#         predictions = tf.cast(best_model.predict(test_batch) > 0.5, tf.float32)
#         for j in range(len(test_batch)):
#             case_metrics.append({
#                 'case': f"{test_ids[i]}_{j}",
#                 'dice': float(dice_coeff(test_masks[j:j+1], predictions[j:j+1])),
#                 'iou': float(iou(test_masks[j:j+1], predictions[j:j+1]))
#             })

#     print("\nTest Set Statistics:")
#     print(f"Dice Score: {np.mean([m['dice'] for m in case_metrics]):.4f} ± {np.std([m['dice'] for m in case_metrics]):.4f}")
#     print(f"IoU Score: {np.mean([m['iou'] for m in case_metrics]):.4f} ± {np.std([m['iou'] for m in case_metrics]):.4f}")

# if __name__ == "__main__":
#     main()
#     test_saved_model()












# anitha ma'am idea (unet instead of attention unet)

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

BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
AUG_PATH = "/home/user/adithyaes/dataset/isles2022_png_aug"
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
AUG_INPUT_PATH = os.path.join(AUG_PATH, "input")
AUG_MASK_PATH = os.path.join(AUG_PATH, "mask")
OUTPUT_DIRECTORY = "./output/ISLESfolder"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

IMG_SIZE = 112
BATCH_SIZE = 4  # 2
LEARNINGRATE = 0.001 # 0.0001
EPOCHS = 100 # 30
EARLYSTOPPING = 40 # 10
scaler = MinMaxScaler(feature_range=(-1, 1))  # Scale to match original nii range better

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

def conv_block(inp, filters):
    x = Conv2D(filters, 3, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def attention_gate(x, g, filters):
    g1 = Conv2D(filters, 1)(g)
    x1 = Conv2D(filters, 1)(x)
    out = Activation('relu')(add([g1, x1]))
    out = Conv2D(1, 1, activation='sigmoid')(out)
    return multiply([x, out])

def get_case_ids(path):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    return sorted(list({f.split('_')[1] for f in files}))

def load_and_preprocess(file_path, is_mask=False):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if not is_mask:
        img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    else:
        img = img / 255.0
    return img

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, aug_ids=None, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.aug_ids = aug_ids if aug_ids is not None else []
        self.all_ids = list_IDs + self.aug_ids
        self.shuffle = shuffle
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

                X.append(load_and_preprocess(img_path))
                y.append(load_and_preprocess(mask_path, is_mask=True))

        return np.expand_dims(np.array(X), -1), np.array(y)

def create_model():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))

    # Encoder with reduced filters
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

    # Decoder with attention
    x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    #x = attention_gate(skip4, x, 256)
    x = concatenate([x, skip4])
    x = conv_block(x, 256)

    x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    #x = attention_gate(skip3, x, 128)
    x = concatenate([x, skip3])
    x = conv_block(x, 128)

    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    #x = attention_gate(skip2, x, 64)
    x = concatenate([x, skip2])
    x = conv_block(x, 64)

    x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    #x = attention_gate(skip1, x, 32)
    x = concatenate([x, skip1])
    x = conv_block(x, 32)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
        loss=dice_loss,
        metrics=['accuracy', dice_coeff, iou]
    )

    return model

def main():
    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    case_ids = get_case_ids(INPUT_PATH)
    aug_case_ids = get_case_ids(AUG_INPUT_PATH) if os.path.exists(AUG_INPUT_PATH) else []

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Aug: {len(aug_case_ids)}")

    train_gen = DataGenerator(train_ids, aug_ids=aug_case_ids)
    val_gen = DataGenerator(val_ids, shuffle=False)
    test_gen = DataGenerator(test_ids, shuffle=False)

    model = create_model()

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_dice_coeff', mode='max',
                       save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
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