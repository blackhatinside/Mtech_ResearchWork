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
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
OUTPUT_DIRECTORY = "./output/ISLESfolder"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

IMG_SIZE = 112
BATCH_SIZE = 2
EPOCHS = 100
scaler = MinMaxScaler(feature_range=(-1, 1))

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
    return Activation('relu')(x)

def attention_gate(x, g, filters):
    g1 = Conv2D(filters, 1)(g)
    x1 = Conv2D(filters, 1)(x)
    out = Activation('relu')(add([g1, x1]))
    out = Conv2D(1, 1, activation='sigmoid')(out)
    return multiply([x, out])

def has_lesion(mask_path):
    """Check if mask contains any lesion pixels"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.sum(mask > 0) > 0

def get_valid_case_ids(path):
    """Get case IDs that have at least one slice with lesions"""
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    valid_cases = set()

    for f in files:
        mask_path = os.path.join(MASK_PATH, f)
        if has_lesion(mask_path):
            case_id = f.split('_')[1]
            valid_cases.add(case_id)

    return sorted(list(valid_cases))

def load_and_preprocess(file_path, is_mask=False):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if not is_mask:
        img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
    else:
        img = img / 255.0
    return img

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.file_pairs = self._get_valid_file_pairs()
        self.on_epoch_end()

    def _get_valid_file_pairs(self):
        """Get all valid image-mask pairs that contain lesions"""
        valid_pairs = []
        for case_id in self.list_IDs:
            input_files = sorted([f for f in os.listdir(INPUT_PATH)
                                if f.startswith(f'slice_{case_id}') and f.endswith('.png')])

            for f in input_files:
                mask_path = os.path.join(MASK_PATH, f)
                if has_lesion(mask_path):
                    valid_pairs.append((
                        os.path.join(INPUT_PATH, f),
                        mask_path
                    ))
        return valid_pairs

    def __len__(self):
        return int(np.floor(len(self.file_pairs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_pairs = [self.file_pairs[k] for k in indexes]
        return self.__data_generation(batch_pairs)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_pairs):
        X, y = [], []
        for img_path, mask_path in batch_pairs:
            X.append(load_and_preprocess(img_path))
            y.append(load_and_preprocess(mask_path, is_mask=True))
        return np.expand_dims(np.array(X), -1), np.array(y)

def create_model():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))

    # Encoder
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=dice_loss,
        metrics=['accuracy', dice_coeff, iou]
    )

    return model

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Get only case IDs that have lesions
    case_ids = get_valid_case_ids(INPUT_PATH)
    print(f"Total valid cases with lesions: {len(case_ids)}")

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    train_gen = DataGenerator(train_ids)
    val_gen = DataGenerator(val_ids, shuffle=False)
    test_gen = DataGenerator(test_ids, shuffle=False)

    print(f"Train slices: {len(train_gen)*BATCH_SIZE}")
    print(f"Val slices: {len(val_gen)*BATCH_SIZE}")
    print(f"Test slices: {len(test_gen)*BATCH_SIZE}")

    model = create_model()

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_dice_coeff', mode='max',
                       save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
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