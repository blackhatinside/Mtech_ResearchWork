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
from scipy.ndimage import distance_transform_edt

# Only original dataset path
BASE_PATH = "/home/user/adithyaes/dataset/isles2022_png"
INPUT_PATH = os.path.join(BASE_PATH, "input")
MASK_PATH = os.path.join(BASE_PATH, "mask")
OUTPUT_DIRECTORY = "./output/ISLESfolder_original_only"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

IMG_SIZE = 112
BATCH_SIZE = 32
LEARNINGRATE = 0.0001
EPOCHS = 100
EARLYSTOPPING = 60
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

def get_distance(mask):
    """Return the signed distance map for a binary mask."""
    return np.where(mask,
                   -distance_transform_edt(mask),
                   distance_transform_edt(1-mask))

def carvemix(img1, img2, mask1, mask2):
    """Perform CarveMix augmentation on a pair of images."""
    # Compute signed distance map
    dist_map = get_distance(mask1)
    dist_min = np.min(dist_map)

    # Sample lambda from mixture of uniform distributions
    if np.random.random() < 0.5:
        lam = np.random.uniform(-0.5 * abs(dist_min), 0)
    else:
        lam = np.random.uniform(0, abs(dist_min))

    # Create carved region mask
    carved_mask = (dist_map <= lam).astype(np.float32)

    # Mix images and masks
    mixed_img = img1 * carved_mask + img2 * (1 - carved_mask)
    mixed_mask = mask1 * carved_mask + mask2 * (1 - carved_mask)

    return mixed_img, mixed_mask

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]

        X, y = [], []
        for case_id in batch_ids:
            # Load original images
            input_files = sorted([f for f in os.listdir(INPUT_PATH)
                               if f.startswith(f'slice_{case_id}') and f.endswith('.png')])

            for f in input_files:
                img1 = load_and_preprocess(os.path.join(INPUT_PATH, f))
                mask1 = load_and_preprocess(os.path.join(MASK_PATH, f), is_mask=True)

                # Randomly select another image for mixing
                other_case = np.random.choice(self.list_IDs)
                other_files = sorted([f for f in os.listdir(INPUT_PATH)
                                   if f.startswith(f'slice_{other_case}') and f.endswith('.png')])

                if len(other_files) > 0:
                    other_file = np.random.choice(other_files)
                    img2 = load_and_preprocess(os.path.join(INPUT_PATH, other_file))
                    mask2 = load_and_preprocess(os.path.join(MASK_PATH, other_file), is_mask=True)

                    # Apply CarveMix
                    mixed_img, mixed_mask = carvemix(img1, img2, mask1, mask2)
                    X.append(mixed_img)
                    y.append(mixed_mask)
                else:
                    X.append(img1)
                    y.append(mask1)

        return np.expand_dims(np.array(X), -1), np.expand_dims(np.array(y), -1)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
		loss=dice_loss
		metrics=['accuracy', dice_coeff, iou]
	)

	return model

def main():
	# GPU memory growth
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	# Only get case IDs from original dataset
	case_ids = get_case_ids(INPUT_PATH)

	train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
	train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

	print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

	# Create data generators without augmented data
	train_gen = DataGenerator(train_ids)
	val_gen = DataGenerator(val_ids, shuffle=False)
	test_gen = DataGenerator(test_ids, shuffle=False)

	model = create_model()

	print("Model Summary: ", model.summary())

	callbacks = [
		ModelCheckpoint(os.path.join(OUTPUT_DIRECTORY, 'best_model.h5'), 
                      monitor='val_dice_coeff', mode='max',
                      save_best_only=True, verbose=1),
		EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING, 
                    restore_best_weights=True, verbose=1),
		ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
	]

	history = model.fit(
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
	
	# Save the results to a text file
	with open(os.path.join(OUTPUT_DIRECTORY, 'results.txt'), 'w') as f:
		f.write("Test Results:\n")
		f.write(f"Loss: {results[0]:.4f}\n")
		f.write(f"Accuracy: {results[1]:.4f}\n")
		f.write(f"Dice Score: {results[2]:.4f}\n")
		f.write(f"IoU: {results[3]:.4f}\n")

if __name__ == "__main__":
	main()
