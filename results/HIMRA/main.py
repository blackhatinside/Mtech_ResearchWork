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
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator

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

# Rest of the Code (unchanged)
def load_and_preprocess(file_path, is_mask=False):
	img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	if not is_mask:
		img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
	else:
		img = img / 255.0
	return img

# Modified DataGenerator Class
class HIMRADataGenerator(tf.keras.utils.Sequence):
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

				img = load_and_preprocess(img_path)
				mask = load_and_preprocess(mask_path, is_mask=True)

				# Apply HIMRA Augmentation
				if is_aug:
					img, mask = biomechanical_deformation(img, mask)
					img, mask = simulate_hemodynamics(img, mask)
					img, mask = attention_occlusion(img, mask)

				X.append(img)
				y.append(mask)

		return np.expand_dims(np.array(X), -1), np.array(y)

# HIMRA Augmentation Functions
def biomechanical_deformation(image, mask):
	"""
	Applies biomechanically realistic deformation using stiffness-weighted elastic deformation.
	"""
	# Get lesion centroid
	lesion_pixels = np.where(mask > 0)
	if len(lesion_pixels[0]) == 0:
		return image  # No deformation if no lesion

	centroid = np.array([np.mean(lesion_pixels[0]), np.mean(lesion_pixels[1])])

	# Simulate tissue stiffness (higher stiffness near lesion)
	stiffness_map = np.exp(-0.01 * ((np.indices(image.shape)[0] - centroid[0])**2 +
								   (np.indices(image.shape)[1] - centroid[1])**2))

	# Create deformation field
	dx = stiffness_map * gaussian_filter(np.random.randn(*image.shape), sigma=5)
	dy = stiffness_map * gaussian_filter(np.random.randn(*image.shape), sigma=5)

	# Apply deformation
	y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
	indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
	deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
	deformed_mask = map_coordinates(mask, indices, order=0, mode='constant').reshape(mask.shape)

	return deformed_image, deformed_mask

def simulate_hemodynamics(image, mask):
	"""
	Simulates hemodynamic effects using synthetic ADC map variations.
	"""
	# Create synthetic ADC map
	adc_map = np.random.normal(1.0, 0.2, size=image.shape)
	adc_map = gaussian_filter(adc_map, sigma=3)

	# Apply hemodynamic effect (stronger near lesions)
	if np.sum(mask) > 0:  # Only apply if lesion exists
		adc_map[mask > 0] *= np.random.uniform(0.8, 1.2)

	return image * adc_map, mask

def attention_occlusion(image, mask, attention_map=None):
	"""
	Applies attention-guided occlusion to simulate adversarial conditions.
	"""
	if attention_map is None:
		attention_map = gaussian_filter(np.random.rand(*image.shape), sigma=2)

	# Create occlusion mask
	occlusion_mask = (attention_map > np.percentile(attention_map, 75)).astype(np.float32)
	occlusion_mask = gaussian_filter(occlusion_mask, sigma=2)

	# Apply occlusion
	occluded_image = image * (1 - occlusion_mask)
	return occluded_image, mask

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
		loss=dice_loss,
		metrics=['accuracy', dice_coeff, iou]
	)

	return model

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

	train_gen = HIMRADataGenerator(train_ids, aug_ids=aug_case_ids)
	val_gen = HIMRADataGenerator(val_ids, shuffle=False)
	test_gen = HIMRADataGenerator(test_ids, shuffle=False)

	model = create_model()

	print("Model Summary: ", model.summary())

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
