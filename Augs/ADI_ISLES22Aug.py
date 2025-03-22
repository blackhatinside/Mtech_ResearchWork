import cv2
import numpy as np
import os
import albumentations as A
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

if os.name == 'nt':
    base_path = "C:\\Cyberkid\\MyMTech\\Labwork\\SecondYear\\MyWork\\Datasets\\ISLES-2022\\ISLES-2022\\"
else:
    base_path = "/home/user/adithyaes/dataset/isles2022_png/"

# transform = A.Compose([
#     A.Resize(height=128, width=128),  # Resize to a standard size
#     A.RandomCrop(height=64, width=64),  # Random crop for local context
#     A.HorizontalFlip(p=0.5),  # Horizontal flip to augment the dataset
#     A.VerticalFlip(p=0.5),  # Vertical flip for additional variation
#     A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
#     A.Rotate(limit=30, p=0.5),  # Rotate by up to 30 degrees to prevent orientation bias
#     A.GaussNoise(var_limit=(10, 20), p=0.5),  # Add noise but keep it moderate
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  # Shift, scale, and rotate
#     A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),  # Elastic transformations to mimic real-world variations
#     A.Normalize(),  # Normalize the pixel values
#     # A.pytorch.transforms.ToTensorV2()  # Convert to PyTorch tensor if needed
# ])

# # # # # Global Variables

IMG_SIZE=112
PATH_DATASET = base_path
PATH_RAWDATA = os.path.join(base_path, "input")
PATH_DERIVATIVES = os.path.join(base_path, "mask")
PATH_OUTPUTRAWDATA = "./output/input"
PATH_OUTPUTDERIVATIVES = "./output/mask"
os.makedirs(PATH_OUTPUTRAWDATA, exist_ok=True)
os.makedirs(PATH_OUTPUTDERIVATIVES, exist_ok=True)

print("PATH_RAWDATA: ", PATH_RAWDATA)
print("PATH_DERIVATIVES: ", PATH_DERIVATIVES)
print("No of Folders Inside Training: ", len(os.listdir(PATH_RAWDATA)))
print("No of Folders Inside Ground Truth: ", len(os.listdir(PATH_DERIVATIVES)))

# # # # # Functions

def filter_and_load_images(verbose=True):
    """Filter out black images and load valid pairs with value tracking"""

    filtered_dwi = []
    filtered_masks = []

    for case_dir in train_subdirs:
        print(case_dir, end = "\t")
        case_name = os.path.basename(os.path.normpath(case_dir))
        files = os.listdir(case_dir)

        dwi_slices = sorted([f for f in files if '_1_slice_' in f])
        mask_slices = sorted([f for f in files if '_5_slice_' in f])

        for dwi_file, mask_file in zip(dwi_slices, mask_slices):
            dwi_path = os.path.join(case_dir, dwi_file)
            mask_path = os.path.join(case_dir, mask_file)

            dwi = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Only check for non-black mask
            if np.any(mask):

                # Pad the images
                dwi_padded = pad_image(dwi)
                mask_padded = pad_image(mask)

                filtered_dwi.append(dwi_padded)
                filtered_masks.append(mask_padded)

    if verbose:
        print(f"Total image pairs processed: {total_pairs}")
        print(f"Non-black image pairs found: {non_black_pairs}")

        # Show sample of normalized values
        dwi_array = np.array(filtered_dwi)
        print(f"Normalized DWI value range: [{dwi_array.min():.2f}, {dwi_array.max():.2f}]")

    return np.array(filtered_dwi), np.array(filtered_masks)

def create_output_directories():
    """Create output directories if they don't exist"""
    os.makedirs(PATH_OUTPUTRAWDATA, exist_ok=True)
    os.makedirs(PATH_OUTPUTDERIVATIVES, exist_ok=True)

def get_paired_image_paths():
    """Get sorted pairs of raw and mask image paths"""
    raw_files = sorted([f for f in os.listdir(PATH_RAWDATA) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(PATH_DERIVATIVES) if f.endswith('.png')])

    # Ensure matching filenames
    paired_files = []
    for raw_f in raw_files:
        base_name = raw_f.replace('_1_slice_', '_5_slice_')
        if base_name in mask_files:
            paired_files.append((raw_f, base_name))

    return paired_files

def load_and_check_image_pair(raw_path, mask_path):
    """Load image pair and check if mask contains any white pixels"""
    raw_img = cv2.imread(os.path.join(PATH_RAWDATA, raw_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(os.path.join(PATH_DERIVATIVES, mask_path), cv2.IMREAD_GRAYSCALE)

    if raw_img is None or mask_img is None:
        return None, None, False

    # Ensure same dimensions
    if raw_img.shape != mask_img.shape:
        mask_img = cv2.resize(mask_img, (raw_img.shape[1], raw_img.shape[0]))

    # Check if mask contains any white pixels
    has_annotation = np.any(mask_img > 0)

    return raw_img, mask_img, has_annotation

def apply_augmentation(raw_img, mask_img, transform):
    """Apply the same augmentation to both raw image and mask"""
    # Convert mask to binary
    mask_img = (mask_img > 127).astype(np.uint8) * 255

    # Apply same random augmentation to both images
    augmented = transform(image=raw_img, mask=mask_img)
    aug_raw = augmented['image']
    aug_mask = augmented['mask']

    # Ensure mask remains binary
    aug_mask = (aug_mask > 127).astype(np.uint8) * 255

    # Verify alignment
    if np.any(mask_img > 0):
        original_positions = np.where(mask_img > 0)
        augmented_positions = np.where(aug_mask > 0)
        if len(original_positions[0]) != len(augmented_positions[0]):
            return None, None

    return aug_raw, aug_mask

def save_augmented_pair(raw_img, mask_img, original_filename, aug_index):
    """Save augmented image pair with appropriate naming"""
    base_name = os.path.splitext(original_filename)[0]

    raw_output_path = os.path.join(PATH_OUTPUTRAWDATA, f"{base_name}_aug_{aug_index}.png")
    mask_output_path = os.path.join(PATH_OUTPUTDERIVATIVES, f"{base_name}_aug_{aug_index}.png")

    cv2.imwrite(raw_output_path, raw_img)
    cv2.imwrite(mask_output_path, mask_img)

    return raw_output_path, mask_output_path

def display_random_samples(saved_pairs, num_samples=5):
    """Display random samples of augmented image pairs"""
    if len(saved_pairs) < num_samples:
        num_samples = len(saved_pairs)

    selected_pairs = random.sample(saved_pairs, num_samples)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Random Samples of Augmented Images\nTop: Raw Images, Bottom: Mask Images', fontsize=12)

    for idx, (raw_path, mask_path) in enumerate(selected_pairs):
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if raw_img is None or mask_img is None:
            continue

        # Verify alignment before displaying
        if np.any(mask_img > 0):
            mask_positions = np.where(mask_img > 0)
            raw_region = raw_img[
                max(0, min(mask_positions[0]) - 5):min(raw_img.shape[0], max(mask_positions[0]) + 5),
                max(0, min(mask_positions[1]) - 5):min(raw_img.shape[1], max(mask_positions[1]) + 5)
            ]
            if np.std(raw_region) < 1:  # Check if the region has variation
                continue

        axes[0, idx].imshow(raw_img, cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Sample {idx + 1}')

        axes[1, idx].imshow(mask_img, cmap='gray')
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.show()

# # # # # Classes



# # # # # Main Program

def main():
    print("Starting medical image augmentation...")

    create_output_directories()

    # Modified transform to ensure better alignment
    transform = A.Compose([
        A.Resize(height=128, width=128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # Reduced shift limit
            scale_limit=0.1,
            rotate_limit=15,     # Reduced rotation limit
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
    ], additional_targets={'mask': 'image'})

    image_pairs = get_paired_image_paths()
    print(f"Found {len(image_pairs)} image pairs")

    valid_samples = 0
    augmentation_count = 3
    processed_pairs = 0
    saved_pairs = []

    for raw_file, mask_file in tqdm(image_pairs):
        raw_img, mask_img, has_annotation = load_and_check_image_pair(raw_file, mask_file)

        if raw_img is None or not has_annotation:
            continue

        valid_samples += 1
        for aug_idx in range(augmentation_count):
            aug_raw, aug_mask = apply_augmentation(raw_img, mask_img, transform)

            if aug_raw is not None and aug_mask is not None:
                raw_path, mask_path = save_augmented_pair(aug_raw, aug_mask, raw_file, aug_idx)
                saved_pairs.append((raw_path, mask_path))
                processed_pairs += 1

    print(f"Processing complete. Generated {len(saved_pairs)} valid augmented pairs from {valid_samples} annotated pairs.")

    print("\nDisplaying random samples of augmented images...")
    display_random_samples(saved_pairs)

if __name__ == "__main__":
    main()