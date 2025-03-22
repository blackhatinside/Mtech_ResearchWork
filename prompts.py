Goal: I want to balance the ISLES 2022 dataset by applying more augmentations to underrepresented classes. The final output should be a class-imbalanced dataset which can be used for Brain Lesion Segmentation

Tasks

	Load image pair and check if mask contains any white pixels. If it does then consider the image pair as valid for our augmentation

	Next, we will add weights to different classes. For this, we will divide all the valid pairs into 4 different classes
		c1 - pairs whose mask image has [1,50] white pixels 				i.e. 1 to 50 lesion pixels
		c2 - pairs whose mask image has [51, 100] white pixels
		c3 - pairs whose mask image has [101, 150] white pixels
		c4 - pairs whose mask image has [151, 200] white pixels
		c5 - pairs whose mask image has [200, ] white pixels				i.e. 200+ lesion pixels



	Let's say we have this kind of distribution (example):

	# Example distribution
	C1 (1-50 pixels):     100 samples  → needs 400 more to reach 500
	C2 (51-100 pixels):   200 samples  → needs 300 more to reach 500
	C3 (101-150 pixels):  300 samples  → needs 200 more to reach 500
	C4 (151-200 pixels):  400 samples  → needs 100 more to reach 500
	C5 (200+ pixels):     500 samples  → reference class (max samples)

	Here's how to approach the augmentation for each class:

		C1 (needs 4x more samples):

		transform_C1 = A.Compose([
		    A.OneOf([
		        A.RandomBrightnessContrast(p=0.8),
		        A.GaussNoise(p=0.8),
		        A.Blur(blur_limit=3, p=0.8),
		    ], p=1.0),
		    A.OneOf([
		        A.Rotate(limit=15, p=0.8),
		        A.Scale(scale_limit=0.1, p=0.8),
		    ], p=1.0),
		], p=1.0)
		# Apply 4 different variations of transforms to each image

		C2 (needs 2.5x more samples):

		transform_C2 = A.Compose([
		    A.OneOf([
		        A.RandomBrightnessContrast(p=0.7),
		        A.GaussNoise(p=0.7),
		        A.Blur(blur_limit=3, p=0.7),
		    ], p=1.0),
		    A.OneOf([
		        A.Rotate(limit=20, p=0.7),
		        A.Scale(scale_limit=0.15, p=0.7),
		        A.ElasticTransform(alpha=30, sigma=3, p=0.7),
		    ], p=1.0),
		], p=1.0)
		# Apply 3 different variations to each image

		C3 (needs 1.67x more samples):

		transform_C3 = A.Compose([
		    A.OneOf([
		        A.RandomBrightnessContrast(p=0.6),
		        A.GaussNoise(p=0.6),
		    ], p=1.0),
		    A.OneOf([
		        A.Rotate(limit=30, p=0.6),
		        A.ElasticTransform(alpha=40, sigma=4, p=0.6),
		        A.GridDistortion(p=0.6),
		    ], p=1.0),
		], p=1.0)
		# Apply 2 different variations to each image

		C4 (needs 1.25x more samples):

		transform_C4 = A.Compose([
		    A.OneOf([
		        A.RandomBrightnessContrast(p=0.5),
		        A.Rotate(limit=45, p=0.5),
		        A.ElasticTransform(alpha=50, sigma=5, p=0.5),
		    ], p=1.0)
		], p=1.0)
		# Apply 1-2 variations to each image

		C5: No augmentation needed (reference class)

	Implementation strategy:

		1. For each class below the target count:

		def augment_to_target(images, masks, current_count, target_count, transform):
		    augmented_images = []
		    augmented_masks = []
		    multiplier = math.ceil(target_count / current_count)

		    for img, mask in zip(images, masks):
		        for _ in range(multiplier - 1):  # -1 because we already have original
		            augmented = transform(image=img, mask=mask)
		            augmented_images.append(augmented['image'])
		            augmented_masks.append(augmented['mask'])

		    return augmented_images, augmented_masks

		2. Validate results:

		def validate_augmentation(mask):
		    """Ensure augmented mask still contains valid lesions"""
		    white_pixels = np.sum(mask > 0)
		    return white_pixels > 0

		3. Keep track of class distribution:

		def get_class_distribution(masks):
		    distribution = {
		        'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0
		    }
		    for mask in masks:
		        white_pixels = np.sum(mask > 0)
		        if 1 <= white_pixels <= 50: distribution['C1'] += 1
		        elif 51 <= white_pixels <= 100: distribution['C2'] += 1
		        elif 101 <= white_pixels <= 150: distribution['C3'] += 1
		        elif 151 <= white_pixels <= 200: distribution['C4'] += 1
		        else: distribution['C5'] += 1
		    return distribution










 In Albumentations, when you apply the same transform to both image and mask using the same transform object, they remain perfectly aligned. That's why we use:

# This ensures synchronized transformation
augmented = transform(image=img, mask=mask)
augmented_image = augmented['image']
augmented_mask = augmented['mask']


However, we should add validation steps to ensure quality:


def validate_augmented_pair(image, mask, original_image, original_mask):
    """Validate augmented image-mask pair"""

    # 1. Check if mask still contains lesions
    white_pixels = np.sum(mask > 0)
    if white_pixels == 0:
        return False

    # 2. Check if lesion intensity in image corresponds to mask
    lesion_region = image[mask > 0]
    non_lesion_region = image[mask == 0]

    # Compare intensity distributions
    lesion_mean = np.mean(lesion_region)
    non_lesion_mean = np.mean(non_lesion_region)

    # Lesions typically appear brighter in T2 MRI
    if lesion_mean <= non_lesion_mean:
        return False

    # 3. Check if relative positions maintained
    original_center = np.mean(np.where(original_mask > 0), axis=1)
    new_center = np.mean(np.where(mask > 0), axis=1)

    # Allow for some movement but not extreme
    max_allowed_shift = min(image.shape) * 0.3  # 30% of image dimension
    if np.linalg.norm(original_center - new_center) > max_allowed_shift:
        return False

    return True


Modified augmentation pipeline with validation:


def safe_augment_pair(image, mask, transform, max_attempts=10):
    """Attempt augmentation with validation"""
    for _ in range(max_attempts):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        if validate_augmented_pair(aug_image, aug_mask, image, mask):
            return aug_image, aug_mask

    # If all attempts fail, return original
    return image, mask

def augment_to_target(images, masks, current_count, target_count, transform):
    augmented_images = []
    augmented_masks = []
    multiplier = math.ceil(target_count / current_count)

    for img, mask in zip(images, masks):
        for _ in range(multiplier - 1):
            aug_img, aug_mask = safe_augment_pair(img, mask, transform)
            # Additional quality checks
            if is_valid_augmentation(aug_img, aug_mask):
                augmented_images.append(aug_img)
                augmented_masks.append(aug_mask)

    return augmented_images, augmented_masks


Additional safeguards in transforms:


# Safer transform for C1 (very small lesions)
transform_C1 = A.Compose([
    # Pixel-level transforms that preserve intensity relationships
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # Limited brightness change
            contrast_limit=0.1,    # Limited contrast change
            p=0.8
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.8),
    ], p=1.0),

    # Very conservative geometric transforms
    A.OneOf([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Scale(scale_limit=0.1, p=0.8),
    ], p=1.0),
], p=1.0)

# Transform for larger lesions (C4, C5) can be more aggressive
transform_C4_C5 = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(p=0.8),
        A.GaussNoise(p=0.8),
    ], p=1.0),

    A.OneOf([
        A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.ElasticTransform(
            alpha=120,
            sigma=12,
            alpha_affine=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.8
        ),
    ], p=1.0),
], p=1.0)


Visual validation helper:


def visualize_augmentation_pair(original_image, original_mask,
                              augmented_image, augmented_mask):
    """Visualize original and augmented pairs for manual verification"""
    plt.figure(figsize=(12, 6))

    # Original pair
    plt.subplot(221)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(222)
    plt.imshow(original_mask, cmap='gray')
    plt.title('Original Mask')

    # Augmented pair
    plt.subplot(223)
    plt.imshow(augmented_image, cmap='gray')
    plt.title('Augmented Image')

    plt.subplot(224)
    plt.imshow(augmented_mask, cmap='gray')
    plt.title('Augmented Mask')

    plt.tight_layout()
    plt.show()