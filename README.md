# Mtech_ResearchWork
Brain Legion Segmentation using Python


Here's a detailed summary of the research project I am working on in my MTech Thesis work and also information about the ISLES22 dataset:

Research Project Overview:
The project focuses on brain lesion segmentation using class-aware augmentation techniques combined with an Attention U-Net architecture. The key objectives were:
- Addressing class imbalance and data scarcity in brain lesion datasets
- Implementing optimized segmentation through Attention U-Net
- Examining effects of various augmentation methods

ISLES22 Dataset Characteristics:
- Multi-center MRI dataset designed for stroke lesion segmentation
- 400 total cases (250 training, 150 test samples)
- Images in NIfTI format converted to PNG (112x112 pixels)
- Contains DWI (Diffusion-weighted imaging) scans with corresponding lesion masks
- Ground truth masks are binary (white for lesion, black for non-lesion)
- Lesions can vary in sizes and shapes, may be as small as a single pixel

Key Methodology:
1. Data Preprocessing:
- Conversion from NIfTI to PNG format
- Intensity normalization using Min-Max scaling
- Standardized image resizing to 112x112 pixels

2. Class-aware Augmentation:
The data was categorized into 5 classes based on lesion size:
- C1: 1-50 pixels (2477 images)
- C2: 51-100 pixels (637 images)
- C3: 101-150 pixels (413 images)
- C4: 151-200 pixels (253 images)
- C5: >200 pixels (1047 images)

Each class received specific augmentation strategies:
- Smaller lesions: More aggressive rotations and transformations
- Larger lesions: Conservative changes to preserve clinical validity

3. Results:
- Initial dataset expanded from 4,827 to 13,174 images
(c1 - 2477 images, c2 - 2548 images, c4 - 2530 images, c4 - 1047 images, c3 - 2478 images)
- Improved Dice score from 0.6651 to 0.7307 with augmentation
- Best performance achieved by U-Net with augmentation (0.7451 Dice score)

The project demonstrated that class-aware augmentation effectively addresses imbalance issues while maintaining clinical relevance, leading to improved segmentation accuracy especially for challenging lesion cases.

The ISLES22 dataset proved valuable for this research due to its:
- Diverse lesion characteristics (size, shape, location)
- Multi-vendor nature reflecting real clinical scenarios
- High-quality annotations
- Standardized evaluation metrics
- Public availability for reproducible research

The dataset's inherent challenges (class imbalance, lesion variability) made it ideal for testing advanced augmentation strategies and segmentation architectures.
