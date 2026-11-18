# Protein Source Recognition

**MSDS 565 — Predictive Modeling & Analytics**

An image classification and explainable AI project that compares three pretrained ImageNet convolutional neural networks for recognizing five protein sources in food photographs.

## 1. Project Overview

This project uses transfer learning to classify food photographs into five categories:

- Beef
- Chicken
- Salmon
- Shrimp
- Tofu

Three pretrained Keras applications are used as frozen feature extractors. A separate trained linear classification probe maps each model's extracted features to the five target classes.

The project also investigates model interpretability using saliency maps, SmoothGrad, and Grad-CAM.

An interactive Streamlit application allows users to upload food photographs and compare model predictions and explanations.

## 2. Dataset and Preprocessing

The original dataset consisted of 1,750 collected food images, with 350 images per class.

After duplicate removal, validation, and manual cleaning, 1,569 images remained.

| Class | Images |
|---|---:|
| Beef | 334 |
| Chicken | 319 |
| Salmon | 313 |
| Shrimp | 308 |
| Tofu | 295 |
| **Total** | **1,569** |

Preprocessing included:

- Converting images to RGB
- Resizing to 224 × 224 pixels with aspect-ratio-preserving black padding
- Stratified 80/20 training and testing split
- Applying model-specific input preprocessing
- Using training-only balancing and augmentation

The final dataset contains 1,255 training images and 314 testing images.

The repository includes the dataset manifest but excludes the original food photographs and processed