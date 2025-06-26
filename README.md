# Chest X-Ray Classifier using Transfer Learning (VGG16)

This project implements a deep learning model for classifying chest X-ray images into different categories (such as Normal, Pneumonia, etc.) using **transfer learning** with the **VGG16** architecture. The goal is to support rapid and accurate medical diagnosis through automated image classification.

## 🩺 Project Overview

- firstly install the chest x ray data set from the kaggle its a genral data set will be avaialable in one search use the zip dataset to run pythonn notebook.

- **Input**: Chest X-ray images  
- **Model**: Pre-trained VGG16 (ImageNet weights)  
- **Techniques**:
  - Transfer Learning  
  - Image Augmentation  
  - Fine-tuning VGG16  
- **Dataset**: Can be downloaded on Kaggle.

## 📁 Folder Structure

/Chest_X-RAY_CLassifier
├── Chest_X-RAY_CLassifier.ipynb # Jupyter notebook with full pipeline.
├── README.md # This file.


## 🧠 Model Architecture

- **Base model**: VGG16 without top layers  
- **Added layers**:
  - Flatten  
  - Dense + Dropout  
  - Batch Normalization  
  - Final Dense (Softmax) for classification  

## 🔄 Training Pipeline

1. **Unzip the dataset**  
2. **Preprocess images** using `ImageDataGenerator` with augmentation  
3. **Load and modify** the VGG16 model  
4. **Train and validate** on the prepared dataset  
5. **Evaluate performance** using accuracy metrics  

## 📊 Evaluation

The model evaluates its performance on the test dataset and visualizes predictions.  
Metrics like **accuracy**, **confusion matrix**, and **loss curves** are used.

## 🚀 Getting Started

```bash
# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook Chest_X-RAY_CLassifier.ipynb
