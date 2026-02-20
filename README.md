# Chest X-Ray Pneumonia Detection using Deep Learning

## Overview
This project implements a deep learning–based medical image classification system for detecting **Pneumonia** from chest X-ray images. The solution leverages **Convolutional Neural Networks (CNNs)** and **transfer learning with EfficientNet** to achieve high diagnostic accuracy while prioritizing clinical reliability.

The project follows a complete end-to-end machine learning workflow, including data exploration, preprocessing, model development, fine-tuning, and medical-focused evaluation. The final model is designed as a **clinical decision support tool**, assisting healthcare professionals rather than replacing medical judgment.

---

## Dataset
- **Source:** Chest X-Ray Images Dataset (Kaggle)
- **Classes:** Normal, Pneumonia
- **Modality:** Grayscale chest X-ray images
- **Split:** Training, Validation, Testing

---

## Project Objectives
- Perform exploratory data analysis (EDA) on medical image data
- Build and evaluate CNN-based models for disease detection
- Apply transfer learning to improve performance and generalization
- Handle class imbalance using medical-aware strategies
- Evaluate models using clinically relevant metrics
- Optimize sensitivity to minimize false negatives

---

## Workflow

### 1. Data Exploration
- Analyze class distribution and imbalance
- Inspect image dimensions and quality
- Visualize representative samples
- Validate dataset integrity

---

### 2. Data Preprocessing
- Resize images to `224 × 224`
- Normalize pixel values
- Apply data augmentation to improve generalization
- Handle class imbalance using weighted loss

---

### 3. Baseline Model
- Implement a custom CNN architecture
- Train and evaluate baseline performance
- Identify overfitting and performance bottlenecks

---

### 4. Transfer Learning
- Use **EfficientNetB0** pretrained on ImageNet
- Freeze backbone layers for feature extraction
- Add a custom classification head
- Train using binary cross-entropy loss

---

### 5. Training Strategy
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Callbacks:
  - Early Stopping
  - Reduce Learning Rate on Plateau
  - Model Checkpointing

---

### 6. Fine-Tuning
- Unfreeze selected upper layers of EfficientNet
- Retrain with a reduced learning rate
- Adapt pretrained features to medical imaging domain

---

### 7. Model Evaluation
The model is evaluated using both standard and medical-relevant metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- ROC-AUC
- Confusion Matrix

Special emphasis is placed on **Recall** to minimize false negative predictions.

---

## Results
- Achieved high validation accuracy and ROC-AUC
- Significant improvement over baseline CNN
- Strong sensitivity suitable for medical screening tasks
- Stable generalization across validation and test sets

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle GPU Environment


---

## Limitations
- Binary classification only (Normal vs Pneumonia)
- Dataset limited to X-ray modality
- No patient metadata included

---

## Future Work
- Integrate explainability techniques (Grad-CAM)
- Explore larger EfficientNet variants
- Apply ensemble learning
- Extend to multi-class disease classification
- Incorporate clinical metadata

---

## Disclaimer
This project is intended for **research and educational purposes only** and must not be used as a substitute for professional medical diagnosis.

---

## Author
Developed as part of a deep learning medical imaging project focusing on reliable and interpretable AI solutions.
  

