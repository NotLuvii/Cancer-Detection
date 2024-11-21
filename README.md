# Cancer Identifier Program

## Overview
This program is a machine learning-based **Cancer Identifier** designed to classify tumor diagnoses as **benign** or **malignant** using a dataset of tumor features. It leverages a neural network model built with **TensorFlow/Keras** and provides detailed performance metrics and visualizations to help understand the model's predictions.

## Features
- **Data Preprocessing**:
  - Automatically normalizes input features using `StandardScaler` for optimal model performance.
  - Splits the dataset into training and testing sets to evaluate model generalization.

- **Neural Network Model**:
  - Built with three layers:
    - Input layer with 128 neurons and ReLU activation.
    - Hidden layer with 64 neurons and ReLU activation.
    - Output layer with a sigmoid activation for binary classification.
  - Incorporates dropout layers for regularization and prevents overfitting.

- **Performance Metrics**:
  - Calculates precision, recall, F1-score, and accuracy for model evaluation.
  - Generates a confusion matrix to visualize classification performance.

- **Visualizations**:
  - **Training Progress**: Plots training and validation loss/accuracy over epochs.
  - **Prediction Probabilities**: Displays the probability of malignancy for each test case.
  - **Confusion Matrix**: Heatmap highlighting true/false positive and negative predictions.

- **Interpretability**:
  - Provides user-friendly insights into the model's performance, such as the likelihood of malignancy and key statistical metrics.

## Dataset
The program works with a CSV dataset containing tumor features and a diagnosis label:
- **Input Features**: Various numerical attributes of tumors.
- **Target Label**: A binary column (`diagnosis(1=m, 0=b)`), where:
  - `1` represents malignant tumors.
  - `0` represents benign tumors.

## Outputs
- **Training Metrics**:
  - Loss and accuracy curves for training and validation data.
- **Confusion Matrix**:
  - A heatmap showing true positives, false positives, true negatives, and false negatives.
- **Probability Distribution**:
  - A scatter plot of predicted probabilities for malignant cancer, with actual diagnoses color-coded for clarity.
- **Classification Report**:
  - Precision, recall, F1-score, and support for each class.
