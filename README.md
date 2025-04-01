# Multi-Modal Attention Model for Parkinson's Disease Detection

## Overview
This project presents a multi-modal attention-based deep learning model for Parkinson's disease detection using both voice and handwriting data. The model leverages advanced feature extraction techniques and an attention mechanism to enhance prediction accuracy.

## Data Sources
- **Handwriting Data**: Spiral and Wave drawings, analyzed using deep convolutional architectures.
- **Voice Data**: Audio recordings processed through a Variational Autoencoder (VAE) and Transformer-based model for feature extraction.

## Model Architecture
- **Handwriting Model**: Utilizes **InceptionV3** and **DenseNet201** for robust feature extraction from image-based handwriting data.
- **Voice Model**: Implements a **VAE** for dimensionality reduction and a **Transformer** for sequence modeling.
- **Fusion Strategy**: A custom attention mechanism combines features from both modalities for final prediction.

## Key Features
- **Multi-Modal Learning**: Joint processing of handwriting and voice data for enhanced accuracy.
- **Deep Feature Extraction**: Uses state-of-the-art architectures for high-quality feature representation.
- **Attention Mechanism**: Improves model interpretability and performance by focusing on key features.
- **Explainability**: SHAP and LIME are used for model interpretability and feature importance analysis.

## Implementation Details
- **Framework**: PyTorch
- **Training Environment**: Kaggle & Google Colab
- **Evaluation Metrics**: Accuracy, F1-score, Precision, Recall

## Results & Comparisons
- The model is compared against existing approaches to Parkinson's detection.
- The combination of **VAE + Transformer** for voice and **InceptionV3 + DenseNet201** for handwriting provides a significant improvement in accuracy over traditional models.

## Future Work
- Expansion of datasets to improve generalization.
- Further fine-tuning of the fusion strategy for better multi-modal representation.
- Integration of additional explainability methods.





