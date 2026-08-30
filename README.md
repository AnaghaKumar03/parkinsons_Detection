# Multi-Modal Attention Model for Parkinson's Disease Detection

📄 **Published:** [Interpretable AI for Parkinson's Disease: Multimodal deep learning with explainable attention for Parkinson's Disease diagnosis](https://www.sciencedirect.com/science/article/abs/pii/S1746809426005604?via%3Dihub) — *Biomedical Signal Processing and Control*, Volume 120, Part A, 2026. DOI: [10.1016/j.bspc.2026.110006](https://doi.org/10.1016/j.bspc.2026.110006)

## Overview
This project presents a multi-modal, attention-based deep learning framework for Parkinson's Disease (PD) diagnosis using both handwriting and speech data. It combines CNN-based handwriting feature extraction with a VAE-Transformer speech model, fused via a cross-modal attention mechanism, and uses SHAP/LIME for interpretability.

## Results
Evaluated on the Parkinson's Drawings dataset (55 participants) and the Italian Parkinson's Voice and Speech dataset (65 participants):

| Metric | Score |
|---|---|
| Accuracy | 93% |
| Precision | 94% |
| Recall | 92% |
| F1-score | 0.93 |

## Data Sources
- **Handwriting Data:** Spiral and wave drawings, analyzed via deep convolutional architectures.
- **Voice Data:** Audio recordings processed through a Variational Autoencoder (VAE) and Transformer-based model.

## Model Architecture
- **Handwriting Model:** InceptionV3 and DenseNet201 for hierarchical feature extraction from drawings.
- **Voice Model:** VAE for dimensionality reduction + Transformer for sequence modeling of vocal patterns.
- **Fusion Strategy:** A cross-modal attention mechanism that adaptively weighs handwriting vs. speech features per patient.

## Key Features
- **Multi-Modal Learning:** Joint processing of handwriting and voice data for more robust diagnosis across heterogeneous symptom presentations.
- **Adaptive Attention Fusion:** Dynamically balances modality contributions based on individual symptom profiles.
- **Explainability:** SHAP and LIME provide global and per-patient interpretability, aligned with known PD biomarkers.

## Implementation Details
- **Framework:** TensorFlow / Keras
- **Training Environment:** Google Colab (NVIDIA T4 GPU)
- **Optimizer:** Adam (lr = 1e-3), categorical cross-entropy loss, batch size 32, up to 50 epochs
- **Regularization:** Dropout, L2 regularization, early stopping on validation loss

## Future Work
- Expansion of datasets to improve generalization across symptom severity levels.
- Further fine-tuning of the fusion strategy.
- Integration of additional explainability methods.

## Citation
If you use this work, please cite:
```
Anumandla, S.K., Krishnakumar, A., Kumar E., S., Kabilan, K., Sandosh, S. (2026).
Interpretable AI for Parkinson's Disease: Multimodal deep learning with explainable
attention for Parkinson's Disease diagnosis. Biomedical Signal Processing and Control,
120(Part A), 110006. https://doi.org/10.1016/j.bspc.2026.110006
```
