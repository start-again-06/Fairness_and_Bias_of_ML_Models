# Fairness-Aware Income Classification using TensorFlow, TFMA, and MinDiff

This repository presents an end-to-end pipeline for training and evaluating a binary income classification model using the ACS Income dataset. The workflow integrates TensorFlow, TensorFlow Model Analysis (TFMA), and MinDiff (from `tensorflow_model_remediation`) to evaluate and mitigate group-based unfairness, particularly with respect to sensitive attributes such as gender.

## Problem Statement
The objective is to predict whether an individual's income exceeds $50,000 based on demographic and socioeconomic features from the ACS 2018 Public Use Microdata Sample. In addition to standard classification performance, the project focuses on quantifying and reducing predictive bias across groups defined by the `SEX` attribute.

## Pipeline Overview

### Data Ingestion and Labeling
- Load ACS 2018 data  
- Binarize income label (`PINCP > 50000 → 1`, else `0`)  

### Model Construction
- Dense feedforward neural network built using the Keras Functional API  

### Training and Evaluation
- Train on 80% of the data  
- Evaluate on the remaining 20% using Accuracy and AUC  

### Fairness Evaluation
- Use TFMA Fairness Indicators to evaluate metrics across gender slices  

### Bias Mitigation using MinDiff
- Apply MMD-based regularization using `MinDiffModel`  
- Minimize distributional divergence between predictions across sensitive groups  

## Dataset Details
- **Source:** Google MLCC ACS Income Dataset  
- **Task:** Binary classification (`PINCP > 50000`)  
- **Sensitive Attribute:** `SEX` (`1.0 = Male`, `2.0 = Female`) 

## Model Architecture
Input (all numerical features)  
↓  
Normalization Layer  
↓  
Dense(64, relu)  
↓  
Dense(32, relu)  
↓  
Dense(1, sigmoid)  

## Compilation Details
- Loss: BinaryCrossentropy  
- Optimizer: Adam  
- Metrics: BinaryAccuracy, AUC  

## Evaluation with TFMA
Model predictions are post-processed and evaluated using TensorFlow Model Analysis (TFMA) with slicing based on the `SEX` attribute to assess group-wise performance and fairness.

### EvalConfig
model_specs {
  prediction_key: "PRED"
  label_key: "PINCP"
}
metrics_specs {
  metrics { class_name: "BinaryAccuracy" }
  metrics { class_name: "AUC" }
  metrics {
    class_name: "FairnessIndicators"
    config: '{"thresholds": [0.50]}'
  }
}
slicing_specs {
  feature_keys: "SEX"
}

### Visualization

## Fairness Remediation using MinDiff

MinDiff introduces a distributional alignment loss between the model outputs of sensitive and non-sensitive groups to mitigate predictive bias.

### MinDiff Training
- Groups:
  - Sensitive group: SEX = 2.0
  - Non-sensitive group: SEX = 1.0
- Loss Function: MMDLoss (Maximum Mean Discrepancy)
- Objective: Align output distributions while preserving classification performance

### MinDiff Model

min_diff_model = min_diff.keras.MinDiffModel(
    original_model=base_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1.0
)

The MinDiff model is evaluated using the same TFMA pipeline as the base model.

## Results Interpretation
- Global Metrics: Accuracy, AUC
- Group Fairness: Fairness Indicators across SEX
- Comparison: Model performance before and after MinDiff-based bias mitigation

## Key Concepts
- Fairness Indicators: Tools for monitoring metric disparities across groups
- MinDiff: Regularization strategy for reducing distributional divergence
- TFMA: Scalable framework for sliced and aggregate model evaluation

