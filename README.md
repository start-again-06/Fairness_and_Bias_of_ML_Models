# 📊 Fairness-Aware Income Classification using TensorFlow, TFMA, and MinDiff

This repository presents a pipeline for training and evaluating a binary income classification model using the ACS Income dataset. The workflow integrates **TensorFlow**, **TensorFlow Model Analysis (TFMA)**, and **MinDiff** (from `tensorflow_model_remediation`) to assess and mitigate group-based unfairness, particularly with respect to sensitive attributes such as gender.

---

## 🔍 Problem Statement

The goal is to predict whether an individual's income exceeds \$50,000 based on various demographic and socioeconomic features from the **ACS 2018 Public Use Microdata Sample**. Beyond standard classification performance, we aim to **quantify and reduce predictive bias** across groups defined by the `SEX` attribute.

---

## 🧱 Pipeline Overview

1. **Data Ingestion and Labeling**  
   Load ACS 2018 data and binarize income label (`PINCP > 50,000` → 1, else 0).
   
2. **Model Construction**  
   A dense feedforward neural network built using the Keras Functional API.

3. **Training and Evaluation**  
   Model trained on 80% of the data; performance evaluated on the held-out 20% using accuracy and AUC.

4. **Fairness Evaluation**  
   Use TFMA’s Fairness Indicators to evaluate metrics across gender slices.

5. **Bias Mitigation using MinDiff**  
   Incorporate **MMD-based regularization** via `MinDiffModel` to minimize distributional divergence between predictions across sensitive groups.

---

## 📦 Dependencies

Install via pip:

```bash
pip install pandas tensorflow tensorflow_model_analysis tensorflow_model_remediation protobuf
🧾 Dataset Details
Source: Google MLCC ACS Income Dataset

Task: Binary classification (PINCP > 50000)

Sensitive Attribute: SEX (1.0 = Male, 2.0 = Female)

## 🧠 Model Architecture
scss
Copy
Edit
Input (all numerical features)
↓
Normalization Layer
↓
Dense(64, relu)
↓
Dense(32, relu)
↓
Dense(1, sigmoid)
Compiled with:

Loss: BinaryCrossentropy

Optimizer: Adam

Metrics: BinaryAccuracy, AUC

## 🧪 Evaluation with TFMA
Model predictions are post-processed and sliced on the SEX attribute using the following EvalConfig:

protobuf
Copy
Edit
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
Visualized using:

python
Copy
Edit
tfma.addons.fairness.view.widget_view.render_fairness_indicator(result)
## ⚖️ Fairness Remediation using MinDiff
MinDiff introduces a distributional alignment loss between the model outputs of sensitive vs. non-sensitive groups.

MinDiff Training
Groups: Datasets manually split into sensitive (SEX=2.0) and non-sensitive (SEX=1.0) positive examples.

Loss Function: MMDLoss (Maximum Mean Discrepancy)

Objective: Align representation and output distributions while maintaining classification fidelity.

python
Copy
Edit
min_diff_model = min_diff.keras.MinDiffModel(
    original_model=base_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1.0
)
Evaluated identically to the base model via TFMA.

## 📊 Results Interpretation
Model performance is evaluated on:

Global metrics: Accuracy, AUC

Group fairness: Fairness Indicators across SEX

Comparison before and after MinDiff-based remediation

## 🧠 Key Concepts
Fairness Indicators: Visual tools to monitor disparity in model metrics across groups.

MinDiff: A regularization strategy that minimizes distributional divergence for fairness.

TFMA: A scalable evaluation toolkit for sliced and aggregate model metrics.

## 📁 File Structure
vbnet
Copy
Edit
.
├── main.py               # End-to-end pipeline implementation
├── README.md             # Project documentation

