# Diabetes Prediction & Data Analytics using Machine Learning

A compact, reproducible pipeline for predicting diabetes from clinical measurements using Logistic Regression and k-Nearest Neighbors (KNN). The project also includes data analytics techniques such as visualizing, preprocessing, and feature engineering on the diabetes dataset.

> **Disclaimer:** This repository is for research and education. It is **not** a medical device and must not be used for clinical decision-making.

---

## 🔎 Overview

This project builds binary classifiers to predict whether a patient is diabetic (`Outcome = 1`) or non-diabetic (`Outcome = 0`). Two models are implemented and compared:

- **Logistic Regression (LR):** Accuracy ≈ **0.76**  
- **KNN (k = 21):** Accuracy ≈ **0.79**, ROC AUC ≈ **0.83**

KNN provided better precision and F1 for the positive class compared to LR.

---

## 🧰 Tech Stack

- **Language:** Python (3.9+)
- **Core libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- **EDA / Data handling:** `pyspark` (schema/stats), `pandas`
- **Environment:** Works locally or in Google Colab

---

## 📦 Dataset

- **Samples:** 768 patients (female, Pima heritage, ≥ 21 years)  
- **Features (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`  
- **Target (1):** `Outcome` ∈ {0,1}  
- **Split:** 70% train / 30% test
  
---

## 🧪 Methods

### 1) Analyzing and Visualizing the data using PySpark and Pandas.

### 2) Preprocessing
- **Missing/invalid values:** Replaced with **feature medians** (for zero/NA where clinically invalid).  
- **Outliers:**  
  - **IQR filtering:** drop points outside 1.5×IQR.  
  - **Local Outlier Factor (LOF):** flagged observations using Negative Outlier Factor; threshold around **−1.76** (tuned to remove sparse-density anomalies).  
- **Normalization:** Standardization (z-score).

### 3) Feature Engineering
- **Logical bins** for **BMI**, **Glucose**, and **Insulin** (e.g., numeric ranges for low/normal/elevated).  
- **Encoding:** One-Hot Encoding for the engineered categorical features.

### 4) Models
- **Logistic Regression:** baseline linear classifier.  
- **KNN:** Euclidean distance; **hyperparameter sweep** for `k` in **[1…30]** → chose **k = 21** (lowest error).

### 5) Metrics
- **Accuracy**, **Precision/Recall/F1**, **Confusion Matrix**, **ROC AUC**.

---

## 📈 Results 

| Model | Precision (class=1) | Recall (class=1) | F1 (class=1) | Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.61 | 0.60 | 0.60 | **0.76** |
| KNN (k=21) | 0.66 | 0.61 | 0.64 | **0.79** |

**Confusion matrices (selected):**
- **LR:** TP=43, TN=129, FP=29, FN=27  
- **KNN (k=21):** TP=43, TN=136, FP=22, FN=27
  
**ROC AUC:** ≈ **0.83** for KNN.

---
