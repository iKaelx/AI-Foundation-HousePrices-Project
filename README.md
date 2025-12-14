# 🏠 Real Estate Price Prediction

**Machine Learning & Deep Learning (PyTorch ANN)**
**AI Foundation – University Project**

---

## 📌 Project Overview

This project implements a **complete, end-to-end machine learning pipeline** to predict **house prices** using the **King County House Prices dataset** (USA). The solution covers **data understanding, preprocessing, feature engineering, model training, evaluation, comparison, and saving models**.

The project compares **classical ML models** with a **deep Artificial Neural Network (ANN)** built using **PyTorch**, and analyzes **overfitting, underfitting, and generalization**.

---

## 📊 Dataset Description

**Dataset:** King County House Prices Dataset
**Samples:** 21,613 houses
**Target Variable:** `price`

### Main Features

* `bedrooms` – Number of bedrooms
* `bathrooms` – Number of bathrooms
* `sqft_living` – Interior living area (ft²)
* `sqft_lot` – Land size (ft²)
* `floors` – Number of floors
* `waterfront` – Waterfront view (0 / 1)
* `view` – View quality (0–4)
* `condition` – House condition (1–5)
* `grade` – Construction quality (1–13)
* `sqft_above` – Area above ground
* `sqft_basement` – Basement area
* `yr_built` – Construction year
* `yr_renovated` – Renovation year (0 = never)
* `zipcode` – Location (postal code)
* `lat`, `long` – Geographic coordinates
* `sqft_living15`, `sqft_lot15` – Nearby houses statistics

📌 **No NULL values** exist, but some columns contain **structural zeros** that behave like missing data.

---

## 🔍 Data Quality & Analysis

### Missing-Like Values

Although the dataset has **no NULLs**, two columns need special handling:

* **`sqft_basement`**

  * `0` can mean *no basement* or *unknown size*
  * Solution:

    * Keep original column
    * Create `basement_exists` (0 / 1)
    * Create `sqft_basement_imputed` using median of non-zero values

* **`yr_renovated`**

  * `0` means *never renovated*
  * Treated as a **binary feature** (`is_renovated`)

---

## 📈 Outlier Detection

Outliers were analyzed using:

* **IQR (Interquartile Range)**
* **Z-Score**
* **Boxplots (single & multi-feature)**
* **Histograms + KDE**

Common outliers include:

* Houses with extremely high prices (> $5M)
* Very large houses (> 10,000 sqft)
* Abnormal bedroom counts

Outliers were **not removed aggressively**, as tree-based models and ANN can handle them effectively.

---

## 🧠 Problem Formulation

* **Task Type:** Regression
* **Objective:** Predict continuous house prices accurately
* **Evaluation Focus:** Generalization to unseen data

### Target Distribution

* Price distribution is **right-skewed**
* Applied **log transformation (`log1p`)** to stabilize variance and improve learning

---

## ⚙️ Data Preprocessing Pipeline

### 1️⃣ Train / Validation / Test Split

* **70%** Training
* **15%** Validation
* **15%** Testing

Ensures unbiased model evaluation and proper hyperparameter tuning.

---

### 2️⃣ Feature Engineering

* `basement_exists`
* `sqft_basement_imputed`
* `price_log` (log-transformed target)

Dropped features with low predictive value or redundancy:

* `id`
* `date`
* `yr_built`
* `yr_renovated` (replaced)
* `sqft_living15`, `sqft_lot15`

---

### 3️⃣ Encoding Categorical Features

| Feature              | Method                   |
| -------------------- | ------------------------ |
| `zipcode`            | Target Encoding          |
| `condition`, `grade` | Label Encoding (ordinal) |
| Others               | Already numeric          |

---

### 4️⃣ Feature Scaling

* **StandardScaler** used for:

  * Linear Regression
  * SVR
  * ANN

* **Tree-based models** (Decision Tree & Random Forest) do **not require scaling**

---

## 🔥 Models Implemented

| Model                      | Library      | Purpose                   |
| -------------------------- | ------------ | ------------------------- |
| 📈 Linear Regression       | scikit-learn | Baseline regression       |
| 🌲 Decision Tree Regressor | scikit-learn | Interpretable model       |
| 🌳 Random Forest Regressor | scikit-learn | High-performance ensemble |
| 🧠 ANN                     | PyTorch      | Deep learning regression  |
| ⚙️ SVR                     | scikit-learn | Non-linear regression     |

---

## 🧠 Artificial Neural Network (ANN)

### Architecture

* Input Layer → feature dimension
* Hidden Layers:

  * 256 neurons + BatchNorm + ReLU + Dropout (0.3)
  * 128 neurons + BatchNorm + ReLU + Dropout (0.2)
  * 64 neurons + ReLU
* Output Layer → 1 neuron

### Training Configuration

* **Optimizer:** Adam (lr = 0.001)
* **Loss:** MSELoss
* **Scheduler:** ReduceLROnPlateau
* **Batch Size:** 32
* **Epochs:** 200

📌 Log-scale predictions are converted back using `expm1()`.

---

## 🌳 Random Forest Regressor

### Configuration

* `n_estimators = 301`
* `max_depth = None`
* `min_samples_split = 2`
* `min_samples_leaf = 1`
* `n_jobs = -1`

✔ Excellent balance between **bias and variance**
✔ Best **R² score** among models

---

## 🌲 Decision Tree Regressor

### Configuration

* `max_depth = 14`
* `min_samples_split = 20`
* `min_samples_leaf = 10`

✔ Highly interpretable
❌ Slight overfitting compared to Random Forest

---

## ⚙️ Support Vector Regression (SVR)

* Kernel: **RBF**
* `C = 10`
* `epsilon = 0.1`
* Strong performance on non-linear data
* Requires **feature & target scaling**

---

## 📊 Model Performance Comparison

| Model         | Train R²  | Val R²    | Test R²   |
| ------------- | --------- | --------- | --------- |
| ANN           | ~0.91     | ~0.87     | ~0.87     |
| Random Forest | **~0.97** | **~0.89** | **~0.88** |
| Decision Tree | ~0.91     | ~0.85     | ~0.84     |
| SVR           | ~0.92     | ~0.87     | ~0.87     |

📌 **Best R²:** Random Forest
📌 **Lowest MSE:** SVR

---

## ⚠️ Overfitting & Underfitting Analysis

### Overfitting Indicators

* Large gap between train and validation scores
* Decreasing train loss while validation loss increases

### Underfitting Indicators

* Poor performance on both train & validation sets

✔ Random Forest and SVR show **best generalization**

---

## 📂 Project Structure

```text
📦 RealEstate-Price-Prediction
┣ 📁 data/ → Dataset CSV
┣ 📁 models/ → Saved ML & ANN models
┣ 📄 train_models.py → Full training pipeline
┣ 📄 README.md → Project documentation
┣ 📄 requirements.txt → Dependencies
```

---

## 💾 Saved Models

* `LinearRegression.joblib`
* `DecisionTree.joblib`
* `RandomForest.joblib`
* `pytorch_ann_best.pth`
* `preprocessor.joblib`

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train all models

```bash
python train_models.py
```

---

## 🎓 Conclusion

This project demonstrates a **professional ML workflow** including:

* Data understanding & cleaning
* Feature engineering
* Model comparison
* Deep learning with PyTorch
* Performance evaluation & analysis

It highlights why **ensemble models** and **ANNs** outperform simple regressors in real-world house price prediction tasks.

---

✨ *AI Foundation – University Project*
