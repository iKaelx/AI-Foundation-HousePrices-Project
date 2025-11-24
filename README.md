# 🏠 Real Estate Price Prediction  
Predicting house prices using Machine Learning + PyTorch ANN  
Made for **AI Foundation University Project**

## 📌 Overview  
This project builds a **complete machine learning pipeline** to predict real estate prices using 4 different models:

### 🔥 Models Used
| Model | Library | Description |
|-------|---------|-------------|
| 🌳 **Random Forest Regressor** | scikit-learn | Powerful ensemble model, great baseline. |
| 🌲 **Decision Tree Regressor** | scikit-learn | Simple and interpretable model. |
| 📈 **Linear Regression** | scikit-learn | Fast baseline regression model. |
| 🧠 **Artificial Neural Network (ANN)** | PyTorch | Deep learning model trained on tabular data. |

All models are trained, validated, tested, compared, and saved.

## 📂 Project Structure
📦 RealEstate-Price-Prediction
┣ 📁 data/ → Put your CSV dataset here
┣ 📁 models/ → Saved ML & ANN models
┣ 📄 train_models.py → Full pipeline code
┣ 📄 README.md → This file
┗ 📄 requirements.txt → Dependencies

## 🧠 How The Pipeline Works (Simple Explanation)

### 1️⃣ **Load the Dataset**  
The CSV file is loaded using `pandas`.  
Dataset is split into:
- **70% Training**
- **15% Validation**
- **15% Testing**

---

### 2️⃣ **Data Preprocessing**
| Step | Icon | Explanation |
|------|------|-------------|
| 🧹 Missing Values | Replace missing values using median/mode |
| 🔢 Scaling | StandardScaler normalizes numeric data |
| 🔤 Encoding | One-Hot Encoding converts text → numbers |
| 🧱 ColumnTransformer | Combines numeric + categorical preprocessing |

This makes the data clean and ready for ML models.

### 3️⃣ **Train ML Models**
Each model learns patterns between features (size, rooms, location…) and price.

- Linear Regression → basic baseline  
- Decision Tree → interpretable  
- Random Forest → more accurate  
- PyTorch ANN → best performance  

### 4️⃣ **Evaluate Accuracy**
Metrics used:
- ✔ MAE (Mean Absolute Error)
- ✔ MSE (Mean Squared Error)
- ✔ R² Score

A summary table is printed at the end comparing 4 models.

### 5️⃣ **Save All Models**
Saved inside `/models/`:
LinearRegression.joblib
DecisionTree.joblib
RandomForest.joblib
pytorch_ann_best.pth
preprocessor.joblib

## 🚀 How to Run
### Install dependencies:
pip install -r requirements.txt

### Train models:
python train_models.py

### Edit dataset path:
Inside `train_models.py`:
```python
DATASET_PATH = "data/house_prices.csv"

