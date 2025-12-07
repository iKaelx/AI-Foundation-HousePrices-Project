import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 1. LOAD THE DATASET

print("Loading dataset...")
DATASET_PATH = "data/kc_house_data.csv"
df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst rows:")
print(df.head())


# 2. CREATE PRICE CATEGORIES

print("\nCreating price categories...")

# Define price ranges
def categorize_price(price):
    if price < 300000:
        return 0  # Cheap
    elif price < 600000:
        return 1  # Medium
    else:
        return 2  # Expensive

df['price_category'] = df['price'].apply(categorize_price)

# Show distribution
print("\nPrice Category Distribution:")
category_names = {0: 'Cheap (<$300k)', 1: 'Medium ($300k-$600k)', 2: 'Expensive (>$600k)'}
for cat, name in category_names.items():
    count = (df['price_category'] == cat).sum()
    percentage = count / len(df) * 100
    print(f"  {name}: {count} houses ({percentage:.1f}%)")


# 3. DATA PREPROCESSING

print("\nPreprocessing data...")

# Remove unnecessary columns
columns_to_drop = ['id', 'date', 'price'] if 'id' in df.columns else ['price']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Handle missing values
print(f"Missing values before: {df.isnull().sum().sum()}")
df = df.fillna(df.median(numeric_only=True))
print(f"Missing values after: {df.isnull().sum().sum()}")

# Separate features (X) and target (y)
X = df.drop('price_category', axis=1)
y = df['price_category']


# 4. SPLIT DATA (70% train, 15% validation, 15% test)

print("\nSplitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# 5. FEATURE SCALING (Normalization)

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# 6. TRAIN LOGISTIC REGRESSION MODEL

print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")


# 7. MAKE PREDICTIONS

print("\nMaking predictions...")
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)


# 8. EVALUATE MODEL PERFORMANCE

print("\n" + "="*60)
print("LOGISTIC REGRESSION - MODEL EVALUATION")
print("="*60)

def evaluate(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{dataset_name} Set:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n  Classification Report:")
    report = classification_report(y_true, y_pred, target_names=['Cheap', 'Medium', 'Expensive'])
    print(report)
    
    return accuracy

train_acc = evaluate(y_train, y_train_pred, "Training")
val_acc = evaluate(y_val, y_val_pred, "Validation")
test_acc = evaluate(y_test, y_test_pred, "Test")


# 9. CONFUSION MATRIX

print("\n" + "="*60)
print("CONFUSION MATRIX (Test Set)")
print("="*60)
cm = confusion_matrix(y_test, y_test_pred)
print("\n                Predicted")
print("              Cheap  Medium  Expensive")
print(f"Actual Cheap    {cm[0][0]:5d}  {cm[0][1]:6d}  {cm[0][2]:9d}")
print(f"      Medium    {cm[1][0]:5d}  {cm[1][1]:6d}  {cm[1][2]:9d}")
print(f"      Expensive {cm[2][0]:5d}  {cm[2][1]:6d}  {cm[2][2]:9d}")


# 10. COMPARISON TABLE

print("\n" + "="*60)
print("COMPARISON WITH OTHER MODELS")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'PyTorch ANN'],
    'Accuracy': [f'{test_acc:.4f}', 'Your teammate\'s result', 'Your teammate\'s result', 'Your teammate\'s result'],
    'Type': ['Classification (3 classes)', 'Classification', 'Classification', 'Regression']
})

print(comparison_df.to_string(index=False))


# 11. SAVE THE MODEL

print("\nSaving model...")
joblib.dump(model, 'models/LogisticRegression.joblib')
joblib.dump(scaler, 'models/preprocessor.joblib')
print("Model saved: models/LogisticRegression.joblib")
print("Scaler saved: models/preprocessor.joblib")

# 12. EXAMPLE PREDICTIONS

print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

category_names = {0: 'Cheap', 1: 'Medium', 2: 'Expensive'}

# Show 5 random predictions
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    actual = category_names[y_test.iloc[idx]]
    predicted = category_names[y_test_pred[idx]]
    match = "CORRECT" if actual == predicted else "WRONG"
    print(f"\nHouse #{idx}:")
    print(f"  Actual Category:    {actual}")
    print(f"  Predicted Category: {predicted} [{match}]")

print("\n" + "="*60)
print("LOGISTIC REGRESSION ANALYSIS COMPLETE!")
print("="*60)


# 13. MODEL INSIGHTS

print("\nModel Insights:")
print(f"  Number of features used: {X_train.shape[1]}")
print(f"  Model type: Logistic Regression (Multi-class classification)")
print(f"  Number of classes: 3 (Cheap, Medium, Expensive)")
print(f"  Best for: Categorizing houses into price ranges")
print(f"  Training time: Very fast")
