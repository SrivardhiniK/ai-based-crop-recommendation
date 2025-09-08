# baseline_model.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\ai-based crop recommendation\Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)

# -----------------------------
# Step 2: Split features & target
# -----------------------------
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Step 3: Train models
# -----------------------------
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluate models
# -----------------------------
print("\n=== Decision Tree Performance ===")
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("\n=== Random Forest Performance ===")
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=False, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# -----------------------------
# Step 5: Save the best model
# -----------------------------
joblib.dump(rf, "crop_recommender.pkl")
print("\nModel saved as crop_recommender.pkl")

# -----------------------------
# Step 6: Test with sample input
# -----------------------------
# Example: N=90, P=42, K=43, Temp=20°C, Humidity=82%, pH=6.5, Rainfall=202mm
sample = [[90, 42, 43, 20, 82, 6.5, 202]]
prediction = rf.predict(sample)[0]
print("\nSample Input Prediction:", prediction)
