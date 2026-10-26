import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset from Excel
excel_file = "Intelligent_Loan_Risk_Assessment_Dataset.xlsx"
df = pd.read_excel(excel_file, sheet_name="Loan_Risk_Dataset")

# Convert 3 risk levels into 2 binary categories (Low Risk vs High Risk)
# Medium Risk ni High Risk కి కిందకి మారుస్తున్నాము (కఠినమైన రూల్స్ కోసం) లేదా Risk_Score బేస్ చేసుకొని చేయవచ్చు
df['Binary_Risk'] = df['Risk_Level'].apply(lambda x: 'Low Risk' if x == 'Low Risk' else 'High Risk')

X = df.drop(columns=["Loan_ID", "Risk_Score", "Risk_Level", "Binary_Risk"], errors="ignore")
y = df["Binary_Risk"]

# Encode categorical feature: Employment_Type
label_encoders = {}
if "Employment_Type" in X.columns:
    le_emp = LabelEncoder()
    X["Employment_Type"] = le_emp.fit_transform(X["Employment_Type"].astype(str))
    label_encoders["Employment_Type"] = le_emp

# Encode target classes (Low Risk, High Risk)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("Target Class Mapping:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MODEL ACCURACY: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Save Artifacts
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "final_loan_risk_model.pkl"))
joblib.dump(list(X.columns), os.path.join(models_dir, "feature_columns.pkl"))
joblib.dump(target_encoder, os.path.join(models_dir, "target_encoder.pkl"))
joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))

importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
joblib.dump(importance_df, os.path.join(models_dir, "feature_importance.pkl"))

print("✅ 2-Class Model trained and saved successfully!")