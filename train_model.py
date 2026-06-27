# train_model.py
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Create necessary directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("⚡ Starting generation of 10,000+ premium banking records...")

np.random.seed(42)
n_records = 10500

# Generating synthetic banking features
genders = np.random.choice(['Male', 'Female'], size=n_records, p=[0.6, 0.4])
married = np.random.choice(['Yes', 'No'], size=n_records, p=[0.65, 0.35])
dependents = np.random.choice(['0', '1', '2', '3+'], size=n_records, p=[0.5, 0.2, 0.15, 0.15])
education = np.random.choice(['Graduate', 'Not Graduate'], size=n_records, p=[0.80, 0.20])
self_employed = np.random.choice(['Yes', 'No'], size=n_records, p=[0.15, 0.85])

# Financial data simulation using negative binomial distribution
applicant_income = np.random.negative_binomial(n=10, p=0.002, size=n_records) + 2000
applicant_income = np.clip(applicant_income, 1500, 50000)

coapplicant_income = np.where(np.random.rand(n_records) > 0.4, 
                               np.random.negative_binomial(n=5, p=0.002, size=n_records) + 1000, 0)
coapplicant_income = np.clip(coapplicant_income, 0, 25000)

loan_amount = (applicant_income + coapplicant_income) * np.random.uniform(1.5, 4.0, size=n_records) / 10
loan_amount = np.clip(loan_amount, 30, 700).astype(int)

loan_term = np.random.choice([120, 180, 240, 360], size=n_records, p=[0.05, 0.05, 0.1, 0.8])
credit_history = np.random.choice([1.0, 0.0], size=n_records, p=[0.85, 0.15])
property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n_records, p=[0.3, 0.4, 0.3])
applicant_age = np.random.randint(21, 65, size=n_records)

# Creating the DataFrame
df = pd.DataFrame({
    'Gender': genders,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'Applicant_Age': applicant_age
})

# Advanced banking logic for determining Loan_Status (Y/N)
total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
monthly_income = total_income / 12
estimated_emi = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
dti_ratio = estimated_emi / monthly_income

# Credit scoring algorithm
scores = (df['Credit_History'] * 65) + (df['Education'] == 'Graduate') * 10 + (df['Applicant_Age'].between(25, 55)) * 10 - (dti_ratio * 30)
scores += np.random.normal(0, 8, n_records)

df['Loan_Status'] = np.where(scores > 30, 'Y', 'N')

# Save dataset to CSV
df.to_csv('dataset/loan_data.csv', index=False)
print("💾 Dataset successfully saved to 'dataset/loan_data.csv'.")

# ML Preprocessing and Label Encoding
df_ml = df.copy()
encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

df_ml['Loan_Status'] = df_ml['Loan_Status'].map({'Y': 1, 'N': 0})

X = df_ml.drop('Loan_Status', axis=1)
y = df_ml['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Classifier
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model training complete! Accuracy: {accuracy * 100:.2f}%")

# Save model pipeline using pickle
model_data = {
    'model': model,
    'encoders': encoders,
    'features': list(X.columns)
}

with open('models/loan_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("🎯 AI Pipeline successfully exported to 'models/loan_model.pkl'.")