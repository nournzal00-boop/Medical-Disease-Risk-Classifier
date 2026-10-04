import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Synthetic Health Metrics Dataset
data = {
    'age': [45, 22, 60, 35, 50, 28, 65, 30],
    'bmi': [28.5, 21.0, 33.2, 24.1, 31.0, 22.5, 35.1, 23.8],
    'glucose_level': [140, 85, 180, 95, 150, 88, 190, 90],
    'high_risk': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: High Risk, 0: Low Risk
}

df = pd.DataFrame(data)

# 2. Separate Features and Target
X = df[['age', 'bmi', 'glucose_level']]
y = df['high_risk']

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X_scaled, y)

# 5. Test with Patient Data
sample_patient = [[52, 29.4, 160]]
sample_patient_scaled = scaler.transform(sample_patient)
risk_pred = classifier.predict(sample_patient_scaled)

print(f"Risk Assessment Result: {'High Risk' if risk_pred[0] == 1 else 'Low Risk'}")
