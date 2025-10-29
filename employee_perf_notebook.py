#!/usr/bin/env python
# coding: utf-8

# # Employee Performance Prediction Model

# ## 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ## 2. Generate Synthetic Dataset
np.random.seed(42)

# Create synthetic employee data
n_samples = 1000

data = {
    'employee_id': range(1, n_samples + 1),
    'age': np.random.randint(22, 60, n_samples),
    'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Operations'], n_samples),
    'years_at_company': np.random.randint(0, 20, n_samples),
    'projects_completed': np.random.randint(0, 50, n_samples),
    'avg_monthly_hours': np.random.randint(120, 280, n_samples),
    'training_hours': np.random.randint(0, 100, n_samples),
    'last_promotion_years': np.random.randint(0, 10, n_samples),
    'salary_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'satisfaction_score': np.random.uniform(0, 1, n_samples),
}

df = pd.DataFrame(data)

# Create performance rating based on features (with some logic)
def calculate_performance(row):
    score = 0
    score += row['projects_completed'] * 0.5
    score += row['training_hours'] * 0.3
    score += (10 - row['last_promotion_years']) * 2
    score += row['satisfaction_score'] * 20
    score -= abs(row['avg_monthly_hours'] - 180) * 0.1
    
    if score < 30:
        return 'Poor'
    elif score < 60:
        return 'Average'
    elif score < 90:
        return 'Good'
    else:
        return 'Excellent'

df['performance_rating'] = df.apply(calculate_performance, axis=1)

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nPerformance Rating Distribution:")
print(df['performance_rating'].value_counts())

# ## 3. Data Preprocessing

# Separate features and target
X = df.drop(['employee_id', 'performance_rating'], axis=1)
y = df['performance_rating']

# Encode categorical variables
le_dept = LabelEncoder()
le_salary = LabelEncoder()
le_performance = LabelEncoder()

X['department'] = le_dept.fit_transform(X['department'])
X['salary_level'] = le_salary.fit_transform(X['salary_level'])
y_encoded = le_performance.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# ## 4. Train the Model

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# ## 5. Evaluate the Model

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=le_performance.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# ## 6. Save the Model and Preprocessors

# Save the trained model
joblib.dump(model, 'employee_performance_model.pkl')

# Save the scalers and encoders
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dept, 'label_encoder_dept.pkl')
joblib.dump(le_salary, 'label_encoder_salary.pkl')
joblib.dump(le_performance, 'label_encoder_performance.pkl')

# Save feature names for reference
joblib.dump(list(X.columns), 'feature_names.pkl')

print("\n" + "="*50)
print("Model and preprocessors saved successfully!")
print("="*50)
print("\nSaved files:")
print("  - employee_performance_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder_dept.pkl")
print("  - label_encoder_salary.pkl")
print("  - label_encoder_performance.pkl")
print("  - feature_names.pkl")

# ## 7. Test Prediction Function

def predict_performance(age, dept, years, projects, hours, training, promotion, salary, satisfaction):
    """Test prediction function"""
    # Prepare input
    input_data = pd.DataFrame({
        'age': [age],
        'department': [dept],
        'years_at_company': [years],
        'projects_completed': [projects],
        'avg_monthly_hours': [hours],
        'training_hours': [training],
        'last_promotion_years': [promotion],
        'salary_level': [salary],
        'satisfaction_score': [satisfaction]
    })
    
    # Transform categorical variables
    input_data['department'] = le_dept.transform(input_data['department'])
    input_data['salary_level'] = le_salary.transform(input_data['salary_level'])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # Decode prediction
    performance = le_performance.inverse_transform(prediction)[0]
    
    return performance, prediction_proba[0]

# Test the prediction function
print("\n" + "="*50)
print("Testing Prediction Function")
print("="*50)

test_employee = {
    'age': 35,
    'dept': 'IT',
    'years': 5,
    'projects': 25,
    'hours': 180,
    'training': 40,
    'promotion': 2,
    'salary': 'High',
    'satisfaction': 0.8
}

performance, probabilities = predict_performance(**test_employee)

print(f"\nTest Employee Profile:")
for key, value in test_employee.items():
    print(f"  {key}: {value}")

print(f"\nPredicted Performance: {performance}")
print(f"\nClass Probabilities:")
for class_name, prob in zip(le_performance.classes_, probabilities):
    print(f"  {class_name}: {prob:.4f}")

print("\n" + "="*50)
print("Ready for Streamlit deployment!")
print("="*50)
