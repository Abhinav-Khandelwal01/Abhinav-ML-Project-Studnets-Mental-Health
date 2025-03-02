import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate dataset
np.random.seed(42)
n_samples = 5000  # Sufficient for training a robust model

data = {
    'study_hours': np.random.randint(2, 16, n_samples),
    'sleep_hours': np.random.randint(3, 9, n_samples),
    'social_activity': np.random.randint(0, 5, n_samples),
    'parental_pressure': np.random.randint(1, 10, n_samples),
    'exam_stress_level': np.random.randint(1, 10, n_samples),
    'physical_activity': np.random.randint(0, 7, n_samples),
    'diet_quality': np.random.randint(1, 5, n_samples),
    'screen_time': np.random.randint(1, 10, n_samples),
    'academic_performance': np.random.randint(40, 100, n_samples),
    'peer_support': np.random.randint(1, 5, n_samples),
    'family_income': np.random.randint(1, 10, n_samples),
    'hobbies_leisure': np.random.randint(0, 5, n_samples),
    'mental_health_history': np.random.choice([0, 1], n_samples),
    'suicidal_risk': np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
}

df = pd.DataFrame(data)

# Split dataset
X = df.drop(columns=['suicidal_risk'])
y = df['suicidal_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Save model
joblib.dump(model, "final_mental_health_model.pkl")
print("Model saved as final_mental_health_model.pkl")
