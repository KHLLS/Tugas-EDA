import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# Load data
df = pd.read_csv('dataset/for_trained_weatherAUS.csv')

# Inisialisasi Target
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save model and feature importance
print("\nSaving model...")
joblib.dump(model, 'model/rain_prediction_model.pkl')
joblib.dump(scaler, 'model/scaler_weather.pkl')
feature_importance.to_csv('dataset/feature_importance.csv', index=False)


print("Model saved successfully!")