import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

# Load data
df = pd.read_csv('dataset/for_trained_weatherAUS.csv')

# df.drop(columns=['RainToday','Rainfall'],inplace=True)
# Inisialisasi Target
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=27, 
    stratify=y
)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    class_weight='balanced',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracu Model: {accuracy:.2f}%")
print(f"F1 Score {f1_score:.2f}")
print(f"Precision {precision}")
print(f"Recall Model: {recall:.2f}")

model_packet = {
    'model_obj': model,
    'accuracy': accuracy,
    'f1_score': f1_score
}

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


# Save model and feature importance
print("\nSaving model...")
joblib.dump(model, 'model/rain_prediction_model.pkl')
joblib.dump(model_packet, 'model/rain_prediction_model_accuracy.pkl')
feature_importance.to_csv('dataset/feature_importance.csv', index=False)


print("Model saved successfully!")