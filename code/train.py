import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import joblib
import numpy as np

# --- 1. Data Loading and Preprocessing ---
print("--- Step 1: Loading and Preprocessing Data ---")
try:
    # Load the dataset. This file must be in the same directory.
    df = pd.read_csv('Epileptic Seizure Recognition.csv')
    df = df.drop('Unnamed', axis=1)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Epileptic Seizure Recognition.csv' not found. Please ensure the file is in the same directory.")
    exit()

# The last column, 'y', is the target variable
X = df.drop('y', axis=1)
y = df['y']

# As per the paper, convert the 5 classes to a binary problem:
# 1 = Seizure, 0 = Non-seizure (Classes 2, 3, 4, 5)
y_binary = y.apply(lambda val: 1 if val == 1 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize the features for optimal model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessed and split into training/testing sets.")

# --- 2. Random Forest Classifier: Train, Save, and Evaluate ---
print("\n--- Step 2: Training, Saving, and Evaluating Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
print("Random Forest model training complete.")

# Save the trained model to a file
joblib.dump(rf_model, 'random_forest_model.joblib')
print("✅ Random Forest model saved to 'random_forest_model.joblib'")

# Evaluate the model on the test set
rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {rf_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# --- 3. 1D-CNN Model: Train, Save, and Evaluate ---
print("\n--- Step 3: Training, Saving, and Evaluating 1D-CNN Model ---")
# Reshape data for the 1D-CNN model (required input format)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Use 'sigmoid' for binary classification
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
print("1D-CNN model training complete.")

# Save the trained model to a file
cnn_model.save('cnn_model.h5')
print("✅ 1D-CNN model saved to 'cnn_model.h5'")

# Evaluate the model on the test set
cnn_predictions_prob = cnn_model.predict(X_test_reshaped, verbose=0)
cnn_predictions = (cnn_predictions_prob > 0.5).astype(int)
cnn_accuracy = accuracy_score(y_test, cnn_predictions)

print("\n1D-CNN Model Evaluation:")
print(f"Accuracy: {cnn_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, cnn_predictions))

print("\n--- Process Complete ---")
