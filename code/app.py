import json
import os
import random
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS

app = Flask(__name__)
# A secret key is required for session management.
# In a production environment, this should be a complex, random value
# stored securely in an environment variable.
app.secret_key = 'a_very_secret_key_that_should_be_in_an_env_file'

# Predefined user credentials for testing purposes.
# In a real application, this would be replaced by a database.
USERS = {
    "user@example.com": {"password": "user_password", "role": "user"},
    "admin@example.com": {"password": "admin_password", "role": "admin"}
}

# Define users who cannot be deleted or have their roles changed.
PROTECTED_USERS = {"admin@example.com", "user@example.com"}

# Load the trained machine learning model from the joblib file.
try:
    rf_model = joblib.load('random_forest_model.joblib')
    print("Random Forest model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    rf_model = None

# --- Before request hook for authentication ---
@app.before_request
def check_authentication():
    """
    Checks for a valid session before rendering dashboard pages.
    If the user is not authenticated, redirects them to the login page.
    """
    if 'user' not in session and request.endpoint in ['user_dashboard', 'admin_dashboard']:
        return redirect(url_for('login_page'))
    if session.get('user', {}).get('role') != 'admin' and request.endpoint == 'admin_dashboard':
        return redirect(url_for('user_dashboard'))

# --- Routes for rendering HTML pages ---
@app.route('/')
def home():
    """Renders the login page."""
    return render_template('login.html')

@app.route('/login.html')
def login_page():
    """Renders the login page."""
    return render_template('login.html')

@app.route('/register.html')
def register_page():
    """Renders the registration page."""
    return render_template('register.html')

@app.route('/user_dashboard.html')
def user_dashboard():
    """Renders the user dashboard."""
    return render_template('user_dashboard.html')

@app.route('/admin_dashboard.html')
def admin_dashboard():
    """Renders the admin dashboard."""
    return render_template('admin_dashboard.html')

# --- Routes for backend API functionality ---
@app.route('/login', methods=['POST'])
def login():
    """
    Handles user login.
    If the user exists and the password is correct, sets a session and redirects.
    """
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = USERS.get(email)
    if user and user['password'] == password:
        session['user'] = {'email': email, 'role': user['role']}
        redirect_url = url_for('admin_dashboard') if user['role'] == 'admin' else url_for('user_dashboard')
        return jsonify({'success': True, 'redirect_url': redirect_url})
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/register', methods=['POST'])
def register():
    """
    Handles new user registration.
    """
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email in USERS:
        return jsonify({'success': False, 'error': 'User already exists.'}), 409

    # Add new user with 'user' role by default
    USERS[email] = {"password": password, "role": "user"}
    session['user'] = {'email': email, 'role': 'user'}
    return jsonify({'success': True, 'redirect_url': url_for('user_dashboard')})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives EEG data and uses the loaded model to predict the class.
    """
    if rf_model is None:
        return jsonify({'error': 'Model not loaded. Please ensure random_forest_model.joblib is in the directory.'}), 500

    try:
        data = request.get_json()
        eeg_data = data.get('eeg_data')
        if not eeg_data or not isinstance(eeg_data, list) or len(eeg_data) != 178:
            return jsonify({'error': 'Invalid data. Expected 178 numerical data points.'}), 400

        # Convert the list to a numpy array and reshape it for the model.
        eeg_features = np.array(eeg_data).reshape(1, -1)
        
        # Make a prediction using the loaded model.
        prediction = rf_model.predict(eeg_features)[0]

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Routes for Admin User Management ---
@app.route('/get_users')
def get_users():
    """Returns a list of all users and their roles."""
    user_list = [{"email": email, "role": data["role"]} for email, data in USERS.items()]
    return jsonify(user_list)

@app.route('/update_user', methods=['POST'])
def update_user():
    """Updates a user's role, but prevents changing the role of protected users."""
    data = request.get_json()
    email = data.get('email')
    role = data.get('role')

    if email in PROTECTED_USERS:
        return jsonify({"success": False, "error": "Cannot change the role of a protected user."}), 403

    if email in USERS:
        USERS[email]['role'] = role
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "User not found"}), 404

@app.route('/delete_user', methods=['POST'])
def delete_user():
    """Deletes a user from the system, but prevents the deletion of protected users."""
    data = request.get_json()
    email = data.get('email')

    if email in PROTECTED_USERS:
        return jsonify({"success": False, "error": "Cannot delete a protected user."}), 403

    if email in USERS:
        del USERS[email]
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "User not found"}), 404

@app.route('/logout', methods=['POST'])
def logout():
    """Logs out the user by clearing the session."""
    session.pop('user', None)
    return jsonify({'success': True, 'redirect_url': url_for('login_page')})

if __name__ == '__main__':
    app.run(debug=True)
