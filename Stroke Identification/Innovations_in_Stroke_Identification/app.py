import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for the prediction endpoint
CORS(app)
# Secret key for session management
app.secret_key = 'your_super_secret_key'

# --- Configuration and Model Loading ---
# Define the path to the Keras model file
MODEL_PATH = 'stroke_detection_model.keras'

# Global variables
model = None
IMAGE_SIZE = None

def load_model_from_keras():
    """
    Loads the trained Keras model from the specified file path.
    This function is called once when the application starts.
    It also dynamically gets the required image size from the model.
    """
    global model, IMAGE_SIZE
    try:
        print("Loading the trained Keras model...")
        # Load the model
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        
        # Dynamically get the required image size from the model's input shape
        # The input shape is typically (None, height, width, channels)
        input_shape = model.input_shape
        if len(input_shape) >= 3:
            # The height and width are usually at indices 1 and 2
            IMAGE_SIZE = (input_shape[2], input_shape[1])
            print(f"Model requires image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.")
        else:
            print("Could not determine model input size. Using default (224, 224).")
            IMAGE_SIZE = (224, 224)

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        IMAGE_SIZE = None

# Load the model when the application starts
load_model_from_keras()

# --- User Authentication (In-Memory) ---
# Hardcoded users, including the new 'patientuser'. New registrations will be added to this dictionary.
users = {
    "adminuser": {"password": "adminpassword", "role": "admin"},
    "patientuser": {"password": "patientpassword", "role": "patient"}
}

# Define a set of protected users that cannot be deleted
PROTECTED_USERS = {"adminuser", "patientuser"}


# --- API Endpoints ---
@app.route('/')
def index():
    """
    Redirects the user to the login page.
    """
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles user login.
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = users[username]['role']
            if session['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handles user registration.
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('register.html', error="Username already exists.")
        
        # New users are automatically given the 'patient' role.
        users[username] = {"password": password, "role": "patient"}
        return render_template('login.html', message="Registration successful! Please log in.")
    return render_template('register.html')

@app.route('/logout')
def logout():
    """
    Logs out the current user.
    """
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/admin_dashboard')
def admin_dashboard():
    """
    Displays the admin dashboard. Requires admin role.
    It now passes the list of all users and protected users to the template.
    """
    if 'logged_in' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html', username=session['username'], users=users, protected_users=PROTECTED_USERS)

@app.route('/patient_dashboard')
def patient_dashboard():
    """
    Displays the patient dashboard. Requires patient role.
    """
    if 'logged_in' not in session or session.get('role') != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_dashboard.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    This endpoint handles POST requests to predict stroke from an image.
    It expects a single image file named 'file' in the form data.
    """
    if model is None or IMAGE_SIZE is None:
        return jsonify({'error': 'Model not loaded or input size not determined.'}), 500
    
    # Simple authentication check for the predict endpoint
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized access.'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Open the image file and convert it to grayscale ('L' mode)
            image = Image.open(file.stream).convert('L')
            # Resize the image dynamically to the size the model expects
            image = image.resize(IMAGE_SIZE)
            # Convert the image to a numpy array
            image_array = np.array(image)
            # Add a batch dimension and a channel dimension (for grayscale)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=-1)
            
            # Normalize the image pixel values to be between 0 and 1
            image_array = image_array / 255.0

            # Make the prediction using the loaded model
            predictions = model.predict(image_array)
            # The model's output is a probability. Get the confidence score.
            confidence = predictions[0][0]
            
            # Determine the predicted class based on a threshold (e.g., 0.5)
            if confidence > 0.5:
                # If the confidence is high, it predicts stroke
                result = 'No Stroke'
            else:
                # Otherwise, it predicts no stroke
                result = 'Stroke'
            
            # Return the prediction and confidence in a JSON response
            return jsonify({
                'prediction': result,
                'confidence': float(confidence)
            })

        except Exception as e:
            # Handle any processing errors gracefully
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500

# --- User Management API Endpoints ---
@app.route('/api/users', methods=['GET'])
def get_users():
    """
    API endpoint to get a list of all users.
    """
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(users)

@app.route('/api/users', methods=['POST'])
def create_user():
    """
    API endpoint to create a new user.
    """
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'patient')
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if username in users:
        return jsonify({'error': 'Username already exists'}), 409
    
    users[username] = {"password": password, "role": role}
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/users/<username>', methods=['PUT'])
def update_user(username):
    """
    API endpoint to update an existing user's details.
    """
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    if username not in users:
        return jsonify({'error': 'User not found'}), 404
        
    data = request.json
    new_password = data.get('password')
    new_role = data.get('role')
    
    if new_password:
        users[username]['password'] = new_password
    if new_role:
        users[username]['role'] = new_role
    
    return jsonify({'message': 'User updated successfully'}), 200

@app.route('/api/users/<username>', methods=['DELETE'])
def delete_user(username):
    """
    API endpoint to delete a user.
    """
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    if username in PROTECTED_USERS:
        return jsonify({'error': 'Cannot delete hardcoded admin or patient users.'}), 403
    if username not in users:
        return jsonify({'error': 'User not found'}), 404
        
    del users[username]
    return jsonify({'message': 'User deleted successfully'}), 200

if __name__ == '__main__':
    # When running the app, it will look for an HTML file in a "templates" folder.
    # So, make sure you save your HTML files in a folder named 'templates'.
    app.run(debug=True, host='0.0.0.0', port=5000)
