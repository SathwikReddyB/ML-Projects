import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def load_image_datasets(dataset_path):
    """
    Loads and prepares the image datasets from the specified directory.
    This function automatically infers labels from subfolder names.

    Args:
        dataset_path (str): The path to the root dataset directory
                            (e.g., 'path/to/stroke-detection-project').

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    print(f"Loading datasets from {dataset_path}...")
    
    # Define the image size and batch size.
    img_height = 227
    img_width = 227
    batch_size = 32

    # Load the training dataset.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, 'train'),
        validation_split=0.0,  # No validation split here as we have a dedicated folder.
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale'
    )
    
    # Load the validation dataset.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, 'validation'),
        validation_split=0.0,
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale'
    )

    # Load the test dataset.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, 'test'),
        validation_split=0.0,
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale'
    )

    # Convert the datasets to a format suitable for the model.
    # We will rescale the pixel values to the [0, 1] range.
    normalization_layer = layers.Rescaling(1./255)

    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, 2)))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, 2)))
    normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, 2)))

    return normalized_train_ds, normalized_val_ds, normalized_test_ds


def create_cnn_bilstm_model():
    """
    Defines and compiles a hybrid CNN-BiLSTM model with improved architecture.
    This architecture first uses a CNN for image feature extraction,
    then a BiLSTM layer to analyze the feature sequence, as implied
    by the paper's GA_BiLSTM model.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential([
        # Layer 1: Convolutional layer with increased filters to capture more features.
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(227, 227, 1)),
        # Layer 2: Max Pooling to downsample the feature maps.
        layers.MaxPooling2D((2, 2)),
        
        # Layer 3: Another convolutional layer.
        layers.Conv2D(128, (3, 3), activation='relu'),
        # Layer 4: Another max pooling layer.
        layers.MaxPooling2D((2, 2)),

        # Add a third set of Conv and MaxPooling layers to extract deeper features.
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Reshape the output from the CNN into a sequence.
        # The output of the last MaxPooling2D layer is 4D.
        # We need to reshape it into a 3D tensor (None, sequence_length, features).
        layers.Reshape((-1, 256)), # -1 automatically calculates the sequence length

        # Layer 5: The Bidirectional LSTM layer.
        # Increased LSTM units to learn more complex patterns.
        layers.Bidirectional(layers.LSTM(128)),

        # Add a Dropout layer to reduce overfitting.
        layers.Dropout(0.5),

        # Layer 6: A dense layer for final classification with more units.
        layers.Dense(256, activation='relu'),
        
        # Output layer with 2 units for binary classification (Normal/Stroke).
        layers.Dense(2, activation='softmax')
    ])

    # Compile the model.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print a summary of the model architecture.
    model.summary()
    return model

def train_model(train_ds, val_ds):
    """
    Main function to train the model.
    
    Args:
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
    """
    # 1. Create the CNN-BiLSTM model.
    model = create_cnn_bilstm_model()

    # 2. Train the model.
    print("\nStarting model training...")
    # The fit method trains the model.
    history = model.fit(
        train_ds,
        epochs=20,  # Increased epochs to allow more training
        validation_data=val_ds
    )
    
    # 3. Print the final results.
    print("\nTraining complete!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # 4. Save the trained model to a file.
    model_save_path = "stroke_detection_model.keras"
    model.save(model_save_path)
    print(f"\nModel saved successfully at: {model_save_path}")
    return model_save_path

def test_model(model_path, test_ds):
    """
    Loads a saved model and evaluates its performance on a test dataset.
    
    Args:
        model_path (str): The path to the saved model file.
        test_ds (tf.data.Dataset): The test dataset.
    """
    try:
        # Load the saved model.
        print("\nLoading the trained model...")
        loaded_model = keras.models.load_model(model_path)
        print("Model loaded successfully.")

        # Evaluate the model's performance on the test data.
        print("Evaluating model on test data...")
        loss, accuracy = loaded_model.evaluate(test_ds, verbose=2)

        # Print the final test results.
        print("\nFinal Test Results:")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main execution block
if __name__ == "__main__":
    # Specify the path to your dataset folder.
    # The path has been updated based on your provided location.
    DATASET_PATH = "C:\\Users\\hp\\Desktop\\test projects\\Innovations_in_Stroke_Identification_A_Machine_Learning-Based_Diagnostic_Model_Using_Neuroimages\\Brain_Stroke_CT-SCAN_image"
    
    # Load the datasets
    train_ds, val_ds, test_ds = load_image_datasets(DATASET_PATH)

    # Train the model
    trained_model_path = train_model(train_ds, val_ds)

    # Test the model
    test_model(trained_model_path, test_ds)
