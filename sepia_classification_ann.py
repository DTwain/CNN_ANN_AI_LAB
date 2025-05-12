"""
Sepia Filter Classification using Artificial Neural Networks (ANN)

This script implements a complete ANN for classifying images as normal or sepia-filtered.
It includes dataset creation, model training, and visualization functions.

Author: Your Name
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import seaborn as sns
from datetime import datetime
import time

# Path setup for dataset
DATA_DIR = "sepia_dataset"
NORMAL_DIR = os.path.join(DATA_DIR, "normal")
SEPIA_DIR = os.path.join(DATA_DIR, "sepia")

def apply_sepia_filter(image):
    """Apply sepia filter to an input image"""
    img_array = np.array(image, dtype=np.float32)
    
    # Sepia filter matrix
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Apply filter
    sepia_img = cv2.transform(img_array, sepia_filter)
    
    # Clip values to valid range
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_img)

def create_sepia_dataset(source_folder, num_samples=500, img_size=(128, 128)):
    """
    Create a dataset of normal and sepia-filtered images.
    
    Args:
        source_folder: Path to folder with source images
        num_samples: Maximum number of image pairs to create
        img_size: Size to resize images to
    """
    # Create directories if they don't exist
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(SEPIA_DIR, exist_ok=True)
    
    # List all image files
    image_files = [f for f in os.listdir(source_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process a subset if needed
    if len(image_files) > num_samples:
        image_files = image_files[:num_samples]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        try:
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path).convert('RGB')
            
            # Resize for consistency
            img = img.resize(img_size)
            
            # Save the normal image
            normal_path = os.path.join(NORMAL_DIR, f"normal_{i}.jpg")
            img.save(normal_path)
            
            # Create and save the sepia version
            sepia_img = apply_sepia_filter(img)
            sepia_path = os.path.join(SEPIA_DIR, f"sepia_{i}.jpg")
            sepia_img.save(sepia_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Dataset created with {len(os.listdir(NORMAL_DIR))} normal images and {len(os.listdir(SEPIA_DIR))} sepia images")

def load_and_preprocess_images(directory, label, img_size=(128, 128)):
    """
    Load and preprocess images from a directory
    
    Args:
        directory: Directory containing images
        label: Label for these images (0 for normal, 1 for sepia)
        img_size: Size to resize images to
    
    Returns:
        images: Array of flattened images
        labels: Array of labels
    """
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                # Flatten for ANN input
                img_flat = img_array.reshape(-1)
                
                images.append(img_flat)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(images), np.array(labels)

def get_dataset(source_folder=None, img_size=(128, 128)):
    """
    Get dataset for training and testing
    
    Args:
        source_folder: Path to folder with source images (optional)
        img_size: Size to resize images to
    
    Returns:
        X: Array of flattened images
        y: Array of labels
    """
    # Check if dataset exists
    if not os.path.exists(NORMAL_DIR) or not os.path.exists(SEPIA_DIR) or \
       len(os.listdir(NORMAL_DIR)) == 0 or len(os.listdir(SEPIA_DIR)) == 0:
        if source_folder is None:
            raise ValueError("Dataset doesn't exist and no source folder provided")
        create_sepia_dataset(source_folder, img_size=img_size)
    
    # Load the dataset
    normal_images, normal_labels = load_and_preprocess_images(NORMAL_DIR, 0, img_size)
    sepia_images, sepia_labels = load_and_preprocess_images(SEPIA_DIR, 1, img_size)
    
    # Combine data
    X = np.concatenate((normal_images, sepia_images), axis=0)
    y = np.concatenate((normal_labels, sepia_labels), axis=0)
    
    return X, y

def build_ann_model(input_shape, hyperparams=None):
    """
    Build the ANN model
    
    Args:
        input_shape: Shape of input data (flattened image)
        hyperparams: Dictionary of hyperparameters (optional)
    
    Returns:
        model: Compiled Keras model
    """
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 0.0005,
            'hidden_layers': [512, 256, 128, 64],
            'dropout_rates': [0.3, 0.3, 0.2, 0],
            'activation': 'relu',
            'use_batch_norm': True
        }
    
    model = Sequential()
    
    # Input layer
    model.add(Dense(hyperparams['hidden_layers'][0], 
                   activation=hyperparams['activation'], 
                   input_shape=(input_shape,)))
    if hyperparams['use_batch_norm']:
        model.add(BatchNormalization())
    if hyperparams['dropout_rates'][0] > 0:
        model.add(Dropout(hyperparams['dropout_rates'][0]))
    
    # Hidden layers
    for i in range(1, len(hyperparams['hidden_layers'])):
        model.add(Dense(hyperparams['hidden_layers'][i], 
                       activation=hyperparams['activation']))
        if hyperparams['use_batch_norm']:
            model.add(BatchNormalization())
        if i < len(hyperparams['dropout_rates']) and hyperparams['dropout_rates'][i] > 0:
            model.add(Dropout(hyperparams['dropout_rates'][i]))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ann_model(X, y, hyperparams=None, epochs=50, batch_size=32):
    """
    Train the ANN model
    
    Args:
        X: Input data (flattened images)
        y: Labels
        hyperparams: Dictionary of hyperparameters (optional)
        epochs: Number of epochs to train for
        batch_size: Batch size for training
    
    Returns:
        model: Trained Keras model
        history: Training history
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print dataset information
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")
    print(f"Image shape (flattened): {X_train.shape[1]}")
    
    # Build model
    input_shape = X_train.shape[1]
    model = build_ann_model(input_shape, hyperparams)
    
    # Model summary
    model.summary()
    
    # Callbacks
    log_dir = "logs/ann_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ModelCheckpoint('best_ann_model.h5', 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        mode='max', 
                        verbose=1),
        EarlyStopping(monitor='val_loss', 
                      patience=10, 
                      restore_best_weights=True, 
                      verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Train the model with history to track loss
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('ANN Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('ANN Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ann_model_performance.png')
    plt.show()
    
    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('ann_confusion_matrix.png')
    plt.show()
    
    # Display classification report
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Sepia'])
    print("\nClassification Report:")
    print(report)
    
    return model, history, (X_test, y_test)

def main(source_folder=None):
    """
    Main function
    
    Args:
        source_folder: Path to folder with source images (optional)
    """
    # Get dataset
    X, y = get_dataset(source_folder)
    
    # Set hyperparameters for ANN
    hyperparams = {
        'learning_rate': 0.0005,
        'hidden_layers': [512, 256, 128, 64],
        'dropout_rates': [0.3, 0.3, 0.2, 0],
        'activation': 'relu',
        'use_batch_norm': True
    }
    
    # Train ANN model
    print("\n========== Training ANN Model ==========")
    ann_model, ann_history, _ = train_ann_model(X, y, hyperparams)
    
    print("ANN training completed!")
    return ann_model, ann_history

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sepia Filter Classification using ANN')
    parser.add_argument('--source_folder', type=str, default=None,
                        help='Path to folder with source images')
    args = parser.parse_args()
    
    # Run main function
    main(args.source_folder)