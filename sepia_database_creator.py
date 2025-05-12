"""
Sepia Dataset Creator and Model Comparison

This script provides utilities for creating a sepia filter dataset
and comparing the performance of ANN and CNN models.

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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import seaborn as sns
from datetime import datetime
import time
import glob
import random
from tabulate import tabulate

DATA_DIR = "sepia_dataset"
NORMAL_DIR = os.path.join(DATA_DIR, "normal")
SEPIA_DIR = os.path.join(DATA_DIR, "sepia")

def apply_sepia_filter(image):
    img_array = np.array(image, dtype=np.float32)
    
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    
    sepia_img = cv2.transform(img_array, sepia_filter)
    
    
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_img)

def create_sepia_dataset(source_folder=None, num_samples=500, img_size=(128, 128)):
    """
    Create a dataset of normal and sepia-filtered images.
    
    Args:
        source_folder: Path to folder with source images. If None, create sample images.
        num_samples: Maximum number of image pairs to create
        img_size: Size to resize images to
    """
    # Create directories if they don't exist
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(SEPIA_DIR, exist_ok=True)
    
    # If no source folder is provided, create random sample images
    if source_folder is None or not os.path.exists(source_folder):
        print("No source folder provided. Creating random sample images...")
        os.makedirs("sample_images", exist_ok=True)
        source_folder = "sample_images"
        
        # Create random sample images
        for i in range(100):
            # Random colored image
            img_array = np.random.randint(0, 256, (img_size[0], img_size[1], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(source_folder, f"sample_{i}.jpg"))
    
    # List all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(source_folder, f"*{ext}")))
    
    # Process a subset if needed
    if len(image_files) > num_samples:
        random.shuffle(image_files)
        image_files = image_files[:num_samples]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        try:
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
            print(f"Error processing {img_path}: {e}")
    
    print(f"Dataset created with {len(os.listdir(NORMAL_DIR))} normal images and {len(os.listdir(SEPIA_DIR))} sepia images")
    
    # Show some sample pairs
    show_sample_pairs(3)

def show_sample_pairs(num_samples=3):
    """Display sample pairs of normal and sepia images"""
    normal_files = os.listdir(NORMAL_DIR)
    if len(normal_files) == 0:
        print("No images found in the dataset.")
        return
    
    if len(normal_files) < num_samples:
        num_samples = len(normal_files)
    
    sample_indices = random.sample(range(len(normal_files)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    
    if num_samples == 1:
        axes = [axes]  # Make it easier to index when there's only one sample
    
    for i, idx in enumerate(sample_indices):
        normal_file = normal_files[idx]
        base_name = os.path.basename(normal_file)
        sepia_file = base_name.replace("normal", "sepia")
        
        # Load images
        normal_img = Image.open(os.path.join(NORMAL_DIR, normal_file))
        sepia_img = Image.open(os.path.join(SEPIA_DIR, sepia_file))
        
        # Display
        axes[i][0].imshow(normal_img)
        axes[i][0].set_title("Normal")
        axes[i][0].axis("off")
        
        axes[i][1].imshow(sepia_img)
        axes[i][1].set_title("Sepia")
        axes[i][1].axis("off")
    
    plt.tight_layout()
    plt.savefig('sample_image_pairs.png')
    plt.show()

def load_dataset(flatten_for_ann=False, img_size=(128, 128)):
    """
    Load the sepia dataset
    
    Args:
        flatten_for_ann: If True, flatten images for ANN. If False, keep spatial structure for CNN.
        img_size: Size of images to resize to
    
    Returns:
        X, y: Images and labels
    """
    # Check if dataset exists
    if not os.path.exists(NORMAL_DIR) or not os.path.exists(SEPIA_DIR) or \
       len(os.listdir(NORMAL_DIR)) == 0 or len(os.listdir(SEPIA_DIR)) == 0:
        print("Dataset doesn't exist. Creating sample dataset...")
        create_sepia_dataset(None, 100, img_size)
    
    # Load normal images
    normal_images = []
    normal_labels = []
    
    for filename in os.listdir(NORMAL_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(NORMAL_DIR, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                if flatten_for_ann:
                    img_array = img_array.reshape(-1)  # Flatten
                
                normal_images.append(img_array)
                normal_labels.append(0)  # 0 for normal
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Load sepia images
    sepia_images = []
    sepia_labels = []
    
    for filename in os.listdir(SEPIA_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(SEPIA_DIR, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                if flatten_for_ann:
                    img_array = img_array.reshape(-1)  # Flatten
                
                sepia_images.append(img_array)
                sepia_labels.append(1)  # 1 for sepia
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Combine data
    X = np.array(normal_images + sepia_images)
    y = np.array(normal_labels + sepia_labels)
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset loaded: {len(X)} images, {X.shape}")
    
    return X, y

def train_and_compare_models(source_folder=None, img_size=(128, 128), epochs_ann=50, epochs_cnn=100):
    """
    Train both ANN and CNN models on the sepia dataset and compare performance.
    
    Args:
        source_folder: Path to folder with source images
        img_size: Size to resize images to
        epochs_ann: Number of epochs for ANN training
        epochs_cnn: Number of epochs for CNN training
    """
    # Create dataset if needed
    if source_folder is not None:
        create_sepia_dataset(source_folder, num_samples=1000, img_size=img_size)
    
    # Load data for ANN (flattened)
    print("\n===== Loading flattened data for ANN =====")
    X_ann, y_ann = load_dataset(flatten_for_ann=True, img_size=img_size)
    
    # Load data for CNN (spatial structure preserved)
    print("\n===== Loading structured data for CNN =====")
    X_cnn, y_cnn = load_dataset(flatten_for_ann=False, img_size=img_size)
    
    # Split data
    X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X_ann, y_ann, test_size=0.2, random_state=42)
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)
    
    # Build ANN model
    def build_ann_model(input_shape):
        model = Sequential([
            # Input layer
            Dense(512, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Build CNN model
    def build_cnn_model(input_shape):
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Train ANN model
    print("\n===== Training ANN Model =====")
    input_shape_ann = X_train_ann.shape[1]
    ann_model = build_ann_model(input_shape_ann)
    
    ann_model.summary()
    
    # Callbacks for ANN
    ann_callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_ann_model.h5', 
                                         monitor='val_accuracy', 
                                         save_best_only=True, 
                                         mode='max', 
                                         verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                       patience=10, 
                                       restore_best_weights=True, 
                                       verbose=1)
    ]
    
    # Train ANN
    ann_start_time = time.time()
    ann_history = ann_model.fit(
        X_train_ann, y_train_ann,
        validation_data=(X_test_ann, y_test_ann),
        epochs=epochs_ann,
        batch_size=32,
        callbacks=ann_callbacks,
        verbose=1
    )
    ann_training_time = time.time() - ann_start_time
    
    # Train CNN model
    print("\n===== Training CNN Model =====")
    input_shape_cnn = X_train_cnn.shape[1:]
    cnn_model = build_cnn_model(input_shape_cnn)
    
    cnn_model.summary()
    
    # Data augmentation for CNN
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(X_train_cnn)
    
    # Callbacks for CNN
    cnn_callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_cnn_model.h5', 
                                         monitor='val_accuracy', 
                                         save_best_only=True, 
                                         mode='max', 
                                         verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                       patience=15, 
                                       restore_best_weights=True, 
                                       verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                           factor=0.5, 
                                           patience=5, 
                                           min_lr=1e-6, 
                                           verbose=1)
    ]
    
    # Train CNN
    cnn_start_time = time.time()
    cnn_history = cnn_model.fit(
        datagen.flow(X_train_cnn, y_train_cnn, batch_size=32),
        validation_data=(X_test_cnn, y_test_cnn),
        epochs=epochs_cnn,
        steps_per_epoch=len(X_train_cnn) // 32,
        callbacks=cnn_callbacks,
        verbose=1
    )
    cnn_training_time = time.time() - cnn_start_time
    
    # Evaluate models
    print("\n===== Evaluating Models =====")
    
    # Load best models
    best_ann_model = load_model('best_ann_model.h5')
    best_cnn_model = load_model('best_cnn_model.h5')
    
    # Evaluate ANN
    ann_loss, ann_accuracy = best_ann_model.evaluate(X_test_ann, y_test_ann)
    print(f"ANN Test Accuracy: {ann_accuracy:.4f}")
    print(f"ANN Test Loss: {ann_loss:.4f}")
    print(f"ANN Training Time: {ann_training_time:.2f} seconds")
    
    # Evaluate CNN
    cnn_loss, cnn_accuracy = best_cnn_model.evaluate(X_test_cnn, y_test_cnn)
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    print(f"CNN Test Loss: {cnn_loss:.4f}")
    print(f"CNN Training Time: {cnn_training_time:.2f} seconds")
    
    # Compare loss curves
    plt.figure(figsize=(15, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(ann_history.history['loss'], label='ANN Training Loss')
    plt.plot(cnn_history.history['loss'], label='CNN Training Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(ann_history.history['val_loss'], label='ANN Validation Loss')
    plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison_loss.png')
    plt.show()
    
    # Compare accuracy curves
    plt.figure(figsize=(15, 6))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(ann_history.history['accuracy'], label='ANN Training Accuracy')
    plt.plot(cnn_history.history['accuracy'], label='CNN Training Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(ann_history.history['val_accuracy'], label='ANN Validation Accuracy')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison_accuracy.png')
    plt.show()
    
    # Make predictions with both models
    ann_preds = (best_ann_model.predict(X_test_ann) > 0.5).astype(int).flatten()
    cnn_preds = (best_cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()
    
    # Create confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ANN confusion matrix
    ann_cm = confusion_matrix(y_test_ann, ann_preds)
    sns.heatmap(ann_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'],
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('ANN Confusion Matrix')
    
    # CNN confusion matrix
    cnn_cm = confusion_matrix(y_test_cnn, cnn_preds)
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'],
                ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('CNN Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('model_comparison_confusion_matrices.png')
    plt.show()
    
    # Display classification reports
    print("\nANN Classification Report:")
    print(classification_report(y_test_ann, ann_preds, target_names=['Normal', 'Sepia']))
    
    print("\nCNN Classification Report:")
    print(classification_report(y_test_cnn, cnn_preds, target_names=['Normal', 'Sepia']))
    
    # Summary table
    summary = {
        'Model': ['ANN', 'CNN'],
        'Accuracy': [ann_accuracy, cnn_accuracy],
        'Loss': [ann_loss, cnn_loss],
        'Training Time (s)': [ann_training_time, cnn_training_time],
        'Parameters': [ann_model.count_params(), cnn_model.count_params()],
        'Epochs': [len(ann_history.history['loss']), len(cnn_history.history['loss'])],
    }
    
    # Print summary as table
    print("\n===== Model Comparison Summary =====")
    headers = list(summary.keys())
    rows = list(zip(*summary.values()))
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Save table as image
    fig, ax = plt.figure(figsize=(10, 2)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig('model_comparison_summary.png', bbox_inches='tight', dpi=300)
    
    # Return best models and their histories
    return (best_ann_model, ann_history), (best_cnn_model, cnn_history)

def main(source_folder=None):
    """
    Main function to run the sepia filter classification
    
    Args:
        source_folder: Path to folder with source images
    """
    # Print welcome message
    print("="*50)
    print("SEPIA FILTER CLASSIFICATION USING ANN AND CNN")
    print("="*50)
    
    # Step 1: Create dataset if source folder is provided
    if source_folder is not None:
        print("\n===== Creating Sepia Dataset =====")
        create_sepia_dataset(source_folder)
    
    # Step 2: Train and compare models
    print("\n===== Training and Comparing Models =====")
    (ann_model, ann_history), (cnn_model, cnn_history) = train_and_compare_models()
    
    # Step 3: Visualize example predictions
    print("\n===== Visualizing Example Predictions =====")
    
    # Load data
    X, y = load_dataset(flatten_for_ann=False)  # Load in CNN format
    X_test_cnn = X[:20]  # Use first 20 images for visualization
    y_test_cnn = y[:20]
    
    # Flatten for ANN
    X_test_ann = np.array([img.reshape(-1) for img in X_test_cnn])
    
    # Make predictions
    ann_preds = ann_model.predict(X_test_ann)
    cnn_preds = cnn_model.predict(X_test_cnn)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, len(X_test_cnn))):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test_cnn[i])
        plt.title(f"True: {'Sepia' if y_test_cnn[i] == 1 else 'Normal'}\n"
                  f"ANN: {'Sepia' if ann_preds[i][0] > 0.5 else 'Normal'} ({ann_preds[i][0]:.2f})\n"
                  f"CNN: {'Sepia' if cnn_preds[i][0] > 0.5 else 'Normal'} ({cnn_preds[i][0]:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.show()
    
    print("\n===== All Done! =====")
    print(f"ANN model saved to: best_ann_model.h5")
    print(f"CNN model saved to: best_cnn_model.h5")
    print(f"Performance visualizations saved to current directory")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sepia Filter Classification')
    parser.add_argument('--source_folder', type=str, default=None,
                        help='Path to folder with source images')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Maximum number of image pairs to create')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size to resize images to')
    args = parser.parse_args()
    
    # Run main function
    main(args.source_folder)