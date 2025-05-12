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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from datetime import datetime
import time
import glob
import random
import argparse
from tabulate import tabulate

# Path setup for dataset
DATA_DIR = "sepia_dataset"
NORMAL_DIR = os.path.join(DATA_DIR, "normal")
SEPIA_DIR = os.path.join(DATA_DIR, "sepia")

# ==================================
# DATASET CREATION FUNCTIONS
# ==================================

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

# ==================================
# MAIN FUNCTIONALITY
# ==================================

def main(args):
    """
    Main function to run the sepia filter classification
    
    Args:
        args: Command line arguments
    """
    # Print welcome message
    print("="*60)
    print("SEPIA FILTER CLASSIFICATION USING ANN AND CNN")
    print("="*60)
    
    # Step 1: Create or load dataset
    if args.source_folder:
        print(f"\n===== Creating dataset from {args.source_folder} =====")
        create_sepia_dataset(args.source_folder, args.num_samples, (args.img_size, args.img_size))
    
    # Step 2: Load dataset for ANN (flattened)
    print("\n===== Loading data for ANN =====")
    X_ann, y_ann = load_dataset(flatten_for_ann=True, img_size=(args.img_size, args.img_size))
    
    # Step 3: Train ANN model
    print("\n===== Training ANN Model =====")
    ann_results = train_ann_model(X_ann, y_ann, epochs=args.ann_epochs, batch_size=args.batch_size)
    
    # Step 4: Load dataset for CNN (spatial structure preserved)
    print("\n===== Loading data for CNN =====")
    X_cnn, y_cnn = load_dataset(flatten_for_ann=False, img_size=(args.img_size, args.img_size))
    
    # Step 5: Train CNN model
    print("\n===== Training CNN Model =====")
    cnn_results = train_cnn_model(X_cnn, y_cnn, epochs=args.cnn_epochs, batch_size=args.batch_size, 
                                 use_augmentation=args.use_augmentation)
    
    # Step 6: Compare models
    print("\n===== Comparing Models =====")
    best_model = compare_models(ann_results, cnn_results)
    
    print("\n===== All Done! =====")
    print(f"ANN model saved to: best_ann_model.h5")
    print(f"CNN model saved to: best_cnn_model.h5")
    print(f"Performance visualizations saved to current directory")
    
    return best_model

# ==================================
# COMMAND LINE INTERFACE
# ==================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sepia Filter Classification using ANN and CNN')
    
    parser.add_argument('--source_folder', type=str, default=None,
                        help='Path to folder with source images')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Maximum number of image pairs to create')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size to resize images to')
    parser.add_argument('--ann_epochs', type=int, default=50,
                        help='Maximum number of epochs for ANN training')
    parser.add_argument('--cnn_epochs', type=int, default=100,
                        help='Maximum number of epochs for CNN training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation for CNN training')
    
    args = parser.parse_args()
    
    # Run main function
    best_model = main(args)('sample_image_pairs.png')
    plt.show()

# ==================================
# DATA LOADING FUNCTIONS
# ==================================

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

# ==================================
# ANN MODEL FUNCTIONS
# ==================================

def build_ann_model(input_shape, hyperparams=None):
    """
    Build an ANN model for sepia classification
    
    Args:
        input_shape: Shape of input data (flattened images)
        hyperparams: Dictionary of hyperparameters
    
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
    Train ANN model for sepia classification
    
    Args:
        X: Input data (flattened images)
        y: Labels
        hyperparams: Dictionary of hyperparameters
        epochs: Maximum number of epochs
        batch_size: Batch size
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print dataset information
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1]}")
    
    # Build model
    input_shape = X_train.shape[1]
    model = build_ann_model(input_shape, hyperparams)
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('best_ann_model.h5', 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        mode='max', 
                        verbose=1),
        EarlyStopping(monitor='val_loss', 
                      patience=10, 
                      restore_best_weights=True, 
                      verbose=1)
    ]
    
    # Train the model
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
    
    # Confusion matrix and classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ANN Confusion Matrix')
    plt.savefig('ann_confusion_matrix.png')
    plt.show()
    
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Sepia'])
    print("\nClassification Report:")
    print(report)
    
    return model, history, (X_test, y_test)

# ==================================
# CNN MODEL FUNCTIONS
# ==================================

def build_cnn_model(input_shape, hyperparams=None):
    """
    Build a CNN model for sepia classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        model: Compiled Keras model
    """
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 0.0001,
            'conv_filters': [32, 64, 128],
            'conv_kernel_size': 3,
            'dense_units': [256, 128],
            'dropout_rates': [0.25, 0.25, 0.25, 0.5, 0.3],
            'activation': 'relu',
            'use_batch_norm': True,
            'double_conv': True  # Whether to use two conv layers per block
        }
    
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(hyperparams['conv_filters'][0], 
                     (hyperparams['conv_kernel_size'], hyperparams['conv_kernel_size']), 
                     activation=hyperparams['activation'], 
                     input_shape=input_shape, 
                     padding='same',
                     name='conv1_1'))
    
    if hyperparams['double_conv']:
        model.add(Conv2D(hyperparams['conv_filters'][0], 
                         (hyperparams['conv_kernel_size'], hyperparams['conv_kernel_size']), 
                         activation=hyperparams['activation'], 
                         padding='same',
                         name='conv1_2'))
    
    if hyperparams['use_batch_norm']:
        model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2, 2)))
    
    if hyperparams['dropout_rates'][0] > 0:
        model.add(Dropout(hyperparams['dropout_rates'][0]))
    
    # Additional convolutional blocks
    for i in range(1, len(hyperparams['conv_filters'])):
        model.add(Conv2D(hyperparams['conv_filters'][i], 
                         (hyperparams['conv_kernel_size'], hyperparams['conv_kernel_size']), 
                         activation=hyperparams['activation'], 
                         padding='same',
                         name=f'conv{i+1}_1'))
        
        if hyperparams['double_conv']:
            model.add(Conv2D(hyperparams['conv_filters'][i], 
                             (hyperparams['conv_kernel_size'], hyperparams['conv_kernel_size']), 
                             activation=hyperparams['activation'], 
                             padding='same',
                             name=f'conv{i+1}_2'))
        
        if hyperparams['use_batch_norm']:
            model.add(BatchNormalization())
        
        model.add(MaxPooling2D((2, 2)))
        
        if i < len(hyperparams['dropout_rates']) and hyperparams['dropout_rates'][i] > 0:
            model.add(Dropout(hyperparams['dropout_rates'][i]))
    
    # Flatten and dense layers
    model.add(Flatten())
    
    for i, units in enumerate(hyperparams['dense_units']):
        model.add(Dense(units, activation=hyperparams['activation']))
        
        if hyperparams['use_batch_norm']:
            model.add(BatchNormalization())
        
        dropout_idx = len(hyperparams['conv_filters']) + i
        if dropout_idx < len(hyperparams['dropout_rates']) and hyperparams['dropout_rates'][dropout_idx] > 0:
            model.add(Dropout(hyperparams['dropout_rates'][dropout_idx]))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn_model(X, y, hyperparams=None, epochs=100, batch_size=32, use_augmentation=True):
    """
    Train CNN model for sepia classification
    
    Args:
        X: Input data (images)
        y: Labels
        hyperparams: Dictionary of hyperparameters
        epochs: Maximum number of epochs
        batch_size: Batch size
        use_augmentation: Whether to use data augmentation
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print dataset information
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1:]}")
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape, hyperparams)
    
    # Model summary
    model.summary()
    
    # Data augmentation
    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('best_cnn_model.h5', 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        mode='max', 
                        verbose=1),
        EarlyStopping(monitor='val_loss', 
                      patience=15, 
                      restore_best_weights=True, 
                      verbose=1),
        ReduceLROnPlateau(monitor='val_loss', 
                          factor=0.5, 
                          patience=5, 
                          min_lr=1e-6, 
                          verbose=1)
    ]
    
    # Train the model
    start_time = time.time()
    
    if use_augmentation:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
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
    plt.title('CNN Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_model_performance.png')
    plt.show()
    
    # Confusion matrix and classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CNN Confusion Matrix')
    plt.savefig('cnn_confusion_matrix.png')
    plt.show()
    
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Sepia'])
    print("\nClassification Report:")
    print(report)
    
    # Visualize feature maps for a sample image
    visualize_feature_maps(model, X_test[0])
    
    return model, history, (X_test, y_test)

def visualize_feature_maps(model, img, num_layers=2):
    """
    Visualize feature maps from convolutional layers
    
    Args:
        model: Trained CNN model
        img: Input image
        num_layers: Number of convolutional layers to visualize
    """
    # Get conv layer names
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name][:num_layers]
    
    for layer_name in conv_layers:
        try:
            # Create feature model
            feature_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = feature_model.predict(np.expand_dims(img, axis=0))
            
            # Plot feature maps
            num_filters = min(16, feature_maps.shape[3])  # Show at most 16 filters
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            for i, ax in enumerate(axes.flat):
                if i < num_filters:
                    ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
                    ax.set_title(f'Filter {i}')
                ax.axis('off')
            
            plt.suptitle(f'Feature maps for layer: {layer_name}')
            plt.tight_layout()
            plt.savefig(f'feature_maps_{layer_name}.png')
            plt.show()
        except Exception as e:
            print(f"Error visualizing layer {layer_name}: {e}")

# ==================================
# MODEL COMPARISON FUNCTIONS
# ==================================

def compare_models(ann_results, cnn_results):
    """
    Compare performance of ANN and CNN models
    
    Args:
        ann_results: Tuple of (model, history, (X_test, y_test)) for ANN
        cnn_results: Tuple of (model, history, (X_test, y_test)) for CNN
    """
    ann_model, ann_history, (X_test_ann, y_test_ann) = ann_results
    cnn_model, cnn_history, (X_test_cnn, y_test_cnn) = cnn_results
    
    # Evaluate models on test data
    ann_loss, ann_accuracy = ann_model.evaluate(X_test_ann, y_test_ann)
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
    
    # Get training times (approximate from history objects)
    ann_training_time = len(ann_history.history['loss']) * 0.1  # Rough estimate
    cnn_training_time = len(cnn_history.history['loss']) * 0.3  # Rough estimate
    
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
    
    # Compare confusion matrices
    ann_pred = (ann_model.predict(X_test_ann) > 0.5).astype(int).flatten()
    cnn_pred = (cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ANN confusion matrix
    ann_cm = confusion_matrix(y_test_ann, ann_pred)
    sns.heatmap(ann_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Sepia'],
                yticklabels=['Normal', 'Sepia'],
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('ANN Confusion Matrix')
    
    # CNN confusion matrix
    cnn_cm = confusion_matrix(y_test_cnn, cnn_pred)
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
    
    # Summary table
    summary = {
        'Model': ['ANN', 'CNN'],
        'Accuracy': [ann_accuracy, cnn_accuracy],
        'Loss': [ann_loss, cnn_loss],
        'Training Epochs': [len(ann_history.history['loss']), len(cnn_history.history['loss'])],
        'Parameters': [ann_model.count_params(), cnn_model.count_params()],
        'Training Time (s)': [ann_training_time, cnn_training_time]
    }
    
    # Print summary as table
    print("\n===== Model Comparison Summary =====")
    headers = list(summary.keys())
    rows = list(zip(*summary.values()))
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Save table as image
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig('model_comparison_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
