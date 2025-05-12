"""
Hyperparameter Tuning for Sepia Classification

This script performs hyperparameter tuning for both ANN and CNN models
to find the optimal architecture for sepia filter classification.

Usage:
    python hyperparameter_tuning.py --source_folder path/to/images

Author: Your Name
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import argparse
import time
from tabulate import tabulate
from tqdm import tqdm

# Import the main functions from our sepia classification script
from combined_main_script import (
    load_dataset, 
    create_sepia_dataset,
    build_ann_model,
    build_cnn_model
)

# Define hyperparameter grids
ANN_PARAM_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'hidden_layer_configs': [
        [512, 256, 128, 64],
        [256, 128, 64],
        [512, 256, 128]
    ],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [16, 32, 64],
}

CNN_PARAM_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'conv_filter_configs': [
        [32, 64, 128],
        [16, 32, 64],
        [32, 64, 128, 256]
    ],
    'double_conv': [True, False],
    'dropout_rate': [0.25, 0.4],
    'batch_size': [16, 32, 64],
}

def generate_ann_hyperparams(param_grid):
    """
    Generate all combinations of ANN hyperparameters
    
    Args:
        param_grid: Dictionary of hyperparameter lists
    
    Returns:
        List of hyperparameter dictionaries
    """
    hyperparams_list = []
    
    for lr in param_grid['learning_rate']:
        for hidden_layers in param_grid['hidden_layer_configs']:
            for dropout in param_grid['dropout_rate']:
                for batch_size in param_grid['batch_size']:
                    # Create dropout rates list matching the number of layers
                    dropout_rates = [dropout] * len(hidden_layers)
                    
                    hyperparams_list.append({
                        'hyperparams': {
                            'learning_rate': lr,
                            'hidden_layers': hidden_layers,
                            'dropout_rates': dropout_rates,
                            'activation': 'relu',
                            'use_batch_norm': True
                        },
                        'batch_size': batch_size
                    })
    
    return hyperparams_list

def generate_cnn_hyperparams(param_grid):
    """
    Generate all combinations of CNN hyperparameters
    
    Args:
        param_grid: Dictionary of hyperparameter lists
    
    Returns:
        List of hyperparameter dictionaries
    """
    hyperparams_list = []
    
    for lr in param_grid['learning_rate']:
        for conv_filters in param_grid['conv_filter_configs']:
            for double_conv in param_grid['double_conv']:
                for dropout in param_grid['dropout_rate']:
                    for batch_size in param_grid['batch_size']:
                        # Create dropout rates list matching the number of layers
                        dropout_rates = [dropout] * (len(conv_filters) + 2)  # +2 for dense layers
                        
                        hyperparams_list.append({
                            'hyperparams': {
                                'learning_rate': lr,
                                'conv_filters': conv_filters,
                                'conv_kernel_size': 3,
                                'dense_units': [128],
                                'dropout_rates': dropout_rates,
                                'activation': 'relu',
                                'use_batch_norm': True,
                                'double_conv': double_conv
                            },
                            'batch_size': batch_size
                        })
    
    return hyperparams_list

def tune_ann(X, y, param_grid, epochs=20, max_combinations=10):
    """
    Tune ANN hyperparameters
    
    Args:
        X: Input data
        y: Labels
        param_grid: Dictionary of hyperparameter lists
        epochs: Maximum number of epochs
        max_combinations: Maximum number of combinations to try
    
    Returns:
        best_params: Best hyperparameters
        results: All results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generate hyperparameter combinations
    hyperparams_list = generate_ann_hyperparams(param_grid)
    
    # Limit combinations if needed
    if len(hyperparams_list) > max_combinations:
        print(f"Limiting to {max_combinations} random combinations out of {len(hyperparams_list)}")
        np.random.shuffle(hyperparams_list)
        hyperparams_list = hyperparams_list[:max_combinations]
    
    # Store results
    results = []
    
    # Initialize best parameters
    best_val_acc = 0
    best_params = None
    
    # Iterate over hyperparameter combinations
    print(f"Testing {len(hyperparams_list)} ANN hyperparameter combinations")
    
    for i, param_dict in enumerate(tqdm(hyperparams_list)):
        hyperparams = param_dict['hyperparams']
        batch_size = param_dict['batch_size']
        
        # Build model
        input_shape = X_train.shape[1]
        model = build_ann_model(input_shape, hyperparams)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Store results
        results.append({
            'hyperparams': hyperparams,
            'batch_size': batch_size,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_train_acc': history.history['accuracy'][-1]
        })
        
        # Update best parameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'hyperparams': hyperparams,
                'batch_size': batch_size
            }
            
        # Print progress
        tqdm.write(f"Combination {i+1}/{len(hyperparams_list)}: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
    
    # Sort results by validation accuracy
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    # Print top results
    print("\nTop 5 ANN Hyperparameter Combinations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"Rank {i+1}:")
        print(f"  Learning Rate: {result['hyperparams']['learning_rate']}")
        print(f"  Hidden Layers: {result['hyperparams']['hidden_layers']}")
        print(f"  Dropout Rates: {result['hyperparams']['dropout_rates']}")
        print(f"  Batch Size: {result['batch_size']}")
        print(f"  Validation Accuracy: {result['val_acc']:.4f}")
        print(f"  Validation Loss: {result['val_loss']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Epochs Trained: {result['epochs_trained']}")
    
    # Print best parameters
    print(f"\nBest ANN hyperparameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_params, sorted_results

def tune_cnn(X, y, param_grid, epochs=10, max_combinations=10):
    """
    Tune CNN hyperparameters
    
    Args:
        X: Input data
        y: Labels
        param_grid: Dictionary of hyperparameter lists
        epochs: Maximum number of epochs
        max_combinations: Maximum number of combinations to try
    
    Returns:
        best_params: Best hyperparameters
        results: All results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generate hyperparameter combinations
    hyperparams_list = generate_cnn_hyperparams(param_grid)
    
    # Limit combinations if needed
    if len(hyperparams_list) > max_combinations:
        print(f"Limiting to {max_combinations} random combinations out of {len(hyperparams_list)}")
        np.random.shuffle(hyperparams_list)
        hyperparams_list = hyperparams_list[:max_combinations]
    
    # Store results
    results = []
    
    # Initialize best parameters
    best_val_acc = 0
    best_params = None
    
    # Iterate over hyperparameter combinations
    print(f"Testing {len(hyperparams_list)} CNN hyperparameter combinations")
    
    for i, param_dict in enumerate(tqdm(hyperparams_list)):
        hyperparams = param_dict['hyperparams']
        batch_size = param_dict['batch_size']
        
        # Build model
        input_shape = X_train.shape[1:]
        model = build_cnn_model(input_shape, hyperparams)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Store results
        results.append({
            'hyperparams': hyperparams,
            'batch_size': batch_size,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_train_acc': history.history['accuracy'][-1]
        })
        
        # Update best parameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'hyperparams': hyperparams,
                'batch_size': batch_size
            }
            
        # Print progress
        tqdm.write(f"Combination {i+1}/{len(hyperparams_list)}: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
    
    # Sort results by validation accuracy
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    # Print top results
    print("\nTop 5 CNN Hyperparameter Combinations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"Rank {i+1}:")
        print(f"  Learning Rate: {result['hyperparams']['learning_rate']}")
        print(f"  Conv Filters: {result['hyperparams']['conv_filters']}")
        print(f"  Double Conv: {result['hyperparams']['double_conv']}")
        print(f"  Dropout Rates: {result['hyperparams']['dropout_rates'][0]}")  # First dropout rate
        print(f"  Batch Size: {result['batch_size']}")
        print(f"  Validation Accuracy: {result['val_acc']:.4f}")
        print(f"  Validation Loss: {result['val_loss']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Epochs Trained: {result['epochs_trained']}")
    
    # Print best parameters
    print(f"\nBest CNN hyperparameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_params, sorted_results

def plot_hyperparameter_comparison(ann_results, cnn_results):
    """
    Plot hyperparameter comparison
    
    Args:
        ann_results: ANN tuning results
        cnn_results: CNN tuning results
    """
    # Sort results by validation accuracy
    ann_sorted = sorted(ann_results, key=lambda x: x['val_acc'], reverse=True)
    cnn_sorted = sorted(cnn_results, key=lambda x: x['val_acc'], reverse=True)
    
    # Take top 10 results
    ann_top10 = ann_sorted[:min(10, len(ann_sorted))]
    cnn_top10 = cnn_sorted[:min(10, len(cnn_sorted))]
    
    # Extract validation accuracy
    ann_accs = [result['val_acc'] for result in ann_top10]
    cnn_accs = [result['val_acc'] for result in cnn_top10]
    
    # Extract validation loss
    ann_losses = [result['val_loss'] for result in ann_top10]
    cnn_losses = [result['val_loss'] for result in cnn_top10]
    
    # Extract training time
    ann_times = [result['training_time'] for result in ann_top10]
    cnn_times = [result['training_time'] for result in cnn_top10]
    
    # Extract epochs trained
    ann_epochs = [result['epochs_trained'] for result in ann_top10]
    cnn_epochs = [result['epochs_trained'] for result in cnn_top10]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot validation accuracy
    plt.subplot(2, 2, 1)
    plt.bar(np.arange(len(ann_accs)) - 0.2, ann_accs, width=0.4, label='ANN')
    plt.bar(np.arange(len(cnn_accs)) + 0.2, cnn_accs, width=0.4, label='CNN')
    plt.xlabel('Model Rank')
    plt.ylabel('Validation Accuracy')
    plt.title('Top 10 Models Validation Accuracy')
    plt.legend()
    plt.xticks(np.arange(len(ann_accs)), np.arange(1, len(ann_accs) + 1))
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.bar(np.arange(len(ann_losses)) - 0.2, ann_losses, width=0.4, label='ANN')
    plt.bar(np.arange(len(cnn_losses)) + 0.2, cnn_losses, width=0.4, label='CNN')
    plt.xlabel('Model Rank')
    plt.ylabel('Validation Loss')
    plt.title('Top 10 Models Validation Loss')
    plt.legend()
    plt.xticks(np.arange(len(ann_losses)), np.arange(1, len(ann_losses) + 1))
    
    # Plot training time
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(len(ann_times)) - 0.2, ann_times, width=0.4, label='ANN')
    plt.bar(np.arange(len(cnn_times)) + 0.2, cnn_times, width=0.4, label='CNN')
    plt.xlabel('Model Rank')
    plt.ylabel('Training Time (seconds)')
    plt.title('Top 10 Models Training Time')
    plt.legend()
    plt.xticks(np.arange(len(ann_times)), np.arange(1, len(ann_times) + 1))
    
    # Plot epochs trained
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(len(ann_epochs)) - 0.2, ann_epochs, width=0.4, label='ANN')
    plt.bar(np.arange(len(cnn_epochs)) + 0.2, cnn_epochs, width=0.4, label='CNN')
    plt.xlabel('Model Rank')
    plt.ylabel('Epochs Trained')
    plt.title('Top 10 Models Epochs Trained')
    plt.legend()
    plt.xticks(np.arange(len(ann_epochs)), np.arange(1, len(ann_epochs) + 1))
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png')
    plt.show()
    
    # Create a figure for learning rate comparison
    plt.figure(figsize=(10, 5))
    
    # Extract learning rates and corresponding accuracies
    ann_lr = [result['hyperparams']['learning_rate'] for result in ann_results]
    ann_lr_acc = [result['val_acc'] for result in ann_results]
    
    cnn_lr = [result['hyperparams']['learning_rate'] for result in cnn_results]
    cnn_lr_acc = [result['val_acc'] for result in cnn_results]
    
    # Group by learning rate
    ann_lr_groups = {}
    for lr, acc in zip(ann_lr, ann_lr_acc):
        if lr not in ann_lr_groups:
            ann_lr_groups[lr] = []
        ann_lr_groups[lr].append(acc)
    
    cnn_lr_groups = {}
    for lr, acc in zip(cnn_lr, cnn_lr_acc):
        if lr not in cnn_lr_groups:
            cnn_lr_groups[lr] = []
        cnn_lr_groups[lr].append(acc)
    
    # Calculate means and stds
    ann_lr_means = [np.mean(ann_lr_groups[lr]) for lr in sorted(ann_lr_groups.keys())]
    ann_lr_stds = [np.std(ann_lr_groups[lr]) for lr in sorted(ann_lr_groups.keys())]
    
    cnn_lr_means = [np.mean(cnn_lr_groups[lr]) for lr in sorted(cnn_lr_groups.keys())]
    cnn_lr_stds = [np.std(cnn_lr_groups[lr]) for lr in sorted(cnn_lr_groups.keys())]
    
    # Plot learning rate comparison
    x = np.arange(len(sorted(ann_lr_groups.keys())))
    width = 0.35
    
    plt.bar(x - width/2, ann_lr_means, width, label='ANN', yerr=ann_lr_stds, capsize=5)
    plt.bar(x + width/2, cnn_lr_means, width, label='CNN', yerr=cnn_lr_stds, capsize=5)
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Validation Accuracy')
    plt.title('Effect of Learning Rate on Validation Accuracy')
    plt.xticks(x, sorted([str(lr) for lr in ann_lr_groups.keys()]))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png')
    plt.show()
    
    # Create a table of the best hyperparameters
    ann_best = ann_sorted[0]
    cnn_best = cnn_sorted[0]
    
    best_params_table = {
        'Model': ['ANN', 'CNN'],
        'Learning Rate': [ann_best['hyperparams']['learning_rate'], cnn_best['hyperparams']['learning_rate']],
        'Batch Size': [ann_best['batch_size'], cnn_best['batch_size']],
        'Validation Accuracy': [ann_best['val_acc'], cnn_best['val_acc']],
        'Validation Loss': [ann_best['val_loss'], cnn_best['val_loss']],
        'Training Time (s)': [ann_best['training_time'], cnn_best['training_time']],
        'Epochs Trained': [ann_best['epochs_trained'], cnn_best['epochs_trained']]
    }
    
    # Print table
    print("\nBest Hyperparameters Comparison:")
    headers = list(best_params_table.keys())
    rows = list(zip(*best_params_table.values()))
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
    plt.savefig('best_hyperparameters.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Return best hyperparameters
    return ann_best, cnn_best

def main(args):
    """
    Main function
    
    Args:
        args: Command line arguments
    """
    # Print welcome message
    print("="*60)
    print("HYPERPARAMETER TUNING FOR SEPIA CLASSIFICATION")
    print("="*60)
    
    # Create dataset if source folder is provided
    if args.source_folder:
        print(f"\n===== Creating dataset from {args.source_folder} =====")
        create_sepia_dataset(args.source_folder, args.num_samples, (args.img_size, args.img_size))
    
    # Load dataset for ANN
    print("\n===== Loading data for ANN =====")
    X_ann, y_ann = load_dataset(flatten_for_ann=True, img_size=(args.img_size, args.img_size))
    
    # Load dataset for CNN
    print("\n===== Loading data for CNN =====")
    X_cnn, y_cnn = load_dataset(flatten_for_ann=False, img_size=(args.img_size, args.img_size))
    
    # Tune ANN
    print("\n===== Tuning ANN Hyperparameters =====")
    ann_best, ann_results = tune_ann(
        X_ann, y_ann,
        ANN_PARAM_GRID,
        epochs=args.max_epochs,
        max_combinations=args.max_combinations
    )
    
    # Tune CNN
    print("\n===== Tuning CNN Hyperparameters =====")
    cnn_best, cnn_results = tune_cnn(
        X_cnn, y_cnn,
        CNN_PARAM_GRID,
        epochs=args.max_epochs,
        max_combinations=args.max_combinations
    )
    
    # Compare results
    print("\n===== Comparing Results =====")
    ann_best_params, cnn_best_params = plot_hyperparameter_comparison(ann_results, cnn_results)
    
    # Save best hyperparameters to a file
    with open('best_hyperparameters.txt', 'w') as f:
        f.write("Best ANN Hyperparameters:\n")
        f.write(f"Learning Rate: {ann_best_params['hyperparams']['learning_rate']}\n")
        f.write(f"Hidden Layers: {ann_best_params['hyperparams']['hidden_layers']}\n")
        f.write(f"Dropout Rates: {ann_best_params['hyperparams']['dropout_rates']}\n")
        f.write(f"Batch Size: {ann_best_params['batch_size']}\n")
        f.write(f"Validation Accuracy: {ann_best_params['val_acc']:.4f}\n")
        f.write(f"Validation Loss: {ann_best_params['val_loss']:.4f}\n")
        f.write("\n")
        f.write("Best CNN Hyperparameters:\n")
        f.write(f"Learning Rate: {cnn_best_params['hyperparams']['learning_rate']}\n")
        f.write(f"Conv Filters: {cnn_best_params['hyperparams']['conv_filters']}\n")
        f.write(f"Double Conv: {cnn_best_params['hyperparams']['double_conv']}\n")
        f.write(f"Dropout Rates: {cnn_best_params['hyperparams']['dropout_rates'][0]}\n")
        f.write(f"Batch Size: {cnn_best_params['batch_size']}\n")
        f.write(f"Validation Accuracy: {cnn_best_params['val_acc']:.4f}\n")
        f.write(f"Validation Loss: {cnn_best_params['val_loss']:.4f}\n")
    
    print("\n===== All Done! =====")
    print(f"Best hyperparameters saved to: best_hyperparameters.txt")
    print(f"Hyperparameter comparison visualizations saved to current directory")
    
    return ann_best_params, cnn_best_params

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Sepia Classification')
    
    parser.add_argument('--source_folder', type=str, default=None,
                        help='Path to folder with source images')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Maximum number of image pairs to create')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size to resize images to')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Maximum number of epochs for training')
    parser.add_argument('--max_combinations', type=int, default=10,
                        help='Maximum number of hyperparameter combinations to try')
    
    args = parser.parse_args()
    
    # Run main function
    ann_best, cnn_best = main(args)